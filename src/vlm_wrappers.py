"""Unified VLM interface.

HFVLM        : any HF image-text-to-text model. Supports free-form
               generation AND first-token logit scoring (Yes/No, A/B/C).
OpenAIVLM    : OpenAI-compatible API (GPT-4o etc.). Generation only —
               logit strategies are skipped automatically for this backend.

The generic HF chat-template path (messages -> apply_chat_template ->
processor(text, images)) works for llava-hf, Qwen2-VL, Qwen2.5-VL,
InternVL3-hf, and Llama-3.2-Vision without per-family special cases.
"""

from __future__ import annotations

import base64
import io
import math
import os

from PIL import Image

try:  # torch only needed for the HF backend; analysis machines may lack it
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _inference(fn):
    """Run the wrapped method under torch.inference_mode()."""
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)
    return wrapper


def _pil(path) -> Image.Image:
    return Image.open(path).convert("RGB")


class HFVLM:
    supports_logits = True

    def __init__(self, hf_id: str, load_in_4bit=True, torch_dtype="float16",
                 device_map="auto", max_new_tokens=256):
        if torch is None:
            raise RuntimeError("torch is required for the HF backend — "
                               "pip install torch (data loading and "
                               "analysis do not need it).")
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                  BitsAndBytesConfig)
        self.hf_id = hf_id
        self.max_new_tokens = max_new_tokens
        dtype = getattr(torch, torch_dtype)
        kwargs = dict(torch_dtype=dtype, device_map=device_map)
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModelForImageTextToText.from_pretrained(hf_id,
                                                                 **kwargs)
        self.model.eval()
        tok = self.processor.tokenizer
        self._token_sets = {}
        for label, variants in {
            "Yes": ["Yes", "yes", " Yes", " yes"],
            "No": ["No", "no", " No", " no"],
            "A": ["A", " A"], "B": ["B", " B"], "C": ["C", " C"],
        }.items():
            ids = set()
            for v in variants:
                enc = tok.encode(v, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            self._token_sets[label] = sorted(ids)

    def _inputs(self, image_path, prompt: str):
        messages = [{"role": "user",
                     "content": [{"type": "image"},
                                 {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[_pil(image_path)],
                                return_tensors="pt")
        return inputs.to(self.model.device)

    @_inference
    def generate(self, image_path, prompt: str) -> str:
        inputs = self._inputs(image_path, prompt)
        out = self.model.generate(**inputs,
                                  max_new_tokens=self.max_new_tokens,
                                  do_sample=False, temperature=None,
                                  top_p=None, top_k=None)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new_tokens,
                                               skip_special_tokens=True)

    @_inference
    def _first_token_logits(self, image_path, prompt: str) -> torch.Tensor:
        inputs = self._inputs(image_path, prompt)
        logits = self.model(**inputs).logits[0, -1, :].float()
        return logits

    def _set_logit(self, logits: torch.Tensor, label: str) -> float:
        ids = self._token_sets[label]
        return max(logits[i].item() for i in ids)

    def yes_no_score(self, image_path, question: str) -> float:
        """S = softmax over (max Yes-variant logit, max No-variant logit).
        Implements Eq. (1) of the manuscript."""
        logits = self._first_token_logits(image_path, question)
        ly = self._set_logit(logits, "Yes")
        ln = self._set_logit(logits, "No")
        return 1.0 / (1.0 + math.exp(ln - ly))

    def option_scores(self, image_path, question: str,
                      letters: list[str]) -> dict[str, float]:
        logits = self._first_token_logits(image_path, question)
        raw = {l: self._set_logit(logits, l) for l in letters}
        m = max(raw.values())
        exps = {l: math.exp(v - m) for l, v in raw.items()}
        z = sum(exps.values())
        return {l: v / z for l, v in exps.items()}

    def unload(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()


class OpenAIVLM:
    supports_logits = False

    def __init__(self, api_model: str, max_new_tokens=256):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
        self.api_model = api_model
        self.max_new_tokens = max_new_tokens

    def generate(self, image_path, prompt: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        # re-encode oversized images down to jpeg to keep payloads small
        if len(b64) > 15_000_000:
            img = _pil(image_path)
            img.thumbnail((1536, 1536))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()
        resp = self.client.chat.completions.create(
            model=self.api_model,
            max_tokens=self.max_new_tokens,
            temperature=0,
            seed=42,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return resp.choices[0].message.content or ""

    def unload(self):
        pass


def build_model(name: str, mcfg: dict, rt: dict):
    if mcfg["backend"] == "hf":
        return HFVLM(mcfg["hf_id"],
                     load_in_4bit=rt.get("load_in_4bit", True),
                     torch_dtype=rt.get("torch_dtype", "float16"),
                     device_map=rt.get("device_map", "auto"),
                     max_new_tokens=rt.get("max_new_tokens", 256))
    if mcfg["backend"] == "openai":
        return OpenAIVLM(mcfg["api_model"],
                         max_new_tokens=rt.get("max_new_tokens", 256))
    raise ValueError(f"Unknown backend for model '{name}'")
