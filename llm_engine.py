"""Thin vLLM wrapper:
- single LLM engine, internal continuous batching
- chat template applied via tokenizer
- optional global guided_json schema (xgrammar backend, compiled once + cached)

Tolerant to vLLM API churn:
  * `guided_decoding_backend` kwarg was removed from EngineArgs in vLLM ≥0.8.
    xgrammar is the default backend now; we just stop passing the kwarg.
  * `GuidedDecodingParams` import path varies between releases — we try the new
    location first, then fall back.
"""
from __future__ import annotations
import inspect
from typing import List, Optional


def _filter_supported(cls, kwargs: dict) -> dict:
    """Drop kwargs that the installed vLLM no longer accepts."""
    try:
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys())
        if "kwargs" in allowed or "args" in allowed:
            return kwargs
        return {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        return kwargs


def _import_guided_decoding_params():
    """vLLM moved the class around across versions; try in priority order."""
    last_err = None
    for path in (
        ("vllm.sampling_params", "GuidedDecodingParams"),
        ("vllm",                 "GuidedDecodingParams"),
    ):
        try:
            mod = __import__(path[0], fromlist=[path[1]])
            return getattr(mod, path[1])
        except (ImportError, AttributeError) as e:
            last_err = e
    raise ImportError(
        f"Could not import GuidedDecodingParams from any known location: {last_err}"
    )


class VLLMRunner:
    def __init__(self, cfg: dict):
        from vllm import LLM
        from transformers import AutoTokenizer

        llm_cfg = cfg["llm"]
        self.model_name = llm_cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=llm_cfg.get("trust_remote_code", True)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build LLM kwargs, then drop anything the installed vLLM doesn't accept.
        wanted = dict(
            model=self.model_name,
            quantization=llm_cfg.get("quantization"),
            dtype=llm_cfg.get("dtype", "auto"),
            max_model_len=llm_cfg.get("max_model_len", 2048),
            gpu_memory_utilization=llm_cfg.get("gpu_memory_utilization", 0.82),
            tensor_parallel_size=llm_cfg.get("tensor_parallel_size", 1),
            enforce_eager=llm_cfg.get("enforce_eager", False),
            max_num_seqs=llm_cfg.get("max_num_seqs", 256),
            max_num_batched_tokens=llm_cfg.get("max_num_batched_tokens", 8192),
            swap_space=llm_cfg.get("swap_space", 4),
            trust_remote_code=llm_cfg.get("trust_remote_code", True),
            # NOTE: removed in vLLM ≥0.8 (xgrammar is the default). _filter_supported
            # drops it on newer installs; older installs still pick it up.
            guided_decoding_backend="xgrammar",
            disable_log_stats=True,
        )
        wanted = _filter_supported(LLM, wanted)
        self.llm = LLM(**wanted)

        # Resolve the GuidedDecodingParams class once.
        self._GuidedDecodingParams = _import_guided_decoding_params()

    def _apply_template(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_text(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.0,
        guided_json: Optional[dict] = None,
    ) -> List[str]:
        if not prompts:
            return []
        from vllm import SamplingParams

        gdp = self._GuidedDecodingParams(json=guided_json) if guided_json else None
        sp = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            guided_decoding=gdp,
            stop=None,
        )
        templated = [self._apply_template(p) for p in prompts]
        outputs = self.llm.generate(templated, sp, use_tqdm=False)
        # vLLM returns in the SAME order as inputs.
        return [o.outputs[0].text for o in outputs]

    def shutdown(self) -> None:
        try:
            del self.llm
        except Exception:
            pass
