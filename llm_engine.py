"""Thin vLLM wrapper, tolerant to vLLM 0.6 ↔ 0.20+ API churn.

Two breaking changes we adapt around:

1) `guided_decoding_backend` was removed from `EngineArgs` in vLLM ≥0.10.
   xgrammar is the default backend; we just drop the kwarg on newer installs.
2) `GuidedDecodingParams` was renamed to `StructuredOutputsParams`, and
   `SamplingParams.guided_decoding` → `SamplingParams.structured_outputs`
   (vLLM ≥0.10). We detect which API is available and use it.
"""
from __future__ import annotations
import inspect
from typing import List, Optional


# ---------- vLLM API detection ----------

def _detect_engine_kwargs() -> Optional[set]:
    """Return the set of EngineArgs field names, or None if introspection fails.
    Used to drop unknown kwargs (e.g. `guided_decoding_backend` on vLLM ≥0.10)."""
    try:
        from vllm.engine.arg_utils import EngineArgs
        import dataclasses
        if dataclasses.is_dataclass(EngineArgs):
            return {f.name for f in dataclasses.fields(EngineArgs)}
        sig = inspect.signature(EngineArgs.__init__)
        return set(sig.parameters.keys())
    except Exception:
        return None


def _try_import(module_paths, class_name):
    """Try to import `class_name` from each module path in order. Return the class
    or None. Used to locate vLLM classes that move between releases."""
    for mod_path in module_paths:
        try:
            mod = __import__(mod_path, fromlist=[class_name])
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    return None


def _detect_structured_outputs_api():
    """Probe-based detection: try constructing SamplingParams with each known API
    until one works. Robust to Pydantic-style SamplingParams (vLLM ≥0.20) where
    inspect.signature doesn't enumerate fields.

    Returns (Cls, sp_field_name) — pass `Cls(json=schema)` as `sp_field_name=` to
    SamplingParams. Returns (None, None) if no API is available.
    """
    from vllm import SamplingParams
    dummy_schema = {"type": "object", "properties": {}, "required": []}

    # Try new API first (vLLM ≥0.10/0.20): StructuredOutputsParams + structured_outputs
    StructuredOutputsParams = _try_import(
        [
            "vllm.sampling_params",
            "vllm",
            "vllm.v1.sampling_params",
            "vllm.v1.structured_output",
        ],
        "StructuredOutputsParams",
    )
    if StructuredOutputsParams is not None:
        try:
            sop = StructuredOutputsParams(json=dummy_schema)
            SamplingParams(structured_outputs=sop, max_tokens=1)
            return (StructuredOutputsParams, "structured_outputs")
        except Exception as e:
            print(f"[vllm-runner] StructuredOutputsParams probe failed: {type(e).__name__}: {e}")

    # Try old API (vLLM <0.10): GuidedDecodingParams + guided_decoding
    GuidedDecodingParams = _try_import(
        ["vllm.sampling_params", "vllm"], "GuidedDecodingParams",
    )
    if GuidedDecodingParams is not None:
        try:
            gdp = GuidedDecodingParams(json=dummy_schema)
            SamplingParams(guided_decoding=gdp, max_tokens=1)
            return (GuidedDecodingParams, "guided_decoding")
        except Exception as e:
            print(f"[vllm-runner] GuidedDecodingParams probe failed: {type(e).__name__}: {e}")

    return (None, None)


# ---------- runner ----------

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
            disable_log_stats=True,
        )

        # Drop any kwarg the installed EngineArgs no longer accepts.
        engine_kwargs = _detect_engine_kwargs()
        if engine_kwargs is not None:
            dropped = {k for k in wanted if k not in engine_kwargs}
            if dropped:
                print(f"[vllm-runner] dropping unsupported kwargs: {sorted(dropped)}")
            wanted = {k: v for k, v in wanted.items() if k in engine_kwargs}

        self.llm = LLM(**wanted)

        # Pick structured-outputs API for this vLLM version.
        self._SOP, self._SO_FIELD = _detect_structured_outputs_api()
        if self._SOP is None:
            print("[vllm-runner] WARN: no structured-outputs API detected; "
                  "guided_json will be ignored.")
        else:
            print(f"[vllm-runner] structured outputs: "
                  f"SamplingParams(... {self._SO_FIELD}={self._SOP.__name__}(json=...))")

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

        sp_kwargs = dict(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            stop=None,
        )
        if guided_json is not None and self._SOP is not None:
            sp_kwargs[self._SO_FIELD] = self._SOP(json=guided_json)

        sp = SamplingParams(**sp_kwargs)
        templated = [self._apply_template(p) for p in prompts]
        outputs = self.llm.generate(templated, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def shutdown(self) -> None:
        try:
            del self.llm
        except Exception:
            pass
