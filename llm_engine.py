"""Thin vLLM wrapper:
- single LLM engine, internal continuous batching
- chat template applied via tokenizer
- optional global guided_json schema (xgrammar backend, compiled once + cached)
"""
from __future__ import annotations
from typing import List, Optional


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

        self.llm = LLM(
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
            guided_decoding_backend="xgrammar",
            disable_log_stats=True,
        )

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
        from vllm.sampling_params import GuidedDecodingParams

        gdp = GuidedDecodingParams(json=guided_json) if guided_json else None
        sp = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            guided_decoding=gdp,
            stop=None,
        )
        templated = [self._apply_template(p) for p in prompts]
        # use_tqdm shows token-level progress for the whole batch
        outputs = self.llm.generate(templated, sp, use_tqdm=False)
        # vLLM returns in the SAME order as inputs
        return [o.outputs[0].text for o in outputs]

    def shutdown(self) -> None:
        try:
            del self.llm
        except Exception:
            pass
