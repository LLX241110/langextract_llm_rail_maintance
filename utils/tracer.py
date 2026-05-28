"""
LLM调用追踪器：基于 LangChain BaseCallbackHandler 实现
记录每次 LLM 调用的 step、model、latency、tokens、费用
输出到 traces/<run_id>.jsonl，方便面试展示可观测性能力
"""
import time
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# 主流模型每1K token定价（USD），面试常问到cost估算
COST_PER_1K_TOKENS = {
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "gpt-4":          {"prompt": 0.03,   "completion": 0.06},
    "gpt-4o":         {"prompt": 0.005,  "completion": 0.015},
    "gpt-4o-mini":    {"prompt": 0.00015,"completion": 0.0006},
}


class LLMCallTracer(BaseCallbackHandler):
    """
    LangChain 回调处理器，自动追踪每次 LLM 调用。

    挂载方式：传给 ChatOpenAI(callbacks=[tracer])
    调用链路：on_llm_start → [LLM执行] → on_llm_end / on_llm_error

    面试考点：
    - LangChain Callback 机制（同步/异步、chain/llm/tool 级别）
    - Token 用量追踪与成本估算
    - JSONL 格式适合流式写入、grep 检索
    """

    def __init__(self, run_id: str, traces_dir: Path, step_name: str = "init"):
        super().__init__()
        self.run_id = run_id
        self.traces_dir = traces_dir
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.step_name = step_name
        self._pending: Dict[str, Dict] = {}   # call_id → 调用开始信息
        self.records: List[Dict] = []         # 本次运行全部记录（内存）

    def set_step(self, step_name: str):
        """在不同 pipeline 步骤间切换，打标签用于追踪归因"""
        self.step_name = step_name

    # ── LangChain 回调三件套 ──────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        **kwargs,
    ):
        self._pending[str(run_id)] = {
            "start_time": time.time(),
            "step": self.step_name,
        }

    def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs):
        call_id = str(run_id)
        pending = self._pending.pop(call_id, {})
        latency_ms = int((time.time() - pending.get("start_time", time.time())) * 1000)

        # 提取 token 用量（OpenAI 返回在 llm_output.token_usage）
        usage = {}
        model = "unknown"
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            model = response.llm_output.get("model_name", "unknown")

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

        record = {
            "call_id":           call_id,
            "run_id":            self.run_id,
            "step":              pending.get("step", self.step_name),
            "model":             model,
            "timestamp":         datetime.now().isoformat(),
            "latency_ms":        latency_ms,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "cost_usd":          cost,
            "status":            "success",
        }
        self._persist(record)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: uuid.UUID,
        **kwargs,
    ):
        call_id = str(run_id)
        pending = self._pending.pop(call_id, {})
        latency_ms = int((time.time() - pending.get("start_time", time.time())) * 1000)

        record = {
            "call_id":           call_id,
            "run_id":            self.run_id,
            "step":              pending.get("step", self.step_name),
            "model":             "unknown",
            "timestamp":         datetime.now().isoformat(),
            "latency_ms":        latency_ms,
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
            "cost_usd":          0.0,
            "status":            f"error: {str(error)[:120]}",
        }
        self._persist(record)

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rates = next(
            (v for k, v in COST_PER_1K_TOKENS.items() if k in model.lower()),
            None,
        )
        if not rates:
            return 0.0
        return round(
            (prompt_tokens / 1000) * rates["prompt"]
            + (completion_tokens / 1000) * rates["completion"],
            6,
        )

    def _persist(self, record: Dict):
        """内存记录 + 追加写入 JSONL（崩溃也不丢数据）"""
        self.records.append(record)
        trace_file = self.traces_dir / f"{self.run_id}.jsonl"
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 汇总展示 ─────────────────────────────────────────────────────────

    def print_summary(self):
        """流水线结束后打印调用汇总表，面试现场直接展示"""
        if not self.records:
            print("  [tracer] 无LLM调用记录")
            return

        total   = len(self.records)
        success = sum(1 for r in self.records if r["status"] == "success")
        t_lat   = sum(r["latency_ms"] for r in self.records)
        t_in    = sum(r["prompt_tokens"] for r in self.records)
        t_out   = sum(r["completion_tokens"] for r in self.records)
        t_cost  = sum(r["cost_usd"] for r in self.records)

        W = 72
        print(f"\n{'─' * W}")
        print(f"  LLM 调用追踪摘要   Run ID: {self.run_id}")
        print(f"{'─' * W}")
        print(f"  调用次数   {total} 次  (成功 {success} / 失败 {total - success})")
        print(f"  总耗时     {t_lat / 1000:.2f} s")
        print(f"  输入tokens {t_in:,}   输出tokens {t_out:,}   合计 {t_in+t_out:,}")
        print(f"  预估费用   ${t_cost:.6f} USD")
        print(f"{'─' * W}")
        print(f"  {'#':<4} {'步骤':<22} {'模型':<18} {'耗时(s)':>8} {'输入':>8} {'输出':>8}  状态")
        print(f"  {'─' * 70}")
        for i, r in enumerate(self.records, 1):
            icon = "✓" if r["status"] == "success" else "✗"
            model_short = r["model"].replace("gpt-", "").replace("-turbo", "")
            print(
                f"  {i:<4} {r['step']:<22} {model_short:<18}"
                f" {r['latency_ms']/1000:>7.2f}s"
                f" {r['prompt_tokens']:>8,}"
                f" {r['completion_tokens']:>8,}"
                f"  {icon}"
            )
        print(f"{'─' * W}")
        trace_file = self.traces_dir / f"{self.run_id}.jsonl"
        print(f"  JSONL 已写入: {trace_file}\n")
