"""
流水线断点续跑（Checkpoint）
按步骤将中间结果序列化到 checkpoints/<run_id>/ 目录。
支持 --resume 从上次失败的步骤继续，避免重复调 LLM 浪费费用。

面试考点：
- LLM pipeline 容错设计（网络抖动/限流后恢复）
- Pydantic model_dump / model_validate 序列化往返
- 增量构建 vs 全量重跑的工程 tradeoff
"""
import json
from pathlib import Path
from typing import Any, List, Optional

from schema.knowledge_schema import ExtractionResult

_CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"


class CheckpointManager:
    """
    每个 pipeline run 对应一个子目录，每个 step 对应一个 JSON 文件。
    文件名格式：step1_docs.json、step2_extracted.json、step3_aligned.json
    """

    def __init__(self, run_id: str, base_dir: Path = _CHECKPOINTS_DIR):
        self.run_id = run_id
        self.run_dir = base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── 核心三接口 ────────────────────────────────────────────────────────

    def save(self, step: str, data: Any) -> Path:
        """序列化并保存。ExtractionResult 用 Pydantic model_dump()，其余直接 json.dump。"""
        path = self.run_dir / f"{step}.json"
        if isinstance(data, ExtractionResult):
            payload = data.model_dump()
        else:
            payload = data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  [checkpoint] saved  {step} → {path.relative_to(path.parent.parent)}")
        return path

    def load(self, step: str, as_extraction_result: bool = False) -> Optional[Any]:
        """反序列化。as_extraction_result=True 时用 Pydantic model_validate 还原类型。"""
        path = self.run_dir / f"{step}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [checkpoint] loaded {step} ← {path.relative_to(path.parent.parent)}")
        if as_extraction_result:
            return ExtractionResult.model_validate(data)
        return data

    def exists(self, step: str) -> bool:
        return (self.run_dir / f"{step}.json").exists()

    # ── 运维工具 ─────────────────────────────────────────────────────────

    @classmethod
    def list_runs(cls, base_dir: Path = _CHECKPOINTS_DIR) -> List[dict]:
        """列出所有历史 run，方便 --list-runs 参数展示"""
        if not base_dir.exists():
            return []
        runs = []
        for d in sorted(base_dir.iterdir()):
            if d.is_dir():
                steps = sorted(f.stem for f in d.glob("*.json"))
                runs.append({"run_id": d.name, "steps": steps})
        return runs

    @classmethod
    def latest_run_id(cls, base_dir: Path = _CHECKPOINTS_DIR) -> Optional[str]:
        runs = cls.list_runs(base_dir)
        return runs[-1]["run_id"] if runs else None
