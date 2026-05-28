"""
钢轨伤损运维知识图谱构建系统
主流程：非结构化文本 → LLM抽取 → 实体对齐 → Neo4j图谱 → 问答辅助

新增能力：
  --resume [run_id]   从上次断点继续，跳过已完成步骤（节省LLM费用）
  --list-runs         查看所有历史run及其checkpoint状态
"""
import sys
import os
import uuid
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraction.text_loader import load_all_documents, split_text_into_chunks
from extraction.llm_extractor import LLMExtractor
from extraction.entity_aligner import align_and_deduplicate
from graph.neo4j_client import Neo4jClient
from graph.graph_builder import GraphBuilder
from qa.knowledge_qa import KnowledgeQA
from schema.knowledge_schema import ExtractionResult
from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.checkpoint import CheckpointManager
from utils.tracer import LLMCallTracer

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
TRACES_DIR  = Path(__file__).parent / "traces"


def make_run_id() -> str:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{ts}_{uid}"


# ── pipeline 步骤 ──────────────────────────────────────────────────────────

def step1_load_documents(ckpt: CheckpointManager) -> list:
    print("\n=== Step 1: 加载运维文档 ===")
    if ckpt.exists("step1_docs"):
        docs = ckpt.load("step1_docs")
        print(f"  [跳过] 已从 checkpoint 恢复，共 {len(docs)} 个文档")
        return docs
    docs = load_all_documents(DATA_DIR)
    print(f"  共加载 {len(docs)} 个文档")
    ckpt.save("step1_docs", docs)
    return docs


def step2_extract_knowledge(
    docs: list,
    ckpt: CheckpointManager,
    tracer: LLMCallTracer,
) -> ExtractionResult:
    print("\n=== Step 2: LLM结构化抽取 ===")
    if ckpt.exists("step2_extracted"):
        result = ckpt.load("step2_extracted", as_extraction_result=True)
        print("  [跳过] 已从 checkpoint 恢复")
        return result

    tracer.set_step("step2_extract")
    extractor = LLMExtractor(callbacks=[tracer])
    merged_result = ExtractionResult()

    for doc in docs:
        print(f"\n  处理文档: {doc['filename']}")
        chunks = split_text_into_chunks(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        doc_result = extractor.extract_from_document(
            doc["content"], doc["filename"], chunks
        )
        merged_result.damage_types.extend(doc_result.damage_types)
        merged_result.maintenance_methods.extend(doc_result.maintenance_methods)
        merged_result.wear_limits.extend(doc_result.wear_limits)
        merged_result.maintenance_cycles.extend(doc_result.maintenance_cycles)
        merged_result.applicable_conditions.extend(doc_result.applicable_conditions)
        merged_result.damage_maintenance_relations.extend(doc_result.damage_maintenance_relations)
        merged_result.wear_limit_trigger_relations.extend(doc_result.wear_limit_trigger_relations)

    print(f"\n  抽取完成（去重前）：")
    print(f"    伤损类型: {len(merged_result.damage_types)}")
    print(f"    维修方式: {len(merged_result.maintenance_methods)}")
    print(f"    磨耗限值: {len(merged_result.wear_limits)}")
    print(f"    维修周期: {len(merged_result.maintenance_cycles)}")
    print(f"    适用条件: {len(merged_result.applicable_conditions)}")
    print(f"    伤损-维修关系: {len(merged_result.damage_maintenance_relations)}")

    ckpt.save("step2_extracted", merged_result)
    return merged_result


def step3_align_entities(
    result: ExtractionResult,
    ckpt: CheckpointManager,
) -> ExtractionResult:
    print("\n=== Step 3: 实体对齐与去重 ===")
    if ckpt.exists("step3_aligned"):
        aligned = ckpt.load("step3_aligned", as_extraction_result=True)
        print("  [跳过] 已从 checkpoint 恢复")
        return aligned
    aligned = align_and_deduplicate(result)
    ckpt.save("step3_aligned", aligned)
    return aligned


def step4_build_graph(result: ExtractionResult):
    print("\n=== Step 4: 构建Neo4j知识图谱 ===")
    client = Neo4jClient()

    if not client.verify_connection():
        print("  Neo4j连接失败，跳过图谱构建")
        print("  请确认Neo4j已启动：bolt://localhost:7687")
        client.close()
        return None, None

    print("  Neo4j连接成功")
    client.create_constraints()
    client.clear_database()

    builder = GraphBuilder(client)
    builder.build_graph(result)

    stats = builder.get_graph_stats()
    print(f"\n  图谱统计：")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    return client, builder


def step5_demo_qa(client: Neo4jClient, tracer: LLMCallTracer):
    print("\n=== Step 5: 问答辅助演示 ===")
    if client is None:
        print("  Neo4j未连接，跳过问答演示")
        return

    tracer.set_step("step5_qa")
    qa = KnowledgeQA(client, callbacks=[tracer])
    questions = [
        "钢轨打磨的工艺流程是什么？需要什么设备？",
        "高速铁路的轨头垂直磨耗限值是多少？",
        "发现钢轨波浪形磨耗应该如何处理？",
        "普通货运线钢轨换轨的磨耗标准是什么？",
        "钢轨焊补适用于哪些情况？有什么限制条件？",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n  问题 {i}: {q}")
        print("  " + "-" * 58)
        answer = qa.answer_question(q)
        print(f"  回答: {answer}")


# ── CLI ───────────────────────────────────────────────────────────────────

def cmd_list_runs():
    runs = CheckpointManager.list_runs()
    if not runs:
        print("暂无历史 run（checkpoints/ 目录为空）")
        return
    print(f"\n{'run_id':<30} {'已完成步骤'}")
    print("─" * 60)
    for r in runs:
        print(f"  {r['run_id']:<28} {', '.join(r['steps']) or '（无）'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="钢轨伤损运维知识图谱构建系统")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        metavar="RUN_ID",
        help="从已有 checkpoint 继续（不传 run_id 则自动取最新）",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="列出所有历史 run 及 checkpoint 状态",
    )
    args = parser.parse_args()

    if args.list_runs:
        cmd_list_runs()
        return

    # ── 确定 run_id ──────────────────────────────────────────────────────
    if args.resume:
        if args.resume == "latest":
            run_id = CheckpointManager.latest_run_id()
            if not run_id:
                print("未找到任何历史 checkpoint，将新建 run。")
                run_id = make_run_id()
        else:
            run_id = args.resume
        print(f"  [resume] 继续 run: {run_id}")
    else:
        run_id = make_run_id()
        print(f"  [new run] run_id: {run_id}")

    ckpt   = CheckpointManager(run_id)
    tracer = LLMCallTracer(run_id=run_id, traces_dir=TRACES_DIR)

    # ── 流水线 ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("  钢轨伤损运维知识图谱构建系统")
    print("=" * 70)

    docs   = step1_load_documents(ckpt)
    result = step2_extract_knowledge(docs, ckpt, tracer)
    result = step3_align_entities(result, ckpt)
    client, builder = step4_build_graph(result)
    step5_demo_qa(client, tracer)

    if client:
        client.close()

    # ── 汇总 ─────────────────────────────────────────────────────────────
    tracer.print_summary()

    print("=" * 70)
    print("  流程完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
