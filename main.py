"""
钢轨伤损运维知识图谱构建系统
主流程：非结构化文本 → LLM抽取 → 实体对齐 → Neo4j图谱 → 问答辅助
"""
import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraction.text_loader import load_all_documents, split_text_into_chunks
from extraction.llm_extractor import LLMExtractor
from extraction.entity_aligner import align_and_deduplicate
from graph.neo4j_client import Neo4jClient
from graph.graph_builder import GraphBuilder
from qa.knowledge_qa import KnowledgeQA
from schema.knowledge_schema import ExtractionResult
from config import CHUNK_SIZE, CHUNK_OVERLAP

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_docs")


def step1_load_documents():
    print("\n=== Step 1: 加载运维文档 ===")
    docs = load_all_documents(DATA_DIR)
    print(f"共加载 {len(docs)} 个文档")
    return docs


def step2_extract_knowledge(docs: list) -> ExtractionResult:
    print("\n=== Step 2: LLM结构化抽取 ===")
    extractor = LLMExtractor()
    merged_result = ExtractionResult()

    for doc in docs:
        print(f"\n处理文档: {doc['filename']}")
        chunks = split_text_into_chunks(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        doc_result = extractor.extract_from_document(
            doc["content"], doc["filename"], chunks
        )
        # 合并所有文档的结果
        merged_result.damage_types.extend(doc_result.damage_types)
        merged_result.maintenance_methods.extend(doc_result.maintenance_methods)
        merged_result.wear_limits.extend(doc_result.wear_limits)
        merged_result.maintenance_cycles.extend(doc_result.maintenance_cycles)
        merged_result.applicable_conditions.extend(doc_result.applicable_conditions)
        merged_result.damage_maintenance_relations.extend(doc_result.damage_maintenance_relations)
        merged_result.wear_limit_trigger_relations.extend(doc_result.wear_limit_trigger_relations)

    print(f"\n抽取完成（去重前）：")
    print(f"  伤损类型: {len(merged_result.damage_types)}")
    print(f"  维修方式: {len(merged_result.maintenance_methods)}")
    print(f"  磨耗限值: {len(merged_result.wear_limits)}")
    print(f"  维修周期: {len(merged_result.maintenance_cycles)}")
    print(f"  适用条件: {len(merged_result.applicable_conditions)}")
    print(f"  伤损-维修关系: {len(merged_result.damage_maintenance_relations)}")
    return merged_result


def step3_align_entities(result: ExtractionResult) -> ExtractionResult:
    print("\n=== Step 3: 实体对齐与去重 ===")
    return align_and_deduplicate(result)


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


def step5_demo_qa(client: Neo4jClient):
    print("\n=== Step 5: 问答辅助演示 ===")
    if client is None:
        print("  Neo4j未连接，跳过问答演示")
        return

    qa = KnowledgeQA(client)
    questions = [
        "钢轨打磨的工艺流程是什么？需要什么设备？",
        "高速铁路的轨头垂直磨耗限值是多少？",
        "发现钢轨波浪形磨耗应该如何处理？",
        "普通货运线钢轨换轨的磨耗标准是什么？",
        "钢轨焊补适用于哪些情况？有什么限制条件？"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n问题 {i}: {q}")
        print("-" * 60)
        answer = qa.answer_question(q)
        print(f"回答: {answer}")


def main():
    print("=" * 70)
    print("  钢轨伤损运维知识图谱构建系统")
    print("=" * 70)

    docs = step1_load_documents()
    result = step2_extract_knowledge(docs)
    result = step3_align_entities(result)
    client, builder = step4_build_graph(result)
    step5_demo_qa(client)

    if client:
        client.close()

    print("\n" + "=" * 70)
    print("  流程完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
