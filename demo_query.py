"""
演示查询脚本：直接从Neo4j图谱查询，展示知识图谱的查询能力。
需要先运行 main.py 构建图谱后使用。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.neo4j_client import Neo4jClient
from qa.knowledge_qa import KnowledgeQA


def demo_direct_cypher(client: Neo4jClient):
    print("\n--- 直接Cypher查询演示 ---\n")

    print("1. 查询所有伤损类型：")
    results = client.run_read("MATCH (d:DamageType) RETURN d.name, d.category ORDER BY d.category, d.name")
    for r in results:
        print(f"   {r['d.category'] or '未分类'} | {r['d.name']}")

    print("\n2. 查询所有维修方式：")
    results = client.run_read("MATCH (m:MaintenanceMethod) RETURN m.name, m.method_type ORDER BY m.method_type")
    for r in results:
        print(f"   {r['m.method_type'] or '其他'} | {r['m.name']}")

    print("\n3. 查询高速铁路垂直磨耗限值：")
    results = client.run_read(
        """
        MATCH (d:DamageType)-[:HAS_WEAR_LIMIT]->(w:WearLimit)
        WHERE w.line_type CONTAINS '高速' AND d.name CONTAINS '垂直磨耗'
        RETURN d.name, w.line_type, w.limit_value, w.unit, w.rail_type
        """
    )
    for r in results:
        print(f"   {r['d.name']} | {r['w.line_type']} | "
              f"限值: {r['w.limit_value']}{r['w.unit']} | 钢轨: {r['w.rail_type']}")

    print("\n4. 查询伤损类型→维修方式关系路径：")
    results = client.run_read(
        """
        MATCH (d:DamageType)-[r:HAS_MAINTENANCE_METHOD]->(m:MaintenanceMethod)
        RETURN d.name as damage, m.name as method, r.condition as condition
        ORDER BY d.name
        LIMIT 10
        """
    )
    for r in results:
        cond = f" [条件: {r['condition']}]" if r.get('condition') else ""
        print(f"   {r['damage']} → {r['method']}{cond}")

    print("\n5. 查询图谱全部节点和关系统计：")
    node_result = client.run_read(
        "MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC"
    )
    for r in node_result:
        print(f"   节点 {r['label']}: {r['cnt']} 个")

    rel_result = client.run_read(
        "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as cnt ORDER BY cnt DESC"
    )
    for r in rel_result:
        print(f"   关系 {r['rel_type']}: {r['cnt']} 条")


def demo_qa(client: Neo4jClient):
    print("\n--- 智能问答演示 ---\n")
    qa = KnowledgeQA(client)

    qa_pairs = [
        ("磨耗限值查询",
         "60kg/m钢轨在高速铁路上的垂直磨耗限值是多少？"),
        ("打磨工艺查询",
         "钢轨在线打磨的具体工艺流程是什么？适用于什么线路？"),
        ("维修建议",
         "发现钢轨有横向裂纹应该怎么处理？"),
        ("周期查询",
         "高速铁路的钢轨打磨周期是多久？"),
        ("综合建议",
         "波浪形磨耗超过0.2mm应该如何处理？有什么紧急措施？"),
    ]

    for title, question in qa_pairs:
        print(f"【{title}】")
        print(f"Q: {question}")
        answer = qa.answer_question(question)
        print(f"A: {answer}")
        print()


def main():
    print("=" * 70)
    print("  钢轨运维知识图谱 — 演示查询")
    print("=" * 70)

    client = Neo4jClient()
    if not client.verify_connection():
        print("\nNeo4j连接失败，请先确认：")
        print("1. Neo4j服务已启动（bolt://localhost:7687）")
        print("2. 已运行 main.py 构建了知识图谱")
        return

    try:
        demo_direct_cypher(client)
        demo_qa(client)
    finally:
        client.close()

    print("\n演示完成！")


if __name__ == "__main__":
    main()
