from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from graph.neo4j_client import Neo4jClient


QA_SYSTEM_PROMPT = """你是钢轨运维知识图谱的智能问答助手。
我会给你提供从知识图谱中查询到的结构化数据，请根据这些数据回答用户的问题。
回答要专业、准确、简洁，使用中文。"""


class KnowledgeQA:
    def __init__(self, client: Neo4jClient):
        self.client = client
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=1000
        )

    # ==================== 查询方法 ====================

    def query_maintenance_methods_for_damage(self, damage_name: str) -> List[dict]:
        """查询某伤损类型对应的维修方式"""
        return self.client.run_read(
            """
            MATCH (d:DamageType)-[:HAS_MAINTENANCE_METHOD]->(m:MaintenanceMethod)
            WHERE d.name CONTAINS $name OR $name CONTAINS d.name
            RETURN d.name as damage_type, m.name as method,
                   m.method_type as method_type, m.procedure as procedure,
                   m.applicable_speed as applicable_speed
            """,
            {"name": damage_name}
        )

    def query_wear_limits(self, damage_name: Optional[str] = None,
                          line_type: Optional[str] = None) -> List[dict]:
        """查询磨耗限值"""
        if damage_name and line_type:
            return self.client.run_read(
                """
                MATCH (d:DamageType)-[:HAS_WEAR_LIMIT]->(w:WearLimit)
                WHERE (d.name CONTAINS $damage OR $damage CONTAINS d.name)
                  AND (w.line_type CONTAINS $line OR $line CONTAINS w.line_type)
                RETURN d.name as damage_type, w.line_type as line_type,
                       w.limit_value as limit_value, w.unit as unit,
                       w.rail_type as rail_type, w.condition as condition
                ORDER BY w.limit_value
                """,
                {"damage": damage_name, "line": line_type}
            )
        elif damage_name:
            return self.client.run_read(
                """
                MATCH (d:DamageType)-[:HAS_WEAR_LIMIT]->(w:WearLimit)
                WHERE d.name CONTAINS $damage OR $damage CONTAINS d.name
                RETURN d.name as damage_type, w.line_type as line_type,
                       w.limit_value as limit_value, w.unit as unit,
                       w.rail_type as rail_type, w.condition as condition
                ORDER BY w.limit_value
                """,
                {"damage": damage_name}
            )
        else:
            return self.client.run_read(
                """
                MATCH (d:DamageType)-[:HAS_WEAR_LIMIT]->(w:WearLimit)
                RETURN d.name as damage_type, w.line_type as line_type,
                       w.limit_value as limit_value, w.unit as unit,
                       w.rail_type as rail_type, w.condition as condition
                ORDER BY w.limit_value
                """
            )

    def query_maintenance_cycles(self, method_name: Optional[str] = None) -> List[dict]:
        """查询维修周期"""
        if method_name:
            return self.client.run_read(
                """
                MATCH (m:MaintenanceMethod)-[:HAS_MAINTENANCE_CYCLE]->(c:MaintenanceCycle)
                WHERE m.name CONTAINS $name OR $name CONTAINS m.name
                RETURN m.name as method, c.line_type as line_type,
                       c.cycle_description as cycle, c.trigger_condition as trigger
                """,
                {"name": method_name}
            )
        else:
            return self.client.run_read(
                """
                MATCH (m:MaintenanceMethod)-[:HAS_MAINTENANCE_CYCLE]->(c:MaintenanceCycle)
                RETURN m.name as method, c.line_type as line_type,
                       c.cycle_description as cycle, c.trigger_condition as trigger
                """
            )

    def query_all_damage_types(self) -> List[dict]:
        """查询所有伤损类型"""
        return self.client.run_read(
            "MATCH (d:DamageType) RETURN d.name as name, d.category as category, d.description as desc"
        )

    def query_grinding_process(self) -> List[dict]:
        """专项：查询打磨工艺相关信息"""
        return self.client.run_read(
            """
            MATCH (m:MaintenanceMethod)
            WHERE m.method_type = '打磨' OR m.name CONTAINS '打磨'
            OPTIONAL MATCH (m)-[:HAS_MAINTENANCE_CYCLE]->(c:MaintenanceCycle)
            OPTIONAL MATCH (m)-[:APPLICABLE_TO]->(a:ApplicableCondition)
            RETURN m.name as method, m.procedure as procedure,
                   m.equipment as equipment, m.applicable_speed as applicable_speed,
                   collect(DISTINCT c.cycle_description) as cycles,
                   collect(DISTINCT a.speed_limit) as speed_limits
            """
        )

    def query_damage_full_info(self, damage_name: str) -> dict:
        """查询伤损类型的完整信息（包括维修方式和磨耗限值）"""
        damage_info = self.client.run_read(
            "MATCH (d:DamageType) WHERE d.name CONTAINS $name RETURN d",
            {"name": damage_name}
        )
        methods = self.query_maintenance_methods_for_damage(damage_name)
        limits = self.query_wear_limits(damage_name=damage_name)
        return {
            "damage_info": damage_info,
            "maintenance_methods": methods,
            "wear_limits": limits
        }

    # ==================== 问答接口 ====================

    def answer_question(self, question: str) -> str:
        """主问答入口：根据问题自动查询并生成回答"""
        context_data = self._gather_context(question)

        if not context_data:
            return "抱歉，知识图谱中暂无相关信息。请确认知识图谱已成功构建，或换一种提问方式。"

        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{question}\n\n从知识图谱检索到的相关数据：\n{context_data}\n\n请基于以上数据回答问题。")
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"LLM回答生成失败: {e}\n\n原始查询数据：\n{context_data}"

    def _gather_context(self, question: str) -> str:
        """根据问题关键词自动决定查询策略并汇总上下文"""
        context_parts = []

        # 打磨相关
        if any(k in question for k in ["打磨", "磨削", "打磨工艺", "打磨方法"]):
            data = self.query_grinding_process()
            if data:
                context_parts.append(f"打磨工艺信息：\n{self._format_records(data)}")

        # 磨耗限值
        if any(k in question for k in ["限值", "磨耗", "超标", "多少", "标准"]):
            damage_keywords = ["垂直磨耗", "侧面磨耗", "波浪", "磨耗"]
            for kw in damage_keywords:
                if kw in question:
                    data = self.query_wear_limits(damage_name=kw)
                    if data:
                        context_parts.append(f"{kw}磨耗限值：\n{self._format_records(data)}")
                    break
            else:
                data = self.query_wear_limits()
                if data:
                    context_parts.append(f"所有磨耗限值：\n{self._format_records(data[:10])}")

        # 维修建议
        damage_types_to_check = [
            "轨头垂直磨耗", "垂直磨耗", "波浪形磨耗", "波磨",
            "横向裂纹", "核伤", "鱼腹", "侧面磨耗"
        ]
        for dt in damage_types_to_check:
            if dt in question:
                data = self.query_damage_full_info(dt)
                if data["maintenance_methods"] or data["wear_limits"]:
                    context_parts.append(f"{dt}相关信息：\n"
                                         f"维修方式：{self._format_records(data['maintenance_methods'])}\n"
                                         f"磨耗限值：{self._format_records(data['wear_limits'])}")
                break

        # 维修周期
        if any(k in question for k in ["周期", "多久", "频率", "何时", "什么时候"]):
            data = self.query_maintenance_cycles()
            if data:
                context_parts.append(f"维修周期信息：\n{self._format_records(data)}")

        # 通用：如果以上都未匹配，返回所有伤损类型
        if not context_parts:
            data = self.query_all_damage_types()
            if data:
                context_parts.append(f"已知伤损类型：\n{self._format_records(data)}")

        return "\n\n".join(context_parts)

    def _format_records(self, records: List[dict]) -> str:
        """格式化查询结果为文本"""
        if not records:
            return "（无数据）"
        lines = []
        for r in records:
            line = ", ".join(f"{k}: {v}" for k, v in r.items() if v and v != "" and v != [])
            lines.append(f"  - {line}")
        return "\n".join(lines)
