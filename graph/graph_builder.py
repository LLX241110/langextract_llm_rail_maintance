from graph.neo4j_client import Neo4jClient
from schema.knowledge_schema import ExtractionResult


class GraphBuilder:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def build_graph(self, result: ExtractionResult):
        """将抽取结果写入Neo4j图数据库"""
        print("  开始构建知识图谱...")
        self._create_damage_types(result)
        self._create_maintenance_methods(result)
        self._create_wear_limits(result)
        self._create_maintenance_cycles(result)
        self._create_applicable_conditions(result)
        self._create_damage_maintenance_relations(result)
        self._create_wear_limit_triggers(result)
        print("  知识图谱构建完成")

    def _create_damage_types(self, result: ExtractionResult):
        for dt in result.damage_types:
            self.client.run(
                """
                MERGE (d:DamageType {name: $name})
                SET d.category = $category,
                    d.severity_levels = $severity_levels,
                    d.description = $description
                """,
                {
                    "name": dt.name,
                    "category": dt.category or "",
                    "severity_levels": dt.severity_levels or [],
                    "description": dt.description or ""
                }
            )
        print(f"    写入 {len(result.damage_types)} 个伤损类型节点")

    def _create_maintenance_methods(self, result: ExtractionResult):
        for m in result.maintenance_methods:
            self.client.run(
                """
                MERGE (m:MaintenanceMethod {name: $name})
                SET m.method_type = $method_type,
                    m.procedure = $procedure,
                    m.equipment = $equipment,
                    m.applicable_speed = $applicable_speed
                """,
                {
                    "name": m.name,
                    "method_type": m.method_type or "",
                    "procedure": m.procedure or "",
                    "equipment": m.equipment or "",
                    "applicable_speed": m.applicable_speed or ""
                }
            )
        print(f"    写入 {len(result.maintenance_methods)} 个维修方式节点")

    def _create_wear_limits(self, result: ExtractionResult):
        for wl in result.wear_limits:
            node_id = f"{wl.damage_type}_{wl.line_type}_{wl.rail_type or ''}_{wl.limit_value}"
            self.client.run(
                """
                MERGE (w:WearLimit {node_id: $node_id})
                SET w.damage_type = $damage_type,
                    w.line_type = $line_type,
                    w.limit_value = $limit_value,
                    w.unit = $unit,
                    w.rail_type = $rail_type,
                    w.condition = $condition
                WITH w
                MATCH (d:DamageType {name: $damage_type})
                MERGE (d)-[:HAS_WEAR_LIMIT]->(w)
                """,
                {
                    "node_id": node_id,
                    "damage_type": wl.damage_type,
                    "line_type": wl.line_type,
                    "limit_value": wl.limit_value,
                    "unit": wl.unit,
                    "rail_type": wl.rail_type or "",
                    "condition": wl.condition or ""
                }
            )
        print(f"    写入 {len(result.wear_limits)} 个磨耗限值节点")

    def _create_maintenance_cycles(self, result: ExtractionResult):
        for cycle in result.maintenance_cycles:
            self.client.run(
                """
                MATCH (m:MaintenanceMethod {name: $method_name})
                MERGE (c:MaintenanceCycle {
                    maintenance_method: $method_name,
                    line_type: $line_type,
                    cycle_description: $cycle_description
                })
                SET c.trigger_condition = $trigger_condition
                MERGE (m)-[:HAS_MAINTENANCE_CYCLE]->(c)
                """,
                {
                    "method_name": cycle.maintenance_method,
                    "line_type": cycle.line_type or "",
                    "cycle_description": cycle.cycle_description,
                    "trigger_condition": cycle.trigger_condition or ""
                }
            )
        print(f"    写入 {len(result.maintenance_cycles)} 个维修周期节点")

    def _create_applicable_conditions(self, result: ExtractionResult):
        for cond in result.applicable_conditions:
            self.client.run(
                """
                MATCH (m:MaintenanceMethod {name: $method_name})
                MERGE (c:ApplicableCondition {
                    maintenance_method: $method_name,
                    line_grade: $line_grade,
                    speed_limit: $speed_limit
                })
                SET c.restriction = $restriction
                MERGE (m)-[:APPLICABLE_TO]->(c)
                """,
                {
                    "method_name": cond.maintenance_method,
                    "line_grade": cond.line_grade or "",
                    "speed_limit": cond.speed_limit or "",
                    "restriction": cond.restriction or ""
                }
            )
        print(f"    写入 {len(result.applicable_conditions)} 个适用条件节点")

    def _create_damage_maintenance_relations(self, result: ExtractionResult):
        count = 0
        for rel in result.damage_maintenance_relations:
            result_set = self.client.run_read(
                "MATCH (d:DamageType {name: $dt}), (m:MaintenanceMethod {name: $mm}) RETURN d, m",
                {"dt": rel.damage_type, "mm": rel.maintenance_method}
            )
            if result_set:
                self.client.run(
                    """
                    MATCH (d:DamageType {name: $dt}), (m:MaintenanceMethod {name: $mm})
                    MERGE (d)-[r:HAS_MAINTENANCE_METHOD]->(m)
                    SET r.condition = $condition
                    """,
                    {
                        "dt": rel.damage_type,
                        "mm": rel.maintenance_method,
                        "condition": rel.condition or ""
                    }
                )
                count += 1
        print(f"    写入 {count} 条伤损-维修关系")

    def _create_wear_limit_triggers(self, result: ExtractionResult):
        count = 0
        for rel in result.wear_limit_trigger_relations:
            self.client.run(
                """
                MATCH (w:WearLimit {damage_type: $damage_type, line_type: $line_type})
                MATCH (m:MaintenanceMethod {name: $method_name})
                MERGE (w)-[:TRIGGERS]->(m)
                """,
                {
                    "damage_type": rel.wear_limit_damage_type,
                    "line_type": rel.wear_limit_line_type,
                    "method_name": rel.triggered_maintenance
                }
            )
            count += 1
        print(f"    写入 {count} 条限值触发关系")

    def get_graph_stats(self) -> dict:
        """获取图谱统计信息"""
        stats = {}
        node_types = ["DamageType", "MaintenanceMethod", "WearLimit",
                      "MaintenanceCycle", "ApplicableCondition"]
        for nt in node_types:
            result = self.client.run_read(f"MATCH (n:{nt}) RETURN count(n) as cnt")
            stats[nt] = result[0]["cnt"] if result else 0

        rel_result = self.client.run_read("MATCH ()-[r]->() RETURN count(r) as cnt")
        stats["total_relations"] = rel_result[0]["cnt"] if rel_result else 0
        return stats
