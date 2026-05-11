from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def run(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            return session.run(query, parameters or {})

    def run_read(self, query: str, parameters: dict = None) -> list:
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def clear_database(self):
        self.run("MATCH (n) DETACH DELETE n")
        print("  数据库已清空")

    def create_constraints(self):
        """创建唯一性约束"""
        constraints = [
            "CREATE CONSTRAINT damage_type_name IF NOT EXISTS FOR (d:DamageType) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT maintenance_method_name IF NOT EXISTS FOR (m:MaintenanceMethod) REQUIRE m.name IS UNIQUE",
        ]
        for c in constraints:
            try:
                self.run(c)
            except Exception as e:
                # 约束可能已存在
                pass
        print("  约束创建完成")

    def verify_connection(self) -> bool:
        try:
            self.run("RETURN 1")
            return True
        except Exception as e:
            print(f"  Neo4j连接失败: {e}")
            return False
