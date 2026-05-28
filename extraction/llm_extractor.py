import json
import re
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, SystemMessage

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from schema.knowledge_schema import ExtractionResult


SYSTEM_PROMPT = """你是一个专业的钢轨运维知识抽取专家。请从提供的钢轨运维文本中，
抽取以下类型的结构化知识：

1. **伤损类型（DamageType）**：钢轨的各种伤损，如垂直磨耗、波浪形磨耗、横向裂纹等
2. **维修方式（MaintenanceMethod）**：处理伤损的方法，如在线打磨、换轨、焊补等
3. **磨耗限值（WearLimit）**：不同线路等级下的磨耗允许值
4. **维修周期（MaintenanceCycle）**：各种维修方式的执行周期或触发条件
5. **适用条件（ApplicableCondition）**：维修方式的适用场景和限制条件
6. **伤损-维修关系（DamageMaintenanceRelation）**：哪种伤损对应哪种维修方式
7. **限值触发关系（WearLimitTriggerRelation）**：超过磨耗限值时触发的维修动作

请严格按照JSON格式输出，不要添加任何额外说明。"""

EXTRACTION_PROMPT = """请从以下钢轨运维文本中抽取结构化知识：

文本内容：
{text}

请按以下JSON格式输出（只输出JSON，不要有其他内容）：
{{
  "damage_types": [
    {{
      "name": "伤损类型名称",
      "category": "轨头伤损/轨腰伤损/轨底伤损",
      "severity_levels": ["一级", "二级"],
      "description": "描述"
    }}
  ],
  "maintenance_methods": [
    {{
      "name": "维修方式名称",
      "method_type": "打磨/换轨/焊补/其他",
      "procedure": "作业流程简述",
      "equipment": "所需设备",
      "applicable_speed": "适用速度等级"
    }}
  ],
  "wear_limits": [
    {{
      "damage_type": "伤损类型名称",
      "line_type": "线路类型",
      "limit_value": 数值,
      "unit": "mm",
      "rail_type": "钢轨型号",
      "condition": "附加条件"
    }}
  ],
  "maintenance_cycles": [
    {{
      "maintenance_method": "维修方式名称",
      "line_type": "线路类型",
      "cycle_description": "周期描述",
      "trigger_condition": "触发条件"
    }}
  ],
  "applicable_conditions": [
    {{
      "maintenance_method": "维修方式名称",
      "line_grade": "线路等级",
      "speed_limit": "速度限制",
      "restriction": "其他限制"
    }}
  ],
  "damage_maintenance_relations": [
    {{
      "damage_type": "伤损类型",
      "maintenance_method": "维修方式",
      "condition": "触发条件"
    }}
  ],
  "wear_limit_trigger_relations": [
    {{
      "wear_limit_damage_type": "伤损类型",
      "wear_limit_line_type": "线路类型",
      "triggered_maintenance": "触发的维修方式"
    }}
  ]
}}

只抽取文本中明确提到的信息，不要推测或补充文本未提及的内容。
如果某个类别没有相关信息，对应列表返回空数组[]。"""


class LLMExtractor:
    def __init__(self, callbacks: Optional[List[BaseCallbackHandler]] = None):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=4000,
            callbacks=callbacks or [],
        )

    def extract_from_chunk(self, text: str) -> Optional[ExtractionResult]:
        """从单个文本块抽取知识"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=EXTRACTION_PROMPT.format(text=text))
        ]

        try:
            response = self.llm.invoke(messages)
            raw_content = response.content.strip()

            # 提取JSON部分（去掉可能的markdown代码块标记）
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return ExtractionResult(**data)
            else:
                print(f"  警告：未找到JSON内容，原始响应：{raw_content[:200]}")
                return None
        except json.JSONDecodeError as e:
            print(f"  警告：JSON解析失败: {e}")
            return None
        except Exception as e:
            print(f"  错误：LLM调用失败: {e}")
            return None

    def extract_from_document(self, doc_content: str, doc_name: str, chunks: list) -> ExtractionResult:
        """从完整文档（多个chunk）中抽取知识，合并结果"""
        print(f"  处理文档: {doc_name}，共 {len(chunks)} 个块")

        merged = ExtractionResult()

        for i, chunk in enumerate(chunks):
            print(f"    抽取第 {i+1}/{len(chunks)} 块...")
            result = self.extract_from_chunk(chunk)
            if result:
                merged.damage_types.extend(result.damage_types)
                merged.maintenance_methods.extend(result.maintenance_methods)
                merged.wear_limits.extend(result.wear_limits)
                merged.maintenance_cycles.extend(result.maintenance_cycles)
                merged.applicable_conditions.extend(result.applicable_conditions)
                merged.damage_maintenance_relations.extend(result.damage_maintenance_relations)
                merged.wear_limit_trigger_relations.extend(result.wear_limit_trigger_relations)

        return merged
