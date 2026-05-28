# 基于 LangExtract 与本地 LLM 的钢轨伤损运维知识图谱构建项目

**项目时间**：2025.09 - 2025.11  
**技术栈**：Python 3.12 · LangChain · OpenAI API · Neo4j · RapidFuzz · Pydantic

## 项目概述

面向钢轨伤损养护场景，整理养护规范、维修手册与历史处理记录等非结构化文本，构建钢轨运维知识抽取与图谱化管理流程。

**技术链路**：
```
非结构化运维文本 → LangExtract信息抽取 → LLM结构化抽取 → 实体对齐/去重 → Neo4j知识图谱 → 查询/问答辅助
```

## 项目结构

```
langextract_raij_maintance/
├── data/sample_docs/           # 模拟运维文本数据
│   ├── rail_maintenance_manual.txt  # 维修手册
│   ├── damage_standards.txt         # 伤损判定标准与磨耗限值
│   └── maintenance_records.txt      # 历史维修工单
├── schema/
│   └── knowledge_schema.py     # Pydantic实体与关系模型定义
├── extraction/
│   ├── text_loader.py          # 文本加载与分块
│   ├── llm_extractor.py        # Schema驱动的LLM结构化抽取
│   └── entity_aligner.py       # 实体对齐与去重（规则+相似度）
├── graph/
│   ├── neo4j_client.py         # Neo4j连接与CRUD操作
│   └── graph_builder.py        # 知识图谱构建逻辑
├── qa/
│   └── knowledge_qa.py         # 问答辅助接口（Cypher+LLM）
├── utils/
│   ├── tracer.py               # LLM调用追踪（LangChain Callback）
│   └── checkpoint.py           # 流水线断点续跑
├── traces/                     # LLM调用日志（JSONL，自动生成）
├── checkpoints/                # 各步骤中间结果（自动生成）
├── main.py                     # 端到端主流程（支持 --resume）
├── demo_query.py               # 演示查询脚本
├── config.py                   # 配置管理
└── requirements.txt
```

## 知识图谱Schema

### 实体类型

| 实体类型 | 说明 | 示例 |
|----------|------|------|
| `DamageType` | 钢轨伤损类型 | 轨头垂直磨耗、波浪形磨耗、横向裂纹 |
| `MaintenanceMethod` | 维修方式 | 在线打磨、换轨、焊补 |
| `WearLimit` | 磨耗限值 | 高速铁路60kg/m钢轨限值5mm |
| `MaintenanceCycle` | 维修周期 | 高速铁路每年至少2次打磨 |
| `ApplicableCondition` | 适用条件 | 焊补仅适用160km/h以下 |

### 关系类型

| 关系 | 含义 |
|------|------|
| `HAS_MAINTENANCE_METHOD` | 伤损类型 → 维修方式 |
| `HAS_WEAR_LIMIT` | 伤损类型 → 磨耗限值 |
| `HAS_MAINTENANCE_CYCLE` | 维修方式 → 维修周期 |
| `APPLICABLE_TO` | 维修方式 → 适用条件 |
| `TRIGGERS` | 磨耗限值 → 触发的维修方式 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
OPENAI_MODEL=gpt-3.5-turbo

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### 3. 启动 Neo4j

确保本地 Neo4j 服务已启动（默认 bolt://localhost:7687）。

### 4. 运行主流程

```bash
# 全新运行（自动生成 run_id）
python main.py

# 从上次断点继续（跳过已完成步骤，节省LLM费用）
python main.py --resume

# 指定 run_id 继续
python main.py --resume 20240528_143022_abc123

# 查看所有历史 run 及 checkpoint 状态
python main.py --list-runs
```

执行步骤：
1. 加载运维文档（维修手册、伤损标准、历史记录）
2. LLM结构化抽取（Schema驱动，JSON输出）
3. 实体对齐与去重（同义词词典 + 字符串相似度）
4. 构建Neo4j知识图谱
5. 问答辅助演示
6. 打印 LLM 调用追踪汇总（调用次数/耗时/tokens/费用）

### 5. 演示查询

```bash
python demo_query.py
```

包含：Cypher直接查询 + 智能问答（5个典型问题）

## 核心实现

### Schema驱动的信息抽取

定义 `ExtractionResult` Pydantic模型，包含7类实体和关系，约束LLM输出为标准JSON结构：

```python
class ExtractionResult(BaseModel):
    damage_types: List[DamageType]
    maintenance_methods: List[MaintenanceMethod]
    wear_limits: List[WearLimit]
    ...
```

### 实体对齐与去重

结合两种方法：
- **规则匹配**：同义词词典（"波磨"→"波浪形磨耗"，"仿形打磨"→"在线打磨"）
- **字符串相似度**：RapidFuzz模糊匹配，阈值0.85

### 多实体关系建模

Neo4j图谱支持复杂查询，例如：

```cypher
-- 查询高速铁路垂直磨耗限值及触发的维修方式
MATCH (d:DamageType)-[:HAS_WEAR_LIMIT]->(w:WearLimit)-[:TRIGGERS]->(m:MaintenanceMethod)
WHERE d.name CONTAINS '垂直磨耗' AND w.line_type CONTAINS '高速'
RETURN d.name, w.limit_value, w.unit, m.name
```

### LLM 调用追踪（Trace）

基于 LangChain `BaseCallbackHandler` 实现，无侵入式挂载：

```python
tracer = LLMCallTracer(run_id=run_id, traces_dir=TRACES_DIR)
extractor = LLMExtractor(callbacks=[tracer])   # 传入即生效
```

每次 LLM 调用自动记录并追加写入 `traces/<run_id>.jsonl`：

```json
{"step": "step2_extract", "model": "gpt-3.5-turbo", "latency_ms": 2341,
 "prompt_tokens": 1204, "completion_tokens": 487, "cost_usd": 0.001333, "status": "success"}
```

流程结束输出汇总表（调用次数 / 总耗时 / tokens / 预估费用）。

### Checkpoint 断点续跑

各步骤完成后序列化中间结果至 `checkpoints/<run_id>/`，`ExtractionResult` 通过 Pydantic `model_dump()` / `model_validate()` 往返序列化：

```python
ckpt.save("step2_extracted", merged_result)    # 保存
result = ckpt.load("step2_extracted", as_extraction_result=True)  # 恢复
```

网络抖动或 LLM 限流导致中断后，`--resume` 可跳过已完成步骤直接继续。

## 典型问答示例

- **打磨工艺查询**：钢轨在线打磨的具体流程是什么？
- **磨耗限值查询**：高速铁路60kg/m钢轨的头部磨耗限值是多少？
- **维修建议**：发现波浪形磨耗超过0.2mm应该如何处理？
- **周期查询**：高速铁路的打磨维修周期是多长？

## 成果

- 支持钢轨打磨工艺、磨耗限值查询及维修建议等典型问题的智能问答辅助
- 知识图谱涵盖5类实体、5种关系，支持多跳复杂查询
- 实体对齐去重率达90%+，显著提升查询一致性
