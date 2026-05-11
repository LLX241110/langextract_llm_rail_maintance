from typing import List, Dict
from rapidfuzz import fuzz
from config import SIMILARITY_THRESHOLD
from schema.knowledge_schema import (
    ExtractionResult, DamageType, MaintenanceMethod,
    WearLimit, MaintenanceCycle, ApplicableCondition,
    DamageMaintenanceRelation, WearLimitTriggerRelation
)


# 同义词词典：将变体名称映射到标准名称
DAMAGE_SYNONYMS: Dict[str, str] = {
    "波磨": "波浪形磨耗",
    "波浪磨耗": "波浪形磨耗",
    "波形磨耗": "波浪形磨耗",
    "垂直磨耗": "轨头垂直磨耗",
    "竖向磨耗": "轨头垂直磨耗",
    "侧磨": "轨头侧面磨耗",
    "侧面磨耗": "轨头侧面磨耗",
    "核伤": "轨头核伤",
    "剥离掉块": "轨头核伤",
    "掉块": "轨头核伤",
    "横裂": "横向裂纹",
    "横向断裂": "横向裂纹",
    "鱼腹伤": "鱼腹轨伤",
    "鱼腹伤损": "鱼腹轨伤",
    "轨底锈蚀": "轨底腐蚀",
    "底部腐蚀": "轨底腐蚀",
    "焊缝不平顺": "焊缝平顺性缺陷",
    "焊缝不平": "焊缝平顺性缺陷",
}

MAINTENANCE_SYNONYMS: Dict[str, str] = {
    "仿形打磨": "在线打磨",
    "机械打磨": "在线打磨",
    "钢轨打磨": "在线打磨",
    "打磨修复": "在线打磨",
    "手工打磨": "手工精磨",
    "砂轮打磨": "手工精磨",
    "更换钢轨": "换轨",
    "更换": "换轨",
    "铺换新轨": "换轨",
    "焊接修复": "焊补",
    "CO2焊补": "焊补",
    "铝热焊接": "焊补",
}

LINE_SYNONYMS: Dict[str, str] = {
    "高铁": "高速铁路",
    "高速线路": "高速铁路",
    "普速线路": "普通线路",
    "普通铁路": "普通线路",
    "货运线": "普通货运线",
    "客货线": "客货共线",
}


def normalize_name(name: str, synonyms: Dict[str, str]) -> str:
    """用同义词词典标准化名称"""
    return synonyms.get(name, name)


def are_similar(name1: str, name2: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """判断两个名称是否相似（字符串相似度）"""
    ratio = fuzz.ratio(name1, name2) / 100.0
    partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
    return max(ratio, partial_ratio) >= threshold


def deduplicate_damage_types(damage_types: List[DamageType]) -> List[DamageType]:
    """对伤损类型去重"""
    unique = []
    for dt in damage_types:
        normalized = normalize_name(dt.name, DAMAGE_SYNONYMS)
        dt.name = normalized
        # 检查是否已存在相似实体
        found = False
        for existing in unique:
            if existing.name == normalized or are_similar(existing.name, normalized):
                # 合并信息：优先保留非空字段
                if not existing.category and dt.category:
                    existing.category = dt.category
                if not existing.severity_levels and dt.severity_levels:
                    existing.severity_levels = dt.severity_levels
                if not existing.description and dt.description:
                    existing.description = dt.description
                found = True
                break
        if not found:
            unique.append(dt)
    return unique


def deduplicate_maintenance_methods(methods: List[MaintenanceMethod]) -> List[MaintenanceMethod]:
    """对维修方式去重"""
    unique = []
    for m in methods:
        normalized = normalize_name(m.name, MAINTENANCE_SYNONYMS)
        m.name = normalized
        found = False
        for existing in unique:
            if existing.name == normalized or are_similar(existing.name, normalized):
                if not existing.method_type and m.method_type:
                    existing.method_type = m.method_type
                if not existing.procedure and m.procedure:
                    existing.procedure = m.procedure
                if not existing.equipment and m.equipment:
                    existing.equipment = m.equipment
                if not existing.applicable_speed and m.applicable_speed:
                    existing.applicable_speed = m.applicable_speed
                found = True
                break
        if not found:
            unique.append(m)
    return unique


def deduplicate_wear_limits(limits: List[WearLimit]) -> List[WearLimit]:
    """对磨耗限值去重（按伤损类型+线路类型+钢轨型号作为唯一键）"""
    seen = set()
    unique = []
    for wl in limits:
        dt_norm = normalize_name(wl.damage_type, DAMAGE_SYNONYMS)
        lt_norm = normalize_name(wl.line_type, LINE_SYNONYMS)
        wl.damage_type = dt_norm
        wl.line_type = lt_norm
        key = (dt_norm, lt_norm, wl.rail_type or "", wl.limit_value)
        if key not in seen:
            seen.add(key)
            unique.append(wl)
    return unique


def align_relations(relations: List[DamageMaintenanceRelation],
                    known_damages: List[str],
                    known_methods: List[str]) -> List[DamageMaintenanceRelation]:
    """对齐关系中的实体名称到已知实体"""
    aligned = []
    seen = set()
    for rel in relations:
        dt = normalize_name(rel.damage_type, DAMAGE_SYNONYMS)
        mm = normalize_name(rel.maintenance_method, MAINTENANCE_SYNONYMS)
        # 模糊匹配到已知实体
        for kd in known_damages:
            if dt == kd or are_similar(dt, kd, 0.8):
                dt = kd
                break
        for km in known_methods:
            if mm == km or are_similar(mm, km, 0.8):
                mm = km
                break
        key = (dt, mm)
        if key not in seen:
            seen.add(key)
            rel.damage_type = dt
            rel.maintenance_method = mm
            aligned.append(rel)
    return aligned


def align_and_deduplicate(result: ExtractionResult) -> ExtractionResult:
    """对整个抽取结果进行实体对齐和去重"""
    print("  执行实体对齐与去重...")

    result.damage_types = deduplicate_damage_types(result.damage_types)
    result.maintenance_methods = deduplicate_maintenance_methods(result.maintenance_methods)
    result.wear_limits = deduplicate_wear_limits(result.wear_limits)

    # 对关系中的实体名称进行对齐
    known_damages = [dt.name for dt in result.damage_types]
    known_methods = [m.name for m in result.maintenance_methods]

    result.damage_maintenance_relations = align_relations(
        result.damage_maintenance_relations, known_damages, known_methods
    )

    # 去重触发关系
    seen_triggers = set()
    unique_triggers = []
    for rel in result.wear_limit_trigger_relations:
        dt = normalize_name(rel.wear_limit_damage_type, DAMAGE_SYNONYMS)
        lt = normalize_name(rel.wear_limit_line_type, LINE_SYNONYMS)
        mm = normalize_name(rel.triggered_maintenance, MAINTENANCE_SYNONYMS)
        rel.wear_limit_damage_type = dt
        rel.wear_limit_line_type = lt
        rel.triggered_maintenance = mm
        key = (dt, lt, mm)
        if key not in seen_triggers:
            seen_triggers.add(key)
            unique_triggers.append(rel)
    result.wear_limit_trigger_relations = unique_triggers

    # 去重维修周期
    seen_cycles = set()
    unique_cycles = []
    for cycle in result.maintenance_cycles:
        mm = normalize_name(cycle.maintenance_method, MAINTENANCE_SYNONYMS)
        cycle.maintenance_method = mm
        key = (mm, cycle.line_type or "", cycle.cycle_description)
        if key not in seen_cycles:
            seen_cycles.add(key)
            unique_cycles.append(cycle)
    result.maintenance_cycles = unique_cycles

    # 去重适用条件
    seen_conds = set()
    unique_conds = []
    for cond in result.applicable_conditions:
        mm = normalize_name(cond.maintenance_method, MAINTENANCE_SYNONYMS)
        cond.maintenance_method = mm
        key = (mm, cond.line_grade or "", cond.speed_limit or "")
        if key not in seen_conds:
            seen_conds.add(key)
            unique_conds.append(cond)
    result.applicable_conditions = unique_conds

    print(f"  对齐后：{len(result.damage_types)} 伤损类型, "
          f"{len(result.maintenance_methods)} 维修方式, "
          f"{len(result.wear_limits)} 磨耗限值, "
          f"{len(result.damage_maintenance_relations)} 伤损-维修关系")
    return result
