from typing import Optional, List
from pydantic import BaseModel, Field


class DamageType(BaseModel):
    """钢轨伤损类型实体"""
    name: str = Field(description="伤损类型名称，如：轨头垂直磨耗、波浪形磨耗、横向裂纹")
    category: Optional[str] = Field(None, description="伤损分类：轨头伤损/轨腰伤损/轨底伤损")
    severity_levels: Optional[List[str]] = Field(None, description="伤损等级列表，如：['一级','二级','三级']")
    description: Optional[str] = Field(None, description="伤损描述")


class MaintenanceMethod(BaseModel):
    """维修方式实体"""
    name: str = Field(description="维修方式名称，如：在线打磨、换轨、焊补")
    method_type: Optional[str] = Field(None, description="方式类型：打磨/换轨/焊补/其他")
    procedure: Optional[str] = Field(None, description="作业流程简述")
    equipment: Optional[str] = Field(None, description="所需设备")
    applicable_speed: Optional[str] = Field(None, description="适用速度等级，如：160km/h以下")


class WearLimit(BaseModel):
    """磨耗限值实体"""
    damage_type: str = Field(description="对应的伤损类型名称")
    line_type: str = Field(description="线路类型，如：高速铁路、客货共线一级、普通货运线")
    limit_value: float = Field(description="限值数值")
    unit: str = Field(description="单位，如：mm")
    rail_type: Optional[str] = Field(None, description="钢轨型号，如：60kg/m")
    condition: Optional[str] = Field(None, description="附加条件说明")


class MaintenanceCycle(BaseModel):
    """维修周期实体"""
    maintenance_method: str = Field(description="对应的维修方式名称")
    line_type: Optional[str] = Field(None, description="适用线路类型")
    cycle_description: str = Field(description="周期描述，如：每年至少1次、累计通过2亿吨")
    trigger_condition: Optional[str] = Field(None, description="触发条件")


class ApplicableCondition(BaseModel):
    """适用条件实体"""
    maintenance_method: str = Field(description="对应的维修方式名称")
    line_grade: Optional[str] = Field(None, description="线路等级")
    speed_limit: Optional[str] = Field(None, description="速度限制")
    restriction: Optional[str] = Field(None, description="其他限制条件")


class DamageMaintenanceRelation(BaseModel):
    """伤损类型与维修方式关系"""
    damage_type: str = Field(description="伤损类型名称")
    maintenance_method: str = Field(description="维修方式名称")
    condition: Optional[str] = Field(None, description="触发该维修方式的条件")


class WearLimitTriggerRelation(BaseModel):
    """磨耗限值触发维修关系"""
    wear_limit_damage_type: str = Field(description="磨耗限值对应的伤损类型")
    wear_limit_line_type: str = Field(description="磨耗限值对应的线路类型")
    triggered_maintenance: str = Field(description="触发的维修方式")


class ExtractionResult(BaseModel):
    """LLM抽取结果的完整结构"""
    damage_types: List[DamageType] = Field(default_factory=list, description="抽取到的伤损类型")
    maintenance_methods: List[MaintenanceMethod] = Field(default_factory=list, description="抽取到的维修方式")
    wear_limits: List[WearLimit] = Field(default_factory=list, description="抽取到的磨耗限值")
    maintenance_cycles: List[MaintenanceCycle] = Field(default_factory=list, description="抽取到的维修周期")
    applicable_conditions: List[ApplicableCondition] = Field(default_factory=list, description="抽取到的适用条件")
    damage_maintenance_relations: List[DamageMaintenanceRelation] = Field(default_factory=list, description="伤损-维修关系")
    wear_limit_trigger_relations: List[WearLimitTriggerRelation] = Field(default_factory=list, description="磨耗限值触发关系")
