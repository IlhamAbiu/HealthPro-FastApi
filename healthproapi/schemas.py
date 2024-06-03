# schemas.py
from pydantic import BaseModel
from typing import List


class HealthMetricsRequest(BaseModel):
    weight: float
    body_fat_percentage: float
    height: float
    basal_metabolism: float
    activity_level: float
    avg_steps_per_week: float
    avg_calories_burned_per_week: float
    age: int
    gender: str  # 'male' or 'female'


class PulseData(BaseModel):
    time: str
    pulse: int


class LifeMetricsRequest(BaseModel):
    systolic_pressure: float
    diastolic_pressure: float
    resting_heart_rate: float
    max_heart_rate: float
    oxygen_levels: List[float]  # [resting_oxygen_level, active_oxygen_level]
