# models.py
from sqlalchemy import Column, Integer, String, Float
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    weight = Column(Float)
    height = Column(Float)


class HealthData(Base):
    __tablename__ = "health_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    bmi = Column(Float)
    visceral_fat_index = Column(Float)
    resting_heart_rate = Column(Float)
    map = Column(Float)
    pp = Column(Float)
    tdee = Column(Float)
    steps = Column(Integer)
    sleep_quality = Column(Float)
    sleep_efficiency = Column(Float)
    sleep_latency = Column(Float)
    deep_sleep_percentage = Column(Float)
    rem_sleep_percentage = Column(Float)
    light_sleep_percentage = Column(Float)
