from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Union
import pandas as pd
import datetime
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
import os

app = FastAPI()

load_dotenv()
# Настройка OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")


class UserPhysicalData(BaseModel):
    height: float = Field(..., example=180)
    weight: float = Field(..., example=100)
    body_fat_percentage: float = Field(..., example=20)
    bmi: float = Field(..., example=20)
    basal_metabolism: float = Field(..., example=2200)
    tdee: float = Field(..., example=2500)
    avg_calories_burned_walking: float = Field(..., example=200)
    avg_calories_burned_exercise: float = Field(..., example=200)
    maintenance_calories: float = Field(..., example=3200)
    weight_loss_calories: float = Field(..., example=2900)
    weight_gain_calories: float = Field(..., example=3500)
    visceral_fat_index: float = Field(..., example=20)
    age: float = Field(..., example=25)
    gender: str = Field(..., example="male")


class UserVitalData(BaseModel):
    resting_pulse: float = Field(..., example=60)
    average_pulse: float = Field(..., example=70)
    median_pulse: float = Field(..., example=68)
    map: float = Field(..., example=93)
    pp: float = Field(..., example=40)
    age: float = Field(..., example=25)
    gender: str = Field(..., example="male")


class UserActivityData(BaseModel):
    step_trends: str = Field(..., example="увеличение на 5 процентов")
    target_calories: int = Field(..., example=2500)
    age: float = Field(..., example=25)
    gender: str = Field(..., example="male")


class UserSleepData(BaseModel):
    sleep_quality_index: float = Field(..., example=47.02)
    sleep_efficiency: float = Field(..., example=94.54)
    sleep_latency: float = Field(..., example=77.00)
    deep_sleep_percentage: float = Field(..., example=15.75)
    rem_sleep_percentage: float = Field(..., example=28.35)
    light_sleep_percentage: float = Field(..., example=55.12)
    age: float = Field(..., example=25)
    gender: str = Field(..., example="male")


class OpenAIRequest(BaseModel):
    prompt_number: int
    user_data: Union[UserPhysicalData, UserVitalData, UserActivityData, UserSleepData]


class OpenAIResponse(BaseModel):
    response: str


def generate_prompt(prompt_number: int, user_data: Union[
    UserPhysicalData, UserVitalData, UserActivityData, UserSleepData]) -> str:
    if prompt_number == 1:
        return (
            "Как медицинский ассистент, проанализируй данные о физических параметрах пользователя, "
            "чтобы предоставить рекомендации и расшифровку этих данных. Вот данные о физических "
            "параметрах пользователя на сегодняшний день:\n\n"
            f"- Рост: {user_data.height} см\n"
            f"- Вес: {user_data.weight} кг\n"
            f"- Процент жира: {user_data.body_fat_percentage}%\n"
            f"- Индекс массы тела (ИМТ): {user_data.bmi}\n"
            f"- Базальный метаболизм (BMR): {user_data.basal_metabolism} ккал/день\n"
            f"- Общая ежедневная потребность в энергии (TDEE): {user_data.tdee} ккал/день\n"
            f"- Среднее количество калорий, сжигаемых в день при ходьбе: {user_data.avg_calories_burned_walking} ккал\n"
            f"- Среднее количество калорий, сжигаемых в день на тренировке: "
            f"{user_data.avg_calories_burned_exercise} ккал\n"
            f"- Количество калорий для поддержания веса: {user_data.maintenance_calories} ккал/день\n"
            f"- Количество калорий для понижения веса: {user_data.weight_loss_calories} ккал/день\n"
            f"- Количество калорий для набора веса: {user_data.weight_gain_calories} ккал/день\n"
            f"- Индекс висцерального жира: {user_data.visceral_fat_index}\n\n"
            f"- Возраст пользователя: {user_data.age}\n\n"
            f"- Пол пользователя: {user_data.gender}\n\n"
            "Основываясь на этих данных, предоставьте подробный анализ и рекомендации для пользователя."
        )
    elif prompt_number == 2:
        return (
            "Как медицинский ассистент, проанализируй данные о жизненных показателях пользователя,"
            " чтобы предоставить рекомендации и расшифровку этих данных. Вот данные о жизненных"
            " показателях пользователя на сегодняшний день:\n\n"
            f"- Покойный пульс: {user_data.resting_pulse} уд/мин\n"
            f"- Средний пульс: {user_data.average_pulse} уд/мин\n"
            f"- Медианный пульс: {user_data.median_pulse} уд/мин\n"
            f"- Среднее артериальное давление (МАР): {user_data.map} мм рт.ст.\n"
            f"- Пульсовое давление: {user_data.pp} мм рт.ст.\n\n"
            f"- Возраст пользователя: {user_data.age}\n\n"
            f"- Пол пользователя: {user_data.gender}\n\n"
            "Основываясь на этих данных, предоставьте подробный анализ и рекомендации для пользователя."
        )
    elif prompt_number == 3:
        return (
            "Как фитнес-консультант, проанализируй данные об активности пользователя, "
            "чтобы предоставить рекомендации и расшифровку этих данных. Вот данные об "
            "активности пользователя на сегодняшний день:\n\n"
            f"- Тенденции количества шагов: {user_data.step_trends}\n"
            f"- Целевые калории: {user_data.target_calories}\n"
            f"- Возраст: {user_data.age} лет\n"
            f"- Пол: {user_data.gender}\n\n"
            "Основываясь на этих данных, предоставьте подробный анализ и рекомендации "
            "для пользователя по количеству шагов в день."
        )
    elif prompt_number == 4:
        return (
            "Как специалист по сну, проанализируй данные о сне пользователя,"
            " чтобы предоставить рекомендации и расшифровку этих данных. "
            "Вот данные о сне пользователя на сегодняшний день:\n\n"
            f"- Показатель качества сна: {user_data.sleep_quality_index}\n"
            f"- Эффективность сна: {user_data.sleep_efficiency}%\n"
            f"- Латентность сна: {user_data.sleep_latency} минут\n"
            f"- Процент глубокого сна: {user_data.deep_sleep_percentage}%\n"
            f"- Процент REM сна: {user_data.rem_sleep_percentage}%\n"
            f"- Процент поверхностного сна: {user_data.light_sleep_percentage}%\n\n"
            f"- Возраст: {user_data.age} лет\n"
            f"- Пол: {user_data.gender}\n\n"
            "Основываясь на этих данных, предоставьте подробный анализ и "
            "рекомендации для улучшения качества сна пользователя."
        )
    else:
        raise ValueError("Invalid prompt number")


@app.post("/generate_text/", response_model=OpenAIResponse)
def generate_text(request: OpenAIRequest) -> OpenAIResponse:
    try:
        prompt = generate_prompt(request.prompt_number, request.user_data)
        response = openai.chat.completions.create(
            model="gpt-4",  # Выбор модели
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_text = response.choices[0].message.content.strip()
        return OpenAIResponse(response=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in OpenAI API request: {str(e)}")


class SleepData(BaseModel):
    total_sleep: float  # Общая продолжительность сна (в часах)
    wake_time: int  # Время бодрствования (в минутах)
    light_sleep: float  # Время поверхностного сна (в часах)
    deep_sleep: float  # Время крепкого сна (в часах)
    rem_sleep: float  # Время быстрого сна (в часах)
    oxygen_level: float  # Средний уровень кислорода в крови (%)
    sleep_start: str  # Время начала сна ('HH:MM' формат)
    sleep_end: str  # Время окончания сна ('HH:MM' формат)


class SleepMetricsResponse(BaseModel):
    sleep_score: float
    sleep_efficiency: float
    sleep_latency: float
    deep_sleep_percent: float
    rem_sleep_percent: float
    light_sleep_percent: float


def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def calculate_sleep_score(total_sleep, wake_time, light_sleep, deep_sleep, rem_sleep, oxygen_level, sleep_start,
                          sleep_end):
    # Установим весовые коэффициенты
    weights = {
        'total_sleep': 0.25,
        'deep_sleep': 0.25,
        'rem_sleep': 0.20,
        'wake_time': 0.10,
        'light_sleep': 0.10,
        'oxygen_level': 0.10
    }

    # Нормализация параметров
    total_sleep_norm = normalize(total_sleep, 4, 8)  # Норма сна: 4-10 часов
    deep_sleep_norm = normalize(deep_sleep, 1, 3)  # Норма глубокого сна: 1-3 часа
    rem_sleep_norm = normalize(rem_sleep, 1, 2)  # Норма REM сна: 1-2 часа
    wake_time_norm = 1 - normalize(wake_time, 0, 60)  # Норма времени бодрствования: 0-60 минут
    light_sleep_norm = normalize(light_sleep, 2, 5)  # Норма поверхностного сна: 2-5 часов
    oxygen_level_norm = normalize(oxygen_level, 90, 100)  # Норма уровня кислорода в крови: 90-100%

    # Рассчитываем итоговый показатель сна
    sleep_score = (weights['total_sleep'] * total_sleep_norm +
                   weights['deep_sleep'] * deep_sleep_norm +
                   weights['rem_sleep'] * rem_sleep_norm +
                   weights['wake_time'] * wake_time_norm +
                   weights['light_sleep'] * light_sleep_norm +
                   weights['oxygen_level'] * oxygen_level_norm)

    return sleep_score * 100  # Приведение к шкале от 0 до 100


def calculate_additional_metrics(total_sleep, wake_time, light_sleep, deep_sleep, rem_sleep, sleep_start, sleep_end):
    # Если время сна перешло через полночь, скорректируем время в постели
    if sleep_end < sleep_start:
        sleep_end += datetime.timedelta(days=1)
    # Эффективность сна
    time_in_bed = (sleep_end - sleep_start).total_seconds() / 3600  # Время в постели (в часах)
    sleep_efficiency = (total_sleep / time_in_bed) * 100

    # Латентность сна
    sleep_latency = (sleep_start - datetime.datetime.combine(sleep_start.date(), datetime.time(22,
                                                                                               0))).total_seconds() / 60  # Время засыпания

    # Соотношение фаз сна
    deep_sleep_percent = (deep_sleep / total_sleep) * 100
    rem_sleep_percent = (rem_sleep / total_sleep) * 100
    light_sleep_percent = (light_sleep / total_sleep) * 100

    return {
        'sleep_efficiency': sleep_efficiency,
        'sleep_latency': sleep_latency,
        'deep_sleep_percent': deep_sleep_percent,
        'rem_sleep_percent': rem_sleep_percent,
        'light_sleep_percent': light_sleep_percent
    }


@app.post("/calculate_sleep_metrics/", response_model=SleepMetricsResponse)
def calculate_sleep_metrics(request: SleepData) -> SleepMetricsResponse:
    try:
        sleep_start = datetime.datetime.strptime(request.sleep_start, '%H:%M')
        sleep_end = datetime.datetime.strptime(request.sleep_end, '%H:%M')

        sleep_score = calculate_sleep_score(
            total_sleep=request.total_sleep,
            wake_time=request.wake_time,
            light_sleep=request.light_sleep,
            deep_sleep=request.deep_sleep,
            rem_sleep=request.rem_sleep,
            oxygen_level=request.oxygen_level,
            sleep_start=sleep_start,
            sleep_end=sleep_end
        )

        additional_metrics = calculate_additional_metrics(
            total_sleep=request.total_sleep,
            wake_time=request.wake_time,
            light_sleep=request.light_sleep,
            deep_sleep=request.deep_sleep,
            rem_sleep=request.rem_sleep,
            sleep_start=sleep_start,
            sleep_end=sleep_end
        )

        return SleepMetricsResponse(
            sleep_score=sleep_score,
            sleep_efficiency=additional_metrics['sleep_efficiency'],
            sleep_latency=additional_metrics['sleep_latency'],
            deep_sleep_percent=additional_metrics['deep_sleep_percent'],
            rem_sleep_percent=additional_metrics['rem_sleep_percent'],
            light_sleep_percent=additional_metrics['light_sleep_percent']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in calculate_sleep_metrics: {str(e)}")


class StepData(BaseModel):
    date: str
    steps: int


class StepTrendsRequest(BaseModel):
    step_data: List[StepData]
    weight_kg: float
    height_cm: float
    age_years: int
    gender: str  # 'male' or 'female'
    activity_level: str  # 'sedentary', 'lightly_active', 'moderately_active', 'very_active', 'extra_active'
    target_calories: float


class StepTrendsResponse(BaseModel):
    mean_steps: float
    steps_needed: float


def calculate_steps(weight_kg, height_cm, age_years, gender, activity_level, target_calories):
    # Рассчитываем BMR
    if gender == 'male':
        BMR = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age_years)
    elif gender == 'female':
        BMR = 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age_years)

    # Уровни активности (PAL)
    activity_levels = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }

    # Рассчитываем TDEE
    PAL = activity_levels[activity_level]
    TDEE = BMR * PAL

    # Рассчитываем количество шагов для достижения целевых калорий
    steps_needed = target_calories / 0.04

    return steps_needed


@app.post("/calculate_step_trends/", response_model=StepTrendsResponse)
def calculate_step_trends(request: StepTrendsRequest) -> StepTrendsResponse:
    try:
        # Преобразуем данные шагов в DataFrame
        step_data = [{"date": s.date, "steps": s.steps} for s in request.step_data]
        df = pd.DataFrame(step_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Рассчитываем среднее количество шагов за день
        daily_steps = df.resample('D').sum()
        mean_steps = daily_steps['steps'].mean()

        # Вызываем функцию calculate_steps
        steps_needed = calculate_steps(
            weight_kg=request.weight_kg,
            height_cm=request.height_cm,
            age_years=request.age_years,
            gender=request.gender,
            activity_level=request.activity_level,
            target_calories=request.target_calories
        )

        return StepTrendsResponse(
            mean_steps=mean_steps,
            steps_needed=steps_needed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in calculate_step_trends: {str(e)}")


class PulseData(BaseModel):
    time: str
    pulse: int


class RestingPulseRequest(BaseModel):
    pulse_data: List[PulseData]


class LifeMetricsRequest(BaseModel):
    systolic_pressure: float
    diastolic_pressure: float
    resting_heart_rate: float
    max_heart_rate: float
    oxygen_levels: List[float]  # [resting_oxygen_level, active_oxygen_level]


def calculate_pulse_metrics(pulse_data: List[Dict[str, int]]) -> Dict[str, Union[float, None]]:
    try:
        df = pd.DataFrame(pulse_data)

        if 'time' not in df.columns or 'pulse' not in df.columns:
            raise ValueError("Data must contain 'time' and 'pulse' columns")

        df['time'] = pd.to_datetime(df['time'])
        df['pulse'] = pd.to_numeric(df['pulse'])

        mean_pulse = df['pulse'].mean()
        median_pulse = df['pulse'].median()

        active_threshold = mean_pulse + 15
        df['active'] = df['pulse'] > active_threshold

        resting_threshold = median_pulse - 5
        df['resting'] = df['pulse'] < resting_threshold

        resting_periods = df[df['resting']]
        resting_pulse = resting_periods['pulse'].mean() if not resting_periods.empty else None

        return {
            'mean_pulse': mean_pulse,
            'median_pulse': median_pulse,
            'resting_pulse': resting_pulse
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in calculate_pulse_metrics: {str(e)}")


def calculate_map(systolic: float, diastolic: float) -> float:
    return diastolic + (systolic - diastolic) / 3


def calculate_pp(systolic: float, diastolic: float) -> float:
    return systolic - diastolic


def calculate_recovery_heart_rate(rhr: float, mhr: float, max_pulse: float) -> float:
    return mhr - rhr


@app.post("/calculate_resting_pulse/")
def calculate_resting_pulse(request: RestingPulseRequest) -> Dict[str, Union[float, None]]:
    pulse_data = [{"time": p.time, "pulse": p.pulse} for p in request.pulse_data]  # Обновляем преобразование данных
    pulse_metrics = calculate_pulse_metrics(pulse_data)
    return {
        'mean_pulse': pulse_metrics['mean_pulse'],
        'median_pulse': pulse_metrics['median_pulse'],
        'resting_pulse': pulse_metrics['resting_pulse']
    }


@app.post("/calculate_life_metrics/")
def calculate_life_metrics(request: LifeMetricsRequest) -> Dict[str, float]:
    try:
        map_value = calculate_map(request.systolic_pressure, request.diastolic_pressure)
        pp_value = calculate_pp(request.systolic_pressure, request.diastolic_pressure)
        recovery_hr = calculate_recovery_heart_rate(request.resting_heart_rate, request.max_heart_rate,
                                                    request.max_heart_rate)  # Use max_heart_rate for recovery

        return {
            'map': map_value,
            'pp': pp_value,
            'recovery_heart_rate': recovery_hr,
            'average_resting_oxygen_level': request.oxygen_levels[0],
            'average_active_oxygen_level': request.oxygen_levels[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


def calculate_bmi(weight: float, height: float) -> float:
    height_m = height / 100
    return weight / (height_m ** 2)


def calculate_bmr(weight: float, height: float, age: int, gender: str, basal_metabolism: int = 0) -> float:
    if basal_metabolism == 0:
        if gender == 'male':
            return (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            return (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        return basal_metabolism


def calculate_tdee(bmr: float, activity_level: float) -> float:
    return bmr * activity_level


def calculate_avg_calories_burned_per_day(height: float, weight: float, avg_steps_per_week: float) -> float:
    return (height / 100 * 0.414) * (0.57 * weight / 1000) * avg_steps_per_week


def calculate_caloric_needs(tdee: float, avg_calories_burned_per_day: float, avg_calories_burned_per_week: float) -> \
        Dict[str, float]:
    avg_calories_burned_per_week_day = avg_calories_burned_per_week / 7
    maintenance_calories = tdee + avg_calories_burned_per_day + avg_calories_burned_per_week_day
    weight_loss_calories = maintenance_calories - 750
    weight_gain_calories = maintenance_calories + 325
    return {
        'maintenance_calories': maintenance_calories,
        'weight_loss_calories': weight_loss_calories,
        'weight_gain_calories': weight_gain_calories
    }


def calculate_visceral_fat_index(bmi: float, body_fat_percentage: float, age: int, gender: str,
                                 activity_level: float) -> float:
    gender_factor = -1 if gender == 'female' else 1
    return bmi * 1.27 + 0.13 * body_fat_percentage - 11.2 + age * 0.12 + gender_factor + activity_level


@app.post("/calculate_health_metrics/")
def calculate_health_metrics(request: HealthMetricsRequest) -> Dict[str, float]:
    try:
        bmi = calculate_bmi(request.weight, request.height)
        bmr = calculate_bmr(request.weight, request.height, request.age, request.gender, request.basal_metabolism)
        tdee = calculate_tdee(bmr, request.activity_level)
        avg_calories_burned_per_day = calculate_avg_calories_burned_per_day(request.height, request.weight,
                                                                            request.avg_steps_per_week)
        caloric_needs = calculate_caloric_needs(tdee, avg_calories_burned_per_day, request.avg_calories_burned_per_week)
        visceral_fat_index = calculate_visceral_fat_index(bmi, request.body_fat_percentage, request.age, request.gender,
                                                          request.activity_level)

        return {
            'BMI': bmi,
            'BMR': bmr,
            'TDEE': tdee,
            'average_calories_burned_per_day': avg_calories_burned_per_day,
            'maintenance_calories': caloric_needs['maintenance_calories'],
            'weight_loss_calories': caloric_needs['weight_loss_calories'],
            'weight_gain_calories': caloric_needs['weight_gain_calories'],
            'visceral_fat_index': visceral_fat_index
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
