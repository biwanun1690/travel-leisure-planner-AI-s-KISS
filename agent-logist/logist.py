# -*- coding: utf-8 -*-
"""
Агент-Логист для мультиагентного планировщика путешествий.

Основа: агент LogistAgent и контракты из файла «агенты (2).ipynb».
Стиль: Pydantic + Cloud.ru/Qwen + run_safe(), как в коде сокомандников.

Метрики:
1. Temporal Feasibility
2. Hard Constraint Satisfaction Rate
3. User Intervention Rate
4. Time-to-First-Draft
"""

# В Google Colab при необходимости:
# %pip install -q -U langchain langchain-openai pydantic

import os
import json
import re
import time
from datetime import datetime, date, timedelta
from getpass import getpass
from typing import Optional, Literal

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictFloat
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ============================================================
# 1. ПОДКЛЮЧЕНИЕ К CLOUD.RU
# ============================================================

def get_llm(temperature: float = 0.1, max_tokens: int = 4000) -> ChatOpenAI:
    """Единая точка подключения к LLM cloud.ru Foundation Models."""
    api_key = os.environ.get("CLOUD_API_KEY")

    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("CLOUD_API_KEY")
        except Exception:
            pass

    if not api_key:
        api_key = getpass("Введите CLOUD_API_KEY: ").strip()

    if not api_key:
        raise RuntimeError("Не указан API-ключ Cloud.ru")

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://foundation-models.api.cloud.ru/v1",
        model="Qwen/Qwen3-Coder-Next",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=2,
    )


# ============================================================
# 2. PYDANTIC-СХЕМЫ (контракты между агентами)
# ============================================================

class UserProfile(BaseModel):
    """Минимальный контракт профиля, используемый Логистом."""
    fitness: Literal["low", "medium", "high"] = "medium"
    group_size: StrictInt = Field(default=2, ge=1, le=50)
    has_children: StrictBool = False
    budget_rub: StrictFloat = Field(default=100000, gt=0)
    destination_hint: str = ""
    duration_days: StrictInt = Field(default=5, ge=1, le=30)
    constraints: list[dict] = Field(default_factory=list)


class BudgetItem(BaseModel):
    place_name: str
    cost_rub: StrictFloat = Field(ge=0)
    covered_hobbies: list[str] = Field(default_factory=list)


class BudgetVariant(BaseModel):
    name: Literal["economy", "balanced", "premium"]
    total_cost_rub: StrictFloat = Field(ge=0)
    items: list[BudgetItem]
    hobbies_covered: StrictInt = Field(default=0, ge=0)
    hobbies_coverage_pct: StrictFloat = Field(default=0, ge=0, le=100)
    notes: str = ""


class GuidePlace(BaseModel):
    name: str
    category: str = "activity"
    description: str = ""
    website_url: str = ""
    maps_url: str = ""
    location_type: str = "city_culture"
    estimated_cost_rub: StrictFloat = Field(default=0, ge=0)
    duration_hours: StrictFloat = Field(default=2, gt=0, le=12)
    rating: StrictFloat = Field(default=4, ge=1, le=5)
    suitable_for_children: StrictBool = True


class GuideOutput(BaseModel):
    destination: str
    places: list[GuidePlace]
    tips: list[str] = Field(default_factory=list)


class DayPlan(BaseModel):
    day_number: StrictInt = Field(ge=1)
    date_hint: str = Field(default="", max_length=50)
    # Сохраняем исходный контракт notebook: activities — список словарей.
    activities: list[dict]
    total_hours: StrictFloat = Field(ge=0)
    transport_note: str = Field(default="", max_length=200)


class LogisticsOutput(BaseModel):
    """Выход Логиста — детальный маршрут."""
    destination: str
    duration_days: StrictInt = Field(ge=1)
    start_date: str
    daily_plans: list[DayPlan] = Field(min_length=1)
    accommodation: dict
    transport_plan: dict
    total_cost_rub: StrictFloat = Field(ge=0)
    files_to_prepare: list[str]
    # Новое поле для измеримой проверки ограничений.
    # Для каждого HARD-ограничения агент обязан дать verdict и объяснение.
    hard_constraint_checks: list[dict] = Field(default_factory=list)


# ============================================================
# 3. СИСТЕМНЫЙ ПРОМПТ ЛОГИСТА
# ============================================================

LOGIST_PROMPT = """
Ты — Агент-Логист мультиагентной системы планирования путешествий.

ЗАДАЧА:
Распределить места из выбранного варианта Бюджетника по дням и составить
реалистичное расписание с временными интервалами, длительностью и транспортом.

ВХОДНЫЕ ДАННЫЕ:
- profile: профиль клиента (fitness, group_size, has_children, constraints,
  budget_rub, destination_hint, duration_days)
- guide: направление и данные мест от Гида
- budget_variant: выбранный вариант Бюджетника с items.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. Распредели ВСЕ места из budget_variant.items по daily_plans.
   Не добавляй новые места, которых нет в items.
2. Число дней должно соответствовать profile.duration_days.
3. В каждый день планируй максимум 2-3 активности, оставляй время на отдых и еду.
4. Для каждой активности укажи:
   time ("09:00-11:00"), place (точное название из items), type,
   cost_rub, duration_hours, notes.
5. Временные интервалы не должны пересекаться. Продолжительность интервала
   должна быть не меньше duration_hours места. Оставляй разумное время
   на перемещение между разными местами (обычно 15-60 минут).
6. Учитывай duration_hours каждого места и группируй географически близкие места.
   Не ставь заведомо удалённые районы в один плотный блок.
7. Если fitness="low", не более 4-5 часов активностей в день.
   Если has_children=true, не более 3-4 часов активностей в день,
   чаще делай перерывы.
8. Учитывай ВСЕ ограничения profile.constraints:
   - type="hard" — обязательное ограничение, его нельзя нарушать;
   - type="soft" — пожелание, старайся соблюдать.
9. Для каждого HARD-ограничения добавь объект в hard_constraint_checks:
   {"constraint": "текст ограничения", "satisfied": true/false,
    "explanation": "почему ограничение соблюдено или что нужно изменить"}.
   Не ставь satisfied=true, если не можешь подтвердить соблюдение.
10. transport_note: способ перемещения между точками.
11. accommodation: тип жилья, район и ориентировочная цена за ночь.
12. transport_plan: как добраться до направления и местный транспорт.
13. files_to_prepare: документы и бронирования, которые нужно подготовить.
14. total_cost_rub: сумма стоимости выбранных мест + проживания + транспорта.
    Не выдумывай стоимость мест: используй цены из budget_variant.items.
15. start_date: если дата не указана, поставь дату через 14 дней от текущей даты.

ФОРМАТ activity:
{"time":"09:00-11:00", "place":"Название места", "type":"museum",
 "cost_rub":1500, "duration_hours":2.0, "notes":"..."}

Верни структурированный ответ строго по схеме LogisticsOutput.
"""


# ============================================================
# 4. БАЗОВЫЙ КЛАСС АГЕНТА И АГЕНТ-ЛОГИСТ
# ============================================================

class BaseAgent:
    name = "base"
    system_prompt = ""
    output_schema = BaseModel

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(self.output_schema)
        self.system_msg = SystemMessage(content=self.system_prompt)

    def run(self, user_input: str | dict) -> BaseModel:
        if isinstance(user_input, dict):
            content = json.dumps(user_input, ensure_ascii=False, indent=2)
        else:
            content = user_input

        return self.llm.invoke([
            self.system_msg,
            HumanMessage(content=content)
        ])

    def run_safe(self, user_input, verbose: bool = False):
        """Безопасный запуск с замером времени и обработкой ошибок."""
        started = time.perf_counter()
        try:
            result = self.run(user_input)
            elapsed = time.perf_counter() - started
            if verbose:
                print(f"[{self.name}] ✓ Первый черновик получен за {elapsed:.2f} сек.")
            return result, None, elapsed
        except Exception as e:
            elapsed = time.perf_counter() - started
            if verbose:
                print(f"[{self.name}] ✗ {type(e).__name__}: {e}")
            return None, e, elapsed


class LogistAgent(BaseAgent):
    name = "logist"
    system_prompt = LOGIST_PROMPT
    output_schema = LogisticsOutput


# ============================================================
# 5. ПОДГОТОВКА ВХОДА (контракт с Бюджетником и Гидом)
# ============================================================

def build_logist_payload(
    profile: UserProfile,
    guide: GuideOutput,
    budget: BudgetOutput | BudgetVariant,
) -> dict:
    """Формирует вход Логиста из контрактов предыдущих агентов."""
    if isinstance(budget, BudgetOutput):
        variant = next(
            v for v in budget.variants if v.name == budget.recommended_variant
        )
    else:
        variant = budget

    return {
        "profile": profile.model_dump(mode="json"),
        "guide": {
            "destination": guide.destination,
            "places": [p.model_dump(mode="json") for p in guide.places],
        },
        "budget_variant": variant.model_dump(mode="json"),
    }


# ============================================================
# 6. ДЕТЕРМИНИРОВАННЫЕ ПРОВЕРКИ И МЕТРИКИ
# ============================================================

TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})\s*$")


def parse_time_interval(value: str):
    """Преобразует '09:00-11:30' в минуты от начала суток."""
    if not isinstance(value, str):
        return None
    match = TIME_RE.match(value)
    if not match:
        return None
    h1, m1, h2, m2 = map(int, match.groups())
    if h1 > 23 or h2 > 24 or m1 > 59 or m2 > 59:
        return None
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if end <= start:
        return None
    return start, end


def evaluate_temporal_feasibility(
    logistics: LogisticsOutput,
    profile: UserProfile,
) -> float:
    """
    Доля дней, в которых расписание временно выполнимо:
    - корректные интервалы времени;
    - нет пересечений;
    - интервал покрывает duration_hours;
    - дневная нагрузка укладывается в ограничения профиля.
    """
    if not logistics.daily_plans:
        return 0.0

    feasible_days = 0
    for day in logistics.daily_plans:
        intervals = []
        day_ok = True
        activity_hours = 0.0

        for activity in day.activities:
            interval = parse_time_interval(activity.get("time", ""))
            duration = activity.get("duration_hours")

            if interval is None or not isinstance(duration, (int, float)):
                day_ok = False
                continue

            start, end = interval
            interval_hours = (end - start) / 60
            if interval_hours + 1e-9 < float(duration):
                day_ok = False

            intervals.append((start, end))
            activity_hours += float(duration)

        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                day_ok = False

        daily_limit = 12.0
        if profile.fitness == "low":
            daily_limit = min(daily_limit, 5.0)
        if profile.has_children:
            daily_limit = min(daily_limit, 4.0)

        if activity_hours > daily_limit:
            day_ok = False

        if day_ok:
            feasible_days += 1

    return feasible_days / len(logistics.daily_plans)


def evaluate_hard_constraint_satisfaction(
    logistics: LogisticsOutput,
    profile: UserProfile,
) -> float:
    """
    Доля HARD-ограничений с явным положительным verdict от Логиста.
    Если HARD-ограничений нет, возвращает 1.0 (нарушений не обнаружено).
    """
    hard_constraints = [
        c for c in profile.constraints
        if str(c.get("type", "")).lower() == "hard"
    ]
    if not hard_constraints:
        return 1.0

    checks = logistics.hard_constraint_checks or []
    satisfied_texts = {
        str(item.get("constraint", "")).strip().lower()
        for item in checks
        if item.get("satisfied") is True
    }

    satisfied = 0
    for constraint in hard_constraints:
        text = str(constraint.get("text", "")).strip().lower()
        if text and text in satisfied_texts:
            satisfied += 1

    return satisfied / len(hard_constraints)


def calculate_user_intervention_rate(
    intervention_runs: int,
    total_runs: int,
) -> float:
    """
    User Intervention Rate = доля запусков, где потребовалось вмешательство
    пользователя (уточнение/исправление исходных данных).
    Число intervention_runs фиксируется оркестратором или тестовым сценарием.
    """
    if total_runs <= 0:
        return 0.0
    return intervention_runs / total_runs


# ============================================================
# 7. ОЦЕНКА ЛОГИСТА (аналогично evaluate_analyst / test_logist)
# ============================================================

def evaluate_logist(
    agent: LogistAgent,
    test_cases: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Запускает Логиста на тестовых payload и агрегирует четыре метрики.

    Формат test_case:
    {
        "payload": {"profile": ..., "guide": ..., "budget_variant": ...},
        "intervention_required": True/False
    }

    Time-to-First-Draft измеряется от начала вызова до первого успешного
    структурированного ответа Логиста, а не до завершения всей мультиагентной
    цепочки с проверкой Критика.
    """
    total = len(test_cases)
    temporal_scores = []
    hard_scores = []
    first_draft_times = []
    intervention_runs = 0
    success_count = 0

    for i, case in enumerate(test_cases, 1):
        if case.get("intervention_required", False):
            intervention_runs += 1

        if verbose:
            print(f"\nТест {i}/{total}: запуск Логиста")

        result, error, elapsed = agent.run_safe(
            case["payload"],
            verbose=verbose,
        )

        if error or result is None:
            if verbose:
                print(f"Тест {i}: ОШИБКА — {error}")
            continue

        success_count += 1
        first_draft_times.append(elapsed)

        profile = UserProfile.model_validate(case["payload"]["profile"])
        temporal = evaluate_temporal_feasibility(result, profile)
        hard_rate = evaluate_hard_constraint_satisfaction(result, profile)

        temporal_scores.append(temporal)
        hard_scores.append(hard_rate)

        if verbose:
            print(f"Temporal Feasibility: {temporal:.2%}")
            print(f"Hard Constraint Satisfaction Rate: {hard_rate:.2%}")

    return {
        "temporal_feasibility": (
            sum(temporal_scores) / len(temporal_scores) if temporal_scores else 0.0
        ),
        "hard_constraint_satisfaction_rate": (
            sum(hard_scores) / len(hard_scores) if hard_scores else 0.0
        ),
        "user_intervention_rate": calculate_user_intervention_rate(
            intervention_runs, total
        ),
        "time_to_first_draft_sec": (
            sum(first_draft_times) / len(first_draft_times)
            if first_draft_times else None
        ),
        "successful_drafts": success_count,
        "total_tests": total,
    }


# ============================================================
# 8. ДЕМО И ТЕСТОВЫЕ ДАННЫЕ
# ============================================================

# Пример структуры входа. В реальном запуске подставьте сюда
# profile, guide и budget_variant от предыдущих агентов.
DEMO_PAYLOAD = {
    "profile": {
        "fitness": "low",
        "group_size": 2,
        "has_children": False,
        "budget_rub": 150000.0,
        "destination_hint": "Стамбул",
        "duration_days": 3,
        "constraints": [
            {
                "text": "Не планировать ранний подъём",
                "type": "hard",
                "category": "comfort"
            }
        ]
    },
    "guide": {
        "destination": "Стамбул",
        "places": [
            {
                "name": "Айя-София",
                "category": "museum",
                "description": "Исторический памятник",
                "website_url": "https://example.com",
                "maps_url": "https://maps.google.com",
                "location_type": "city_culture",
                "estimated_cost_rub": 1500.0,
                "duration_hours": 2.0,
                "rating": 4.8,
                "suitable_for_children": True
            },
            {
                "name": "Дворец Топкапы",
                "category": "sight",
                "description": "Дворцовый комплекс",
                "website_url": "https://example.com",
                "maps_url": "https://maps.google.com",
                "location_type": "city_culture",
                "estimated_cost_rub": 2000.0,
                "duration_hours": 2.5,
                "rating": 4.7,
                "suitable_for_children": True
            }
        ]
    },
    "budget_variant": {
        "name": "balanced",
        "total_cost_rub": 3500.0,
        "items": [
            {
                "place_name": "Айя-София",
                "cost_rub": 1500.0,
                "covered_hobbies": ["музеи"]
            },
            {
                "place_name": "Дворец Топкапы",
                "cost_rub": 2000.0,
                "covered_hobbies": ["история"]
            }
        ],
        "hobbies_covered": 2,
        "hobbies_coverage_pct": 100.0,
        "notes": "Сбалансированный вариант"
    }
}

TEST_CASES = [
    {
        "payload": DEMO_PAYLOAD,
        # Для данного теста уточнение от пользователя не потребовалось.
        "intervention_required": False,
    }
]


# ============================================================
# 9. ДЕМОНСТРАЦИЯ
# ============================================================

def demo_logist():
    print("\n" + "=" * 65)
    print("ДЕМО: АГЕНТ-ЛОГИСТ")
    print("=" * 65)

    llm = get_llm(temperature=0.1)
    logist_agent = LogistAgent(llm)

    result, error, elapsed = logist_agent.run_safe(
        DEMO_PAYLOAD,
        verbose=True,
    )

    if error or result is None:
        print("\nОшибка запуска:", error)
        return None

    print("\nИТОГОВЫЙ МАРШРУТ:")
    print(result.model_dump_json(indent=2, ensure_ascii=False))

    print("\n" + "=" * 65)
    print("МЕТРИКИ КАЧЕСТВА ЛОГИСТА")
    print("=" * 65)

    metrics = evaluate_logist(logist_agent, TEST_CASES, verbose=True)
    for name, value in metrics.items():
        if isinstance(value, float):
            if name.endswith("_sec"):
                print(f"{name}: {value:.2f} сек.")
            else:
                print(f"{name}: {value:.2%}")
        else:
            print(f"{name}: {value}")

    return result


if __name__ == "__main__":
    demo_logist()
