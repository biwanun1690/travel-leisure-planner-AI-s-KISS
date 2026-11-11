```python
# -*- coding: utf-8 -*-
"""
Агент-Гид для мультиагентного планировщика путешествий.

Основа: агент GuideAgent и контракты из файла «агенты (2).ipynb».
Стиль: Pydantic + Cloud.ru/Qwen + run_safe(), как в коде сокомандников.

Метрики (согласно дорожной карте):
1. Semantic Relevance Score
2. Faithfulness Score
3. Hallucination Rate
4. Schema Compliance Rate
5. Context Precision & Recall
"""

# В Google Colab при необходимости:
# %pip install -q -U langchain langchain-openai pydantic

import os
import json
import time
from getpass import getpass
from typing import Literal

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictFloat
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ============================================================
# 1. ПОДКЛЮЧЕНИЕ К CLOUD.RU
# ============================================================

def get_llm(temperature: float = 0.2, max_tokens: int = 4000) -> ChatOpenAI:
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

class Hobby(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    location_type: Literal[
        "open_horizon", "water", "mountains",
        "city_culture", "forest", "indoor"
    ]
    priority_weight: StrictInt = Field(ge=1, le=10)


class Constraint(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    type: Literal["hard", "soft"]
    category: Literal["health", "safety", "kids", "logistics", "comfort"]


class UserProfile(BaseModel):
    """Профиль клиента от Аналитика (вход Гида)."""
    hobbies: list[Hobby] = Field(min_length=1, max_length=10)
    constraints: list[Constraint] = Field(default_factory=list)
    fitness: Literal["low", "medium", "high"] = "medium"
    group_size: StrictInt = Field(default=2, ge=1, le=50)
    has_children: StrictBool = False
    budget_rub: StrictFloat = Field(default=100000, gt=0)
    destination_hint: str = Field(default="", max_length=100)
    duration_days: StrictInt = Field(default=5, ge=1, le=30)


class Place(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: Literal[
        "sight", "museum", "restaurant", "nature",
        "activity", "spa", "beach", "shop"
    ]
    description: str = Field(max_length=300)
    website_url: str = Field(description="Ссылка на официальный сайт")
    maps_url: str = Field(description="Ссылка на геолокацию (Google Maps / Yandex)")
    location_type: Literal[
        "open_horizon", "water", "mountains",
        "city_culture", "forest", "indoor"
    ]
    estimated_cost_rub: StrictFloat = Field(ge=0)
    duration_hours: StrictFloat = Field(ge=0.5, le=12)
    rating: StrictFloat = Field(ge=1, le=5)
    suitable_for_children: StrictBool = True


class GuideOutput(BaseModel):
    """Выход Гида — список мест под профиль."""
    destination: str
    places: list[Place] = Field(min_length=3, max_length=20)
    tips: list[str] = Field(max_length=5)


# ============================================================
# 3. СИСТЕМНЫЙ ПРОМПТ ГИДА
# ============================================================

GUIDE_PROMPT = """
Ты — Агент-Гид с глубокими знаниями о достопримечательностях.

ЗАДАЧА: подобрать места под профиль клиента.

ТРЕБОВАНИЯ К МЕСТАМ:
1. Каждое место ОБЯЗАТЕЛЬНО должно иметь:
   - name: реальное название
   - category: sight/museum/restaurant/nature/activity/spa/beach/shop
   - description: краткое (до 300 символов)
   - website_url: РЕАЛЬНАЯ ссылка на официальный сайт (https://...)
   - maps_url: РЕАЛЬНАЯ ссылка на геолокацию (Google Maps или Yandex Maps)
   - location_type: одно из open_horizon/water/mountains/city_culture/forest/indoor
   - estimated_cost_rub: примерная стоимость на 1 человека
   - duration_hours: сколько часов занимает
   - rating: 1..5
   - suitable_for_children: true/false

2. Покрой ВСЕ хобби клиента (особенно с высоким priority_weight).
3. Учитывай constraints (если children — только suitable_for_children=true).
4. Учитывай fitness (если low — без тяжёлых активностей).
5. Минимум 3 места, максимум 20.
6. Добавь 3-5 практических советов (tips).

ФОРМАТ ССЫЛОК:
- website_url: "https://example.com" (полный URL)
- maps_url: "https://maps.google.com/?q=..." или "https://yandex.ru/maps/?text=..."

Если не знаешь точной ссылки — используй поиск Google Maps:
"https://www.google.com/maps/search/+НАЗВАНИЕ+ГОРОД"

НЕ ВЫДУМЫВАЙ несуществующие места. Если направление не указано — выбери популярное под хобби.

Верни структурированный ответ строго по схеме GuideOutput.
"""


# ============================================================
# 4. БАЗОВЫЙ КЛАСС АГЕНТА И АГЕНТ-ГИД
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
                print(f"[{self.name}] ✓ Результат получен за {elapsed:.2f} сек.")
            return result, None, elapsed
        except Exception as e:
            elapsed = time.perf_counter() - started
            if verbose:
                print(f"[{self.name}] ✗ {type(e).__name__}: {e}")
            return None, e, elapsed


class GuideAgent(BaseAgent):
    name = "guide"
    system_prompt = GUIDE_PROMPT
    output_schema = GuideOutput


# ============================================================
# 5. ДЕТЕРМИНИРОВАННЫЕ ПРОВЕРКИ И МЕТРИКИ
# ============================================================

def evaluate_semantic_relevance(guide: GuideOutput, profile: UserProfile) -> float:
    """
    Semantic Relevance Score: доля предложенных мест, 
    соответствующих типам локаций из хобби клиента.
    """
    requested_types = {h.location_type for h in profile.hobbies}
    if not requested_types or not guide.places:
        return 0.0

    relevant_places = sum(
        1 for p in guide.places 
        if p.location_type in requested_types or p.category == "restaurant"
    )
    return relevant_places / len(guide.places)


def evaluate_faithfulness(guide: GuideOutput) -> float:
    """
    Faithfulness Score: доля мест с валидными URL-адресами 
    (как прокси реальности существования места).
    """
    if not guide.places:
        return 0.0

    valid_urls = sum(
        1 for p in guide.places 
        if p.website_url.startswith("http") and p.maps_url.startswith("http")
    )
    return valid_urls / len(guide.places)


def evaluate_context_precision_and_recall(guide: GuideOutput, profile: UserProfile) -> tuple[float, float]:
    """
    Context Precision: доля релевантных мест среди всех предложенных.
    Context Recall: доля покрытых типов локаций из запрошенных.
    """
    requested_types = {h.location_type for h in profile.hobbies}
    
    # Precision
    relevant_places = sum(
        1 for p in guide.places 
        if p.location_type in requested_types
    )
    precision = relevant_places / len(guide.places) if guide.places else 0.0

    # Recall
    provided_types = {p.location_type for p in guide.places}
    recall = len(requested_types.intersection(provided_types)) / len(requested_types) if requested_types else 0.0

    return precision, recall


def evaluate_guide(
    agent: GuideAgent,
    test_cases: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Запускает Гида на тестовых профилях и агрегирует метрики.
    """
    total = len(test_cases)
    schema_ok = 0
    semantic_scores = []
    faithfulness_scores = []
    precision_scores = []
    recall_scores = []
    first_draft_times = []

    for i, case in enumerate(test_cases, 1):
        if verbose:
            print(f"\nТест {i}/{total}: запуск Гида")

        result, error, elapsed = agent.run_safe(
            case["profile"],
            verbose=verbose,
        )

        if error or result is None:
            if verbose:
                print(f"Тест {i}: ОШИБКА — {error}")
            continue

        schema_ok += 1
        first_draft_times.append(elapsed)

        profile = UserProfile.model_validate(case["profile"])
        
        semantic = evaluate_semantic_relevance(result, profile)
        faithful = evaluate_faithfulness(result)
        precision, recall = evaluate_context_precision_and_recall(result, profile)

        semantic_scores.append(semantic)
        faithfulness_scores.append(faithful)
        precision_scores.append(precision)
        recall_scores.append(recall)

        if verbose:
            print(f"Semantic Relevance: {semantic:.2%}")
            print(f"Faithfulness: {faithful:.2%}")
            print(f"Context Precision: {precision:.2%}")
            print(f"Context Recall: {recall:.2%}")

    # Агрегация
    avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
    avg_faithful = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    return {
        "schema_compliance_rate": schema_ok / total if total > 0 else 0.0,
        "semantic_relevance_score": avg_semantic,
        "faithfulness_score": avg_faithful,
        "hallucination_rate": 1.0 - avg_faithful,
        "context_precision": avg_precision,
        "context_recall": avg_recall,
        "time_to_first_draft_sec": sum(first_draft_times) / len(first_draft_times) if first_draft_times else None,
        "successful_drafts": schema_ok,
        "total_tests": total,
    }


# ============================================================
# 6. ДЕМО И ТЕСТОВЫЕ ДАННЫЕ
# ============================================================

# Пример профиля клиента (вход от Аналитика)
DEMO_PROFILE = {
    "hobbies": [
        {"name": "музеи", "location_type": "city_culture", "priority_weight": 8},
        {"name": "гастрономия", "location_type": "city_culture", "priority_weight": 8},
        {"name": "SPA", "location_type": "indoor", "priority_weight": 7}
    ],
    "constraints": [],
    "fitness": "low",
    "group_size": 2,
    "has_children": False,
    "budget_rub": 150000.0,
    "destination_hint": "Стамбул",
    "duration_days": 5
}

TEST_CASES = [
    {
        "profile": DEMO_PROFILE
    }
]


# ============================================================
# 7. ДЕМОНСТРАЦИЯ
# ============================================================

def demo_guide():
    print("\n" + "=" * 65)
    print("ДЕМО: АГЕНТ-ГИД")
    print("=" * 65)

    llm = get_llm(temperature=0.2)
    guide_agent = GuideAgent(llm)

    result, error, elapsed = guide_agent.run_safe(
        DEMO_PROFILE,
        verbose=True,
    )

    if error or result is None:
        print("\nОшибка запуска:", error)
        return None

    print("\nИТОГОВЫЙ СПИСОК МЕСТ:")
    print(result.model_dump_json(indent=2, ensure_ascii=False))

    print("\n" + "=" * 65)
    print("МЕТРИКИ КАЧЕСТВА ГИДА")
    print("=" * 65)

    metrics = evaluate_guide(guide_agent, TEST_CASES, verbose=True)
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
    demo_guide()
```
