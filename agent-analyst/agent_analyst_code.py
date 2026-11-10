
# ============================================================
# 1. УСТАНОВКА ПАКЕТОВ
# ============================================================

%pip install -q -U langchain langchain-openai pydantic openai


# ============================================================
# 2. ИМПОРТЫ
# ============================================================

import json
import os
import re

from enum import Enum
from getpass import getpass
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ============================================================
# 3. ПОДКЛЮЧЕНИЕ CLOUD.RU
# ============================================================

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
    raise RuntimeError(
        "Не указан API-ключ Cloud.ru Foundation Models."
    )

BASE_URL = "https://foundation-models.api.cloud.ru/v1"
MODEL = "Qwen/Qwen3-Coder-Next"

llm = ChatOpenAI(
    api_key=api_key,
    base_url=BASE_URL,
    model=MODEL,
    temperature=0.0,
    max_tokens=2000,
    timeout=60,
    max_retries=1
)

print(f"Подключено: {BASE_URL}")
print(f"Модель: {MODEL}")


# ============================================================
# 4. PYDANTIC-СХЕМА
# ============================================================

class FitnessLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConstraintType(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class Hobby(BaseModel):

    name: str = Field(min_length=1, max_length=80)

    location_type: Literal[
        "open_horizon",
        "water",
        "mountains",
        "city_culture",
        "forest",
        "indoor"
    ]

    priority_weight: StrictInt = Field(ge=1, le=10)


class Constraint(BaseModel):

    text: str = Field(min_length=1, max_length=200)

    type: ConstraintType

    category: Literal[
        "health",
        "safety",
        "kids",
        "logistics",
        "comfort"
    ]


class UserProfile(BaseModel):
    """Структурированный профиль пользователя."""

    hobbies: list[Hobby] = Field(min_length=1, max_length=10)

    constraints: list[Constraint] = Field(default_factory=list)

    fitness: FitnessLevel = FitnessLevel.MEDIUM

    group_size: StrictInt = Field(default=2, ge=1, le=50)

    has_children: StrictBool = False

    budget_priority: StrictInt = Field(default=5, ge=1, le=10)

    comfort_priority: StrictInt = Field(default=5, ge=1, le=10)

    raw_summary: str = Field(default="", max_length=300)

    @field_validator("hobbies")
    @classmethod
    def no_duplicate_hobbies(cls, v):

        names = [h.name.lower().strip() for h in v]

        if len(set(names)) != len(names):
            raise ValueError(
                "Дублирующиеся хобби в профиле"
            )

        return v


# ============================================================
# 5. СИСТЕМНЫЙ ПРОМПТ
# ============================================================

SYSTEM_PROMPT = """
Ты — Агент-аналитик предпочтений и хобби
в мультиагентной системе планирования путешествий.

ТВОЯ ЗАДАЧА:
Извлечь из текста пользователя структурированный профиль.

Верни СТРОГО JSON без markdown и пояснений.

ПРАВИЛА ДЕКОМПОЗИЦИИ ХОББИ:

open_horizon — астрофотография, наблюдение звёзд.
water — дайвинг, снорклинг, серфинг, плавание.
mountains — скалолазание, треккинг.
city_culture — музеи, архитектура, гастрономия.
forest — хайкинг, сбор ягод.
indoor — йога, кулинария, SPA, бассейн.

Учитывай смысловые соответствия:
бассейн и плавание — одно направление;
SPA и спа — одно направление.

Не создавай отдельные хобби для одного и того же занятия.

ПРАВИЛА ОГРАНИЧЕНИЙ:

HARD — обязательные ограничения:
аллергии, травмы, дети, визы, физические ограничения.

SOFT — предпочтения:
темп, эстетика, время подъёма,
отсутствие экстремальных активностей.

ДОПУСТИМЫЕ КАТЕГОРИИ:

health — здоровье, аллергии, травмы.
safety — безопасность.
kids — дети и условия для них.
logistics — транспорт, мобильность, визы.
comfort — комфорт, темп отдыха и предпочтения.

ВАЖНО:
Используй ТОЛЬКО перечисленные категории.
Категорию pace запрещено использовать.
Предпочтения по темпу отдыха относить к comfort.

Если упомянуты дети:
has_children=true.
Добавь ограничение категории kids.

Если упомянута аллергия:
Добавь HARD-ограничение категории health.

Если упомянута коляска или ограниченная мобильность:
fitness="low".
Добавь HARD-ограничение категории logistics.

Каждое ограничение ОБЯЗАТЕЛЬНО должно быть объектом
со следующими тремя полями:

{
  "text": "Описание ограничения",
  "type": "hard",
  "category": "logistics"
}

ЗАПРЕЩЕНО возвращать ограничения простыми строками.

ФИЗИЧЕСКАЯ ФОРМА:

LOW — коляска, пожилые, отсутствие нагрузок.
MEDIUM — по умолчанию.
HIGH — спортивный поход, треккинг.

ВЕСА ПРИОРИТЕТОВ (1..10):

budget_priority:
1 — бюджет не важен.
10 — строгий бюджет.

comfort_priority:
1 — минимальный комфорт.
10 — высокий уровень комфорта.

ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ:

group_size=2, если количество не указано.
budget_priority=5, если приоритет не указан.
comfort_priority=5, если приоритет не указан.
fitness="medium", если форма не указана.
has_children=false, если дети не упомянуты.
constraints=[], если ограничений нет.

ВЕСА ХОББИ:
Целые числа от 1 до 10.

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ JSON:

hobbies
constraints
fitness
group_size
has_children
budget_priority
comfort_priority
raw_summary

ФОРМАТ ОТВЕТА:

{
  "hobbies": [
    {
      "name": "музеи",
      "location_type": "city_culture",
      "priority_weight": 8
    }
  ],
  "constraints": [
    {
      "text": "Передвижение на инвалидной коляске",
      "type": "hard",
      "category": "logistics"
    }
  ],
  "fitness": "low",
  "group_size": 2,
  "has_children": false,
  "budget_priority": 5,
  "comfort_priority": 5,
  "raw_summary": "Культурный тур с ограниченной мобильностью"
}

Не придумывай увлечения и ограничения.
Все числовые поля возвращай числами.
Возвращай только JSON.
"""


# ============================================================
# 6. ИЗВЛЕЧЕНИЕ JSON
# ============================================================

def extract_json_from_response(text) -> dict:

    if isinstance(text, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict)
            else str(block)
            for block in text
        )

    text = str(text).strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")

    if start == -1:
        raise ValueError("JSON не найден в ответе модели")

    decoder = json.JSONDecoder()

    try:
        data, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Некорректный JSON: {e}"
        ) from e

    if not isinstance(data, dict):
        raise ValueError("Ответ должен быть JSON-объектом")

    return data


# ============================================================
# 7. НОРМАЛИЗАЦИЯ ХОББИ
# ============================================================

HOBBY_ALIASES = {
    "бассейн": "плавание",
    "плавание": "плавание",
    "spa": "spa",
    "спа": "spa",
    "йога": "йога",
    "прогулки": "прогулки"
}


def normalize_hobby(name: str) -> str:

    name = name.lower().strip()

    return HOBBY_ALIASES.get(name, name)


# ============================================================
# 8. АГЕНТ-АНАЛИТИК
# ============================================================

class AnalystAgent:

    def __init__(self, llm):

        self.llm = llm
        self.system = SystemMessage(content=SYSTEM_PROMPT)

    def analyze(self, user_text: str, verbose=True) -> UserProfile:

        if not user_text or not user_text.strip():
            raise ValueError("Пустой запрос пользователя")

        # Защита от простых попыток промпт-инъекций

        suspicious = [
            "ignore previous",
            "забудь инструк",
            "system:",
            "<|im_start|>"
        ]

        if any(p in user_text.lower() for p in suspicious):
            raise ValueError(
                "Обнаружена попытка инъекции"
            )

        # Вызов языковой модели

        response = self.llm.invoke([
            self.system,
            HumanMessage(content=user_text)
        ])

        raw_text = response.content

        if verbose:
            print("\nОтвет модели:")
            print(raw_text)

        # Извлечение JSON

        try:
            data = extract_json_from_response(raw_text)

        except ValueError as e:
            raise ValueError(
                f"Модель вернула некорректный JSON:\n{raw_text}"
            ) from e

        # Pydantic-валидация

        try:

            profile = UserProfile.model_validate(data)

            if verbose:
                print("\nПрофиль успешно прошёл валидацию!")

            return profile

        except ValidationError as e:

            if verbose:

                print("\nОШИБКА ВАЛИДАЦИИ PYDANTIC")

                print("\nJSON модели:")
                print(json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2
                ))

                print("\nКОНКРЕТНЫЕ ОШИБКИ:")

                for error in e.errors():

                    print(f"Поле: {error['loc']}")
                    print(f"Ошибка: {error['msg']}")
                    print(f"Тип: {error['type']}")
                    print("-" * 40)

            raise

    # Контракт с гидом

    @staticmethod
    def for_guide(p: UserProfile) -> list:

        return [
            {
                "hobby": h.name,
                "location": h.location_type,
                "weight": h.priority_weight
            }
            for h in p.hobbies
        ]

    # Контракт с логистом

    @staticmethod
    def for_logist(p: UserProfile) -> dict:

        return {
            "fitness": p.fitness.value,
            "has_children": p.has_children,
            "group_size": p.group_size,
            "daily_pace_limit_km": (
                5 if p.fitness == FitnessLevel.LOW else 15
            )
        }

    # Контракт с бюджетным агентом

    @staticmethod
    def for_budget(p: UserProfile) -> dict:

        return {
            "budget_priority": p.budget_priority,
            "comfort_priority": p.comfort_priority
        }

    # Контракт с критиком

    @staticmethod
    def for_critic(p: UserProfile) -> dict:

        return p.model_dump(mode="json")


# ============================================================
# 9. МЕТРИКИ КАЧЕСТВА
# ============================================================

def evaluate_analyst(agent, test_cases):

    total = len(test_cases)

    schema_ok = 0
    conflicts = 0

    correct = 0
    total_expected = 0

    for i, case in enumerate(test_cases, 1):

        # Ожидаемые хобби учитываются для каждого теста

        expected = {
            normalize_hobby(h)
            for h in case.get("expected_hobbies", [])
        }

        total_expected += len(expected)

        try:

            profile = agent.analyze(
                case["input"],
                verbose=False
            )

            schema_ok += 1

            got = {
                normalize_hobby(h.name)
                for h in profile.hobbies
            }

            matched = expected & got
            correct += len(matched)

            print(f"\nТест {i}: УСПЕШНО")
            print("Ожидалось:", expected)
            print("Распознано:", got)
            print("Совпадения:", matched)

        except Exception as e:

            conflicts += 1

            print(f"\nТест {i}: ОШИБКА")
            print(type(e).__name__, str(e))

    return {
        "schema_adherence": schema_ok / max(total, 1),
        "conflict_rate": conflicts / max(total, 1),
        "extraction_recall": correct / max(total_expected, 1)
    }


# ============================================================
# 10. ТЕСТОВЫЕ ДАННЫЕ
# ============================================================

TEST_CASES = [

    {
        "input": """
        После очень напряжённого рабочего года хочется
        устроить спокойный отпуск на 5–7 дней.

        Не интересуют экскурсии с утра до вечера,
        экстремальные развлечения и длительные походы.

        Хотелось бы посещать SPA, бассейн,
        заниматься йогой, гулять по красивым местам
        и хорошо высыпаться.

        Желательно выбрать гостиницу с хорошим сервисом,
        бассейном, рестораном и возможностью заказать массаж.

        Едем вдвоём. Предпочитаем комфортный отдых,
        без необходимости самостоятельно готовить
        и заниматься организацией каждого дня.

        Бюджет не самый ограниченный, важнее качество
        проживания и возможность действительно отдохнуть.
        """,

        "expected_hobbies": [
            "SPA",
            "йога",
            "плавание",
            "прогулки"
        ]
    },

]


# ============================================================
# 11. ДЕМОНСТРАЦИЯ
# ============================================================

agent = AnalystAgent(llm)

print("\n" + "=" * 60)
print("ДЕМО: АНАЛИЗ ОДНОГО ЗАПРОСА")
print("=" * 60)

try:

    profile = agent.analyze(
        TEST_CASES[0]["input"],
        verbose=True
    )

    print("\nИТОГОВЫЙ ПРОФИЛЬ:")

    print(
        profile.model_dump_json(
            indent=2,
            ensure_ascii=False
        )
    )

    print("\n" + "=" * 60)
    print("КОНТРАКТЫ С ДРУГИМИ АГЕНТАМИ")
    print("=" * 60)

    print("Гиду:    ", agent.for_guide(profile))
    print("Логисту: ", agent.for_logist(profile))
    print("Бюджету: ", agent.for_budget(profile))
    print("Критику: ", agent.for_critic(profile))

except Exception as e:

    print("\nОШИБКА:")
    print(type(e).__name__, str(e))


# ============================================================
# 12. МЕТРИКИ КАЧЕСТВА
# ============================================================

print("\n" + "=" * 60)
print("МЕТРИКИ КАЧЕСТВА")
print("=" * 60)

metrics = evaluate_analyst(agent, TEST_CASES)

print("\nИТОГОВЫЕ ПОКАЗАТЕЛИ:")

for name, value in metrics.items():
    print(f"{name}: {value:.2%}")
