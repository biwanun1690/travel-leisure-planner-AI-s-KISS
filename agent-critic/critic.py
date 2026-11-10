# -*- coding: utf-8 -*-
"""
Планировщик путешествий v2.1 — Исправление TypeError и улучшение архитектуры.
Исправлено: ошибка конкатенации SystemMessage, убраны глобальные переменные.
"""

import os
import json
from getpass import getpass
from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from typing import TypedDict

# ============================================================
# 1. ПОДКЛЮЧЕНИЕ К CLOUD.RU
# ============================================================

def get_llm(temperature: float = 0.1) -> ChatOpenAI:
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
        max_tokens=4000,
        timeout=90,
        max_retries=2,
    )

class CriticOutput(BaseModel):
    approved: bool
    score: int = Field(ge=0, le=100)
    issues: list[str] = Field(description="Конкретные замечания, если есть. Если нет - пустой список.")
    final_message_to_user: str = Field(description="Финальный красивый ответ клиенту. Заполнять только если approved=True")


CRITIC_PROMPT = """Ты Критик. Проверь план Логиста.
Чеклист:
1. Все ли места из выбранного варианта Бюджетника включены в daily_plans?
2. Не более 3 активностей в день?
3. Учтены ли constraints (например, если low fitness, нет ли тяжелых активностей)?
Если все ОК: approved=true, score=90-100, issues=[], final_message_to_user="Красивое приветствие и краткое описание плана".
Если есть проблемы: approved=false, score=<80, issues=["Конкретная проблема 1", "Конкретная проблема 2"], final_message_to_user="".
Верни строго JSON по схеме CriticOutput."""

# ============================================================
# 4. КЛАССЫ АГЕНТОВ (ИСПРАВЛЕННЫЙ)
# ============================================================

class Agent:
    def __init__(self, name: str, prompt: str, schema: type[BaseModel], temperature: float = 0.1):
        self.name = name
        self.base_prompt = prompt
        self.schema = schema
        self.temperature = temperature

    def run(self, input_data: dict, extra_prompt: str = "") -> BaseModel:
        print(f"[{self.name}] Думаю...", end=" ", flush=True)

        current_prompt = self.base_prompt
        if extra_prompt:
            current_prompt += f"\n\nВАЖНО: {extra_prompt}"

        llm = get_llm(temperature=self.temperature).with_structured_output(self.schema)

        # ИСПРАВЛЕНИЕ: используем ChatPromptTemplate вместо прямой конкатенации SystemMessage
        chain = ChatPromptTemplate.from_messages([
            ("system", current_prompt),
            ("human", "{input_data}")
        ]) | llm

        result = chain.invoke({"input_data": json.dumps(input_data, ensure_ascii=False, indent=2)})
        print("✓")
        return result

# Инициализация агентов
critic = Agent("Критик", CRITIC_PROMPT, CriticOutput, temperature=0.0)

def node_critic(state: PlannerState):
    critic_out = critic.run({
        "profile": state["profile"],
        "budget_metrics": state["budget_metrics"],
        "logistics": state["logistics"],
        "critic_feedback": state.get("critic_feedback", "")
    })

    feedback = "\n".join(critic_out.issues) if not critic_out.approved else ""

    return {
        "critic": critic_out.model_dump(),
        "critic_feedback": feedback,
        "iteration": state["iteration"] + 1
    }

# основной код