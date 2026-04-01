import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


class GeometryEntity(BaseModel):
    name: str = Field(description="Название фигуры или объекта (например, 'пирамида SABCD')")
    type: str = Field(description="Тип объекта (например, 'многогранник', 'плоскость')")
    properties: List[str] = Field(description="Известные характеристики (например, ['правильная', 'высота = 6'])")


class GeometrySGRResponse(BaseModel):
    step_1_extracted_facts: List[GeometryEntity] = Field(
        description="Шаг 1: Сбор фактов. Извлеки все геометрические объекты и их свойства из условия."
    )
    step_2_goal_definition: str = Field(
        description="Шаг 2: Формулировка цели. Кратко опиши, что именно требуется найти или доказать."
    )
    step_3_theorems_and_formulas: List[str] = Field(
        description="Шаг 3: Математическая база. Перечисли теоремы и формулы, необходимые для решения этой задачи."
    )
    step_4_solution_plan: str = Field(
        description="Шаг 4: Пошаговый план решения. Опиши логику применения формул из Шага 3 к фактам из Шага 1."
    )
    step_5_final_result: str = Field(
        description="Шаг 5: Итоговый ответ. Краткий финальный численный или алгебраический результат."
    )


app = FastAPI(title="StereoMind API Prototype")


class SolveRequest(BaseModel):
    problem_text: str
    temperature: Optional[float] = 0.0


@app.post("/solve", response_model=GeometrySGRResponse)
async def solve_geometry_problem(request: SolveRequest):
    try:
        llm = ChatOpenAI(
            model="gpt-oss-120b",
            api_key=os.getenv("CUSTOM_API_KEY"),
            base_url="https://litellm.happyhub.ovh/v1",
            temperature=request.temperature
        )

        structured_llm = llm.with_structured_output(GeometrySGRResponse)

        system_prompt = (
            "<role>\n"
            "Ты — ведущий эксперт в области аналитической геометрии и математического моделирования. "
            "Твоя специализация — декомпозиция текстовых задач для их последующего решения методами линейной алгебры.\n"
            "</role>\n\n"

            "<goal>\n"
            "Твоя цель: преобразовать текст геометрической задачи в структурированную математическую модель, "
            "через строгую последовательность шагов рассуждения. \n"
            "</goal>\n\n"

            "<context>\n"
            "Данная задача является частью системы StereoMind-AI. Нам важно не просто получить ответ, "
            "а выделить точные компоненты (точки, векторы, плоскости) и их свойства для построения 3D-визуализации.\n"
            "</context>\n\n"

            "<instructions>\n"
            "Выполни задачу, строго заполняя поля JSON-схемы шаг за шагом:\n"
            "1. В поле 'step_1_extracted_facts' тщательно проанализируй текст из <problem_text> и идентифицируй все фигуры и их численные характеристики.\n"
            "2. В поле 'step_2_goal_definition' определи искомый результат.\n"
            "3. В поле 'step_3_theorems_and_formulas' выпиши математические зависимости или теоремы, которые нужно применить.\n"
            "4. В поле 'step_4_solution_plan' сформулируй пошаговую логику решения.\n"
            "5. В поле 'step_5_final_result' укажи только финальный ответ.\n"
            "</instructions>\n\n"

            "<constraints>\n"
            "- Инструкции должны быть четкими, без лишних слов.\n"
            "- В характеристиках объектов (properties) на шаге 1 записывай только факты (например, 'сторона = 5'), без длинных предложений.\n"
            "- Если в задаче недостаточно данных для однозначного решения, укажи это в поле 'step_4_solution_plan'.\n"
            "</constraints>\n\n"

            "Входные данные (текст задачи) будут переданы пользователем ниже."
        )

        response = structured_llm.invoke([
            ("system", system_prompt),
            ("human", f"<problem_text>\n{request.problem_text}\n</problem_text>")
        ])

        return response

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




""""

{
  "problem_text": "В правильной треугольной пирамиде сторона основания равна 4, а высота равна 6. Найдите объем пирамиды.",
  "temperature": 0.0
}


{
  "problem_text": "Дана правильная треугольная призма ABCA1B1C1, все рёбра основания которой равны 2*sqrt(7). Сечение, проходящее через боковое ребро AA1 и середину M ребра B1C1, является квадратом. Найдите расстояние между прямыми A1B и AM",
  "temperature": 0.0
}


"""