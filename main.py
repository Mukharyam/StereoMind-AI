import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


class GeometryEntity(BaseModel):
    name: str = Field(description="Название фигуры или объекта")
    type: str = Field(description="Тип объекта")
    properties: List[str] = Field(description="Список свойств")


class GeometryResponse(BaseModel):
    reasoning: str = Field(description="Логика решения")
    entities: List[GeometryEntity] = Field(description="Геометрические сущности")
    result: str = Field(description="Ответ")


app = FastAPI(title="StereoMind API Prototype")


class SolveRequest(BaseModel):
    problem_text: str
    temperature: Optional[float] = 0.1


@app.post("/solve", response_model=GeometryResponse)
async def solve_geometry_problem(request: SolveRequest):
    try:
        llm = ChatOpenAI(
            model="gpt-oss-120b",
            api_key=os.getenv("CUSTOM_API_KEY"),
            base_url="https://litellm.happyhub.ovh/v1",
            temperature=request.temperature
        )

        structured_llm = llm.with_structured_output(GeometryResponse)

        system_prompt = (
            "<role>\n"
            "Ты — ведущий эксперт в области аналитической геометрии и математического моделирования. "
            "Твоя специализация — декомпозиция текстовых задач для их последующего решения методами линейной алгебры.\n"
            "</role>\n\n"

            "<goal>\n"
            "Твоя цель: преобразовать текст геометрической задачи в структурированную математическую модель, "
            "которая послужит основой для генерации программного кода.\n"
            "</goal>\n\n"

            "<context>\n"
            "Данная задача является частью системы StereoMind-AI. Нам важно не просто получить ответ, "
            "а выделить точные компоненты (точки, векторы, плоскости) и их свойства для построения 3D-визуализации.\n"
            "</context>\n\n"

            "<instructions>\n"
            "Выполни задачу пошагово:\n"
            "1. Тщательно проанализируй текст задачи, предоставленный в тегах <problem_text>.\n"
            "2. Идентифицируй все геометрические фигуры и объекты. Для каждого объекта укажи его тип и все известные численные характеристики.\n"
            "3. Сформулируй пошаговую логику решения (reasoning). Опиши, какие математические зависимости или теоремы нужно применить.\n"
            "4. Определи финальный искомый результат.\n"
            "5. Сформируй ответ строго в формате JSON, соответствующем заданной схеме.\n"
            "</instructions>\n\n"

            "<constraints>\n"
            "- Инструкции должны быть четкими, без лишних слов.\n"
            "- В поле 'properties' записывай только факты (например, 'сторона = 5'), без длинных предложений.\n"
            "- Если в задаче недостаточно данных для однозначного решения, укажи это в поле 'reasoning'.\n"
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
  "temperature": 0.1
}

"""