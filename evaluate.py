import json
import requests
import pandas as pd
from tqdm import tqdm
import time

# Конфигурация
API_URL = "http://127.0.0.1:8000/solve"
DATASET_PATH = "dataset.json"
OUTPUT_PATH = "evaluation_results.csv"


def run_evaluation():
    # 1. Загрузка проверочного набора данных
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    results = []

    print(f"Начинаю обработку {len(dataset)} примеров...")

    # 2. Автоматическое получение результатов от API
    for item in tqdm(dataset):
        payload = {
            "problem_text": item["problem"],
            "temperature": 0.1  # Ставим низкую для стабильности теста
        }

        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload, timeout=60)
            latency = round(time.time() - start_time, 2)

            if response.status_code == 200:
                data = response.json()
                # Берем финальный результат и логику
                result_text = data.get("result", "Н/Д")
                reasoning = data.get("reasoning", "Н/Д")
                status = "Success"
            else:
                result_text = f"Error: {response.status_code}"
                reasoning = response.text
                status = "API_Error"

        except Exception as e:
            result_text = "Connection Error"
            reasoning = str(e)
            status = "System_Error"
            latency = 0

        # Формируем данные для строки таблицы
        results.append({
            "ID": item["id"],
            "запрос": item["problem"],
            "результат": result_text,
            "логика_решения": reasoning,
            "время_сек": latency,
            "статус": status
        })

    # 3. Формирование сводной таблицы
    df = pd.DataFrame(results)

    # Сохраняем в CSV для анализа в Excel/Google Sheets
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f"\nОценка завершена! Результаты сохранены в {OUTPUT_PATH}")
    print(df[["ID", "запрос", "результат", "статус"]])


if __name__ == "__main__":
    run_evaluation()
