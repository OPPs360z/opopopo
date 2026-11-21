# backend/main.py
import csv
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Импортируем логику парсинга из pars.py
from pars import extract_text_from_pdf_bytes, extract_data_with_llm

app = FastAPI()

# Разрешаем CORS (чтобы фронтенд мог обращаться к бэкенду)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка путей через pathlib
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mix" / "tabl"
FRONTEND_FILE = BASE_DIR.parent / "frontend" / "index.html"


def read_csv(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# 1. Отдаем index.html при заходе на корень сайта
@app.get("/")
def serve_frontend():
    if not FRONTEND_FILE.exists():
        return {"error": f"Файл index.html не найден по пути {FRONTEND_FILE}"}
    return FileResponse(FRONTEND_FILE)


# 2. Эндпоинт для загрузки PDF и извлечения данных
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Разрешены только PDF файлы")

    try:
        # Читаем файл
        pdf_bytes = await file.read()

        # 1. Парсим текст
        text = extract_text_from_pdf_bytes(pdf_bytes)

        # 2. Отправляем в LLM
        result = extract_data_with_llm(text, file.filename)

        # 3. Возвращаем результат на фронтенд в виде JSON
        return JSONResponse(content={
            "status": "success",
            "data": result.model_dump()
        })

    except Exception as e:
        print(f"Ошибка при обработке PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 3. Основной API для таблицы на сайте
@app.get("/api/results")
def get_results():
    results = read_csv("results.csv")
    issues = read_csv("issues.csv")

    issues_dict = {}
    for iss in issues:
        r_ids = iss.get("record_ids", "").split(";")
        for r_id in r_ids:
            if r_id:
                issues_dict[r_id.strip()] = {
                    "check_name": iss.get("check_name", ""),
                    "description": iss.get("description", "")
                }

    merged_data = []
    for row in results:
        rid = row.get("result_id", "")
        row["has_issue"] = rid in issues_dict
        row["issue_details"] = issues_dict.get(rid, {})
        merged_data.append(row)

    return merged_data


# 4. Эндпоинты для JSON и схемы
@app.get("/api/export/json")
def export_full_json():
    """Возвращает ПОЛНЫЙ датасет (все 7 таблиц) в виде одного JSON-файла"""
    all_data = {
        "publications": read_csv("publications.csv"),
        "configurations": read_csv("configurations.csv"),
        "datasets": read_csv("datasets.csv"),
        "tasks": read_csv("tasks.csv"),
        "experiments": read_csv("experiments.csv"),
        "results": read_csv("results.csv"),
        "issues": read_csv("issues.csv")
    }
    return JSONResponse(content=all_data)


@app.get("/api/schema")
def get_schema():
    """Возвращает паспорт поля (schema.json), требуемый в README"""
    schema = {
        "results.csv": {
            "result_id": {"type": "string", "description": "Уникальный идентификатор результата"},
            "experiment_id": {"type": "string", "description": "Связь с таблицей experiments"},
            "config_id": {"type": "string", "description": "Связь с configurations"},
            "dataset_id": {"type": "string", "description": "Связь с datasets"},
            "task_id": {"type": "string", "description": "Связь с tasks"},
            "metric_name": {"type": "string", "description": "Название метрики"},
            "metric_value": {"type": "float", "description": "Нормализованное значение"},
            "metric_unit": {"type": "string", "enum": ["ratio", "percent"], "description": "Единица измерения"},
            "metric_direction": {"type": "string", "enum": ["higher_is_better", "lower_is_better", "target"]},
            "source_page": {"type": "integer", "description": "Номер страницы PDF"},
            "source_fragment": {"type": "string", "description": "Дословная цитата из PDF"}
        }
    }
    return JSONResponse(content=schema)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)