import csv
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mix" / "tabl"
FRONTEND_FILE = BASE_DIR.parent / "frontend" / "index.html"

def read_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ВНИМАНИЕ: Файл {path} не найден!")
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

@app.get("/")
def serve_frontend():
    """Отдает HTML-страницу из папки frontend"""
    if not FRONTEND_FILE.exists():
        return {"error": f"Файл index.html не найден по пути {FRONTEND_FILE}"}
    return FileResponse(FRONTEND_FILE)

@app.get("/api/results")
def get_results():
    """Объединяет данные результатов и найденные ошибки"""
    results = read_csv("results.csv")
    issues = read_csv("issues.csv")
    
    # Создаем словарь ошибок для быстрого поиска
    issues_dict = {}
    for iss in issues:
        r_ids = iss.get("record_ids", "").split(";")
        for r_id in r_ids:
            if r_id:
                issues_dict[r_id.strip()] = {
                    "check_name": iss.get("check_name", ""),
                    "description": iss.get("description", "")
                }

    # Объединяем
    merged_data = []
    for row in results:
        rid = row.get("result_id", "")
        row["has_issue"] = rid in issues_dict
        row["issue_details"] = issues_dict.get(rid, {})
        merged_data.append(row)

    return merged_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)