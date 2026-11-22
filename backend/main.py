import csv
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from pars import PaperExtraction, extract_metrics_with_llm, extract_text_from_pdf
from validator import check_results

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DATA_DIR = BASE_DIR / "mix" / "tabl"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_SCHEMAS: Dict[str, List[str]] = {
    "publications.csv": [
        "publication_id",
        "title",
        "authors",
        "year",
        "venue",
        "DOI",
        "language",
    ],
    "configurations.csv": [
        "publication_id",
        "config_id",
        "rag_variant",
        "retrieval_method",
        "chunking_method",
        "chunk_size",
        "chunk_overlap",
        "chunk_unit",
        "generator_model",
        "embedding_model",
        "retriever_model",
        "reranker_model",
    ],
    "experiments.csv": [
        "publication_id",
        "experiment_id",
        "prompt_strategy",
        "context_strategy",
        "evaluation_protocol",
        "baseline_name",
    ],
    "datasets.csv": [
        "publication_id",
        "dataset_id",
        "dataset_name",
    ],
    "tasks.csv": [
        "publication_id",
        "task_id",
        "task_type",
    ],
    "results.csv": [
        "publication_id",
        "result_id",
        "experiment_id",
        "config_id",
        "dataset_id",
        "task_id",
        "metric_group",
        "metric_name",
        "metric_variant",
        "metric_direction",
        "metric_k",
        "metric_value",
        "metric_unit",
        "baseline_value",
        "aggregation",
        "sample_size",
        "raw_value",
        "raw_unit",
        "normalized_value",
        "normalized_unit",
        "result_origin",
        "cited_publication_id",
        "source_page",
        "source_section",
        "source_table",
        "source_figure",
        "source_fragment",
        "extraction_status",
        "duplicate_of",
        "note",
    ],
    "issues.csv": [
        "issue_id",
        "publication_id",
        "record_ids",
        "check_name",
        "description",
        "source_page",
        "source_fragment",
    ],
}

app = FastAPI(title="anchor. RAG Research Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_csv_files() -> None:
    for filename, headers in CSV_SCHEMAS.items():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()


def read_csv_dicts(filename: str) -> List[Dict[str, Any]]:
    ensure_csv_files()
    file_path = DATA_DIR / filename
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(filename: str, rows: List[Dict[str, Any]]) -> None:
    headers = CSV_SCHEMAS[filename]
    file_path = DATA_DIR / filename
    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {h: ("" if row.get(h) is None else row.get(h)) for h in headers}
            writer.writerow(cleaned)


def upsert_publication_data(extraction: PaperExtraction) -> None:
    ensure_csv_files()
    pub_id = extraction.publication.publication_id.strip() or "PUB_UNKNOWN"

    # 1. publications.csv
    pubs = [r for r in read_csv_dicts("publications.csv") if r.get("publication_id", "").strip() != pub_id]
    pub_row = extraction.publication.model_dump()
    pub_row["publication_id"] = pub_id
    pubs.append(pub_row)
    write_csv_dicts("publications.csv", pubs)

    # 2. configurations.csv
    cfgs = [r for r in read_csv_dicts("configurations.csv") if r.get("publication_id", "").strip() != pub_id]
    for cfg in extraction.configurations:
        row = cfg.model_dump()
        row["publication_id"] = pub_id
        cfgs.append(row)
    write_csv_dicts("configurations.csv", cfgs)

    # 3. experiments.csv
    exps = [r for r in read_csv_dicts("experiments.csv") if r.get("publication_id", "").strip() != pub_id]
    for exp in extraction.experiments:
        row = exp.model_dump()
        row["publication_id"] = pub_id
        exps.append(row)
    write_csv_dicts("experiments.csv", exps)

    # 4. datasets.csv
    dsets = [r for r in read_csv_dicts("datasets.csv") if r.get("publication_id", "").strip() != pub_id]
    for ds in extraction.datasets:
        row = ds.model_dump()
        row["publication_id"] = pub_id
        dsets.append(row)
    write_csv_dicts("datasets.csv", dsets)

    # 5. tasks.csv
    tasks = [r for r in read_csv_dicts("tasks.csv") if r.get("publication_id", "").strip() != pub_id]
    for tk in extraction.tasks:
        row = tk.model_dump()
        row["publication_id"] = pub_id
        tasks.append(row)
    write_csv_dicts("tasks.csv", tasks)

    # 6. results.csv (разворачиваем provenance и status в плоскую таблицу)
    results = [r for r in read_csv_dicts("results.csv") if r.get("publication_id", "").strip() != pub_id]
    for res in extraction.metric_results:
        prov = res.provenance
        st = res.status
        flat_row = {
            "publication_id": prov.publication_id.strip() or pub_id,
            "result_id": res.result_id,
            "experiment_id": res.experiment_id,
            "config_id": res.config_id,
            "dataset_id": res.dataset_id,
            "task_id": res.task_id,
            "metric_group": res.metric_group,
            "metric_name": res.metric_name,
            "metric_variant": res.metric_variant,
            "metric_direction": res.metric_direction,
            "metric_k": res.metric_k,
            "metric_value": res.metric_value,
            "metric_unit": res.metric_unit,
            "baseline_value": res.baseline_value,
            "aggregation": res.aggregation,
            "sample_size": res.sample_size,
            "raw_value": st.raw_value,
            "raw_unit": st.raw_unit,
            "normalized_value": st.normalized_value,
            "normalized_unit": st.normalized_unit,
            "result_origin": prov.result_origin,
            "cited_publication_id": prov.cited_publication_id,
            "source_page": prov.source_page,
            "source_section": prov.source_section,
            "source_table": prov.source_table,
            "source_figure": prov.source_figure,
            "source_fragment": prov.source_fragment,
            "extraction_status": st.extraction_status,
            "duplicate_of": res.duplicate_of,
            "note": res.note,
        }
        results.append(flat_row)
    write_csv_dicts("results.csv", results)

    # 7. Перезапускаем валидатор и обновляем issues.csv
    check_results(DATA_DIR)


def get_full_state() -> Dict[str, Any]:
    ensure_csv_files()
    check_results(DATA_DIR)
    return {
        "results": read_csv_dicts("results.csv"),
        "publications": read_csv_dicts("publications.csv"),
        "configurations": read_csv_dicts("configurations.csv"),
        "experiments": read_csv_dicts("experiments.csv"),
        "datasets": read_csv_dicts("datasets.csv"),
        "tasks": read_csv_dicts("tasks.csv"),
        "issues": read_csv_dicts("issues.csv"),
    }


@app.on_event("startup")
def on_startup():
    ensure_csv_files()


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Файл frontend/index.html не найден")
    return FileResponse(index_path)


@app.get("/api/bootstrap")
def api_bootstrap():
    return get_full_state()


@app.get("/api/results")
def api_results():
    return read_csv_dicts("results.csv")


@app.get("/api/publications")
def api_publications():
    return read_csv_dicts("publications.csv")


@app.get("/api/issues")
def api_issues():
    return read_csv_dicts("issues.csv")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Поддерживаются только PDF-файлы.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        text = extract_text_from_pdf(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=422, detail="Не удалось извлечь текст из PDF (возможно, это скан без OCR).")

        extraction = extract_metrics_with_llm(text)
        upsert_publication_data(extraction)
        state = get_full_state()

        return {
            "status": "ok",
            "extracted_count": len(extraction.metric_results),
            "data": extraction.model_dump(),
            "state": state,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки PDF: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
