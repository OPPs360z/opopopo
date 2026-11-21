# backend/pars.py
import fitz  # PyMuPDF
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from tenacity import retry, stop_after_attempt, wait_exponential

OPENROUTER_API_KEY = "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"

# --- Модели Pydantic ---
class Provenance(BaseModel):
    publication_id: str
    result_origin: Literal["primary", "cited"]
    cited_publication_id: Optional[str] = None
    source_page: int
    source_section: Optional[str] = None
    source_table: Optional[str] = None
    source_figure: Optional[str] = None
    source_fragment: str

class RecordStatus(BaseModel):
    raw_value: str
    raw_unit: str
    normalized_value: Optional[float] = None
    normalized_unit: Optional[str] = None
    extraction_status: Literal["extracted", "present_not_extractable", "absent"] = "extracted"

class MetricResult(BaseModel):
    result_id: str
    experiment_id: str
    config_id: str
    dataset_id: str
    task_id: str
    metric_group: str
    metric_name: str
    metric_variant: Optional[str] = None
    metric_direction: Optional[Literal["higher_is_better", "lower_is_better"]] = None
    metric_k: Optional[int] = None
    metric_value: float
    metric_unit: Optional[str] = None
    baseline_value: Optional[float] = None
    aggregation: Optional[str] = None
    sample_size: Optional[int] = None
    note: Optional[str] = None
    duplicate_of: Optional[str] = None
    provenance: Provenance
    status: RecordStatus

class ExtractionOutput(BaseModel):
    metric_results: List[MetricResult]

# --- Парсинг текста из PDF ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += f"\n\n--- [СТРАНИЦА {page_num + 1}] ---\n\n"
        full_text += page.get_text("text")
    return full_text

# --- Вызов LLM ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=20))
def extract_data_with_llm(text: str, filename: str) -> ExtractionOutput:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    pub_id = filename.replace(".pdf", "")

    system_prompt = f"""
    Ты — эксперт по извлечению данных из научных статей.
    Твоя задача — извлечь результаты метрик из текста и вернуть их строго по JSON-схеме.

    ПРАВИЛА:
    1. publication_id: {pub_id}. result_origin: "primary".
    2. Уникальные ID: config_id (например, cfg_gpt4), task_id, dataset_id.
    3. МЕТРИКИ @K: "recall@5" -> metric_name = "Recall", metric_k = 5.
    4. metric_direction: "higher_is_better" для BLEU, ROUGE, Recall и т.д.
    5. raw_value (как в тексте, например ".204"), normalized_value (0.204).
    6. source_table: "Table 1", source_page: номер страницы, source_fragment: цитата.
    """
    
    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-super-vl:free", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вот текст статьи:\n\n{text}"}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractionOutput",
                "strict": True,
                "schema": ExtractionOutput.model_json_schema(),
            },
        },
        temperature=0.0
    )
    return ExtractionOutput.model_validate_json(response.choices[0].message.content)