import json
import os
from typing import List, Literal, Optional
import fitz 
from openai import OpenAI
from pydantic import BaseModel, Field

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")


class PublicationMeta(BaseModel):
    publication_id: str = Field(description="Уникальный ID публикации, например PUB_001 или по фамилии/году")
    title: str = Field(default="")
    authors: str = Field(default="")
    year: Optional[int] = Field(default=None)
    venue: str = Field(default="")
    DOI: str = Field(default="")
    language: str = Field(default="en")


class ConfigurationMeta(BaseModel):
    config_id: str = Field(description="ID конфигурации RAG, например CFG_001")
    rag_variant: str = Field(default="")
    retrieval_method: str = Field(default="")
    chunking_method: str = Field(default="")
    chunk_size: Optional[int] = Field(default=None)
    chunk_overlap: Optional[int] = Field(default=None)
    chunk_unit: str = Field(default="")
    generator_model: str = Field(default="")
    embedding_model: str = Field(default="")
    retriever_model: str = Field(default="")
    reranker_model: str = Field(default="")


class ExperimentMeta(BaseModel):
    experiment_id: str = Field(description="ID эксперимента, например EXP_001")
    prompt_strategy: str = Field(default="")
    context_strategy: str = Field(default="")
    evaluation_protocol: str = Field(default="")
    baseline_name: str = Field(default="")


class DatasetMeta(BaseModel):
    dataset_id: str = Field(description="ID датасета, например DS_NQ")
    dataset_name: str = Field(default="")


class TaskMeta(BaseModel):
    task_id: str = Field(description="ID задачи, например TASK_QA")
    task_type: str = Field(default="")


class Provenance(BaseModel):
    publication_id: str = Field(description="Идентификатор публикации")
    result_origin: Literal["primary", "cited"] = Field(description="primary или cited")
    cited_publication_id: Optional[str] = Field(default=None)
    source_page: Optional[int] = Field(default=None)
    source_section: str = Field(default="")
    source_table: Optional[str] = Field(default=None)
    source_figure: Optional[str] = Field(default=None)
    source_fragment: str = Field(description="Точная цитата из текста, подтверждающая значение")


class Status(BaseModel):
    raw_value: str = Field(default="")
    raw_unit: str = Field(default="")
    normalized_value: Optional[float] = Field(default=None)
    normalized_unit: str = Field(default="")
    extraction_status: Literal["extracted", "present_not_extractable", "absent"] = Field(default="extracted")


class MetricResult(BaseModel):
    result_id: str = Field(description="Уникальный ID результата внутри публикации, например RES_001")
    experiment_id: str = Field(description="ID связанного эксперимента")
    config_id: str = Field(description="ID связанной конфигурации")
    dataset_id: str = Field(description="ID связанного датасета")
    task_id: str = Field(description="ID связанной задачи")
    metric_group: str = Field(default="")
    metric_name: str = Field(description="Название метрики, например Recall@5, Exact Match, Faithfulness")
    metric_variant: Optional[str] = Field(default=None)
    metric_direction: Literal["higher_is_better", "lower_is_better", "target"] = Field(default="higher_is_better")
    metric_k: Optional[int] = Field(default=None)
    metric_value: Optional[float] = Field(default=None)
    metric_unit: str = Field(default="")
    baseline_value: Optional[float] = Field(default=None)
    aggregation: Optional[str] = Field(default=None)
    sample_size: Optional[int] = Field(default=None)
    provenance: Provenance
    status: Status
    duplicate_of: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)


class PaperExtraction(BaseModel):
    publication: PublicationMeta
    configurations: List[ConfigurationMeta] = Field(default_factory=list)
    experiments: List[ExperimentMeta] = Field(default_factory=list)
    datasets: List[DatasetMeta] = Field(default_factory=list)
    tasks: List[TaskMeta] = Field(default_factory=list)
    metric_results: List[MetricResult] = Field(default_factory=list)


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages_text = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pages_text.append(f"--- PAGE {page_num + 1} ---\n{page.get_text()}")
    finally:
        doc.close()
    return "\n".join(pages_text)


def extract_metrics_with_llm(text: str) -> PaperExtraction:
    api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("Не задан OPENROUTER_API_KEY в переменных окружения.")

    client = OpenAI(
        base_url="https://openrouter.ai",
        api_key=api_key,
        timeout = 90.0
    )

    schema_json = json.dumps(PaperExtraction.model_json_schema(), ensure_ascii=False)
    prompt = f"""Ты — эксперт по анализу научных статей в области Retrieval-Augmented Generation (RAG).
Твоя задача — извлечь метаданные публикации, конфигурации RAG, эксперименты, датасеты, задачи и все экспериментальные метрики из текста статьи.

СТРОГИЕ ПРАВИЛА:
1. Верни ответ строго в формате валидного JSON, соответствующего JSON Schema:
{schema_json}
2. Не выдумывай числа. Если числовое значение отсутствует в тексте, ставь null и extraction_status="absent" или "present_not_extractable".
3. В поле source_page указывай номер страницы из маркеров "--- PAGE N ---".
4. В поле source_fragment приводи точную цитату из текста (или строки таблицы).
5. Связывай каждый объект в metric_results с соответствующими experiment_id, config_id, dataset_id, task_id и publication_id из списков выше.
6. Для метрик вида Recall@5, NDCG@10, Precision@3 обязательно заполняй metric_k (например, 5, 10, 3).
7. Для долей от 0 до 1 используй metric_unit="ratio", для процентов от 0 до 100 — metric_unit="percent".

Текст статьи:
{text[:100000]}
"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-super-vl:free",
        messages=[
            {"role": "system", "content": "Ты — точный научный ассистент. Отвечай исключительно валидным JSON без markdown-оберток."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw_content = response.choices[0].message.content or "{}"
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    data_dict = json.loads(cleaned)
    return PaperExtraction(**data_dict)
