import csv
import math
from pathlib import Path

# справочники
RESULT_ORIGIN = {"primary", "cited"}
EXTRACTION_STATUS = {"extracted", "present_not_extractable", "absent"}
DIRECTIONS = {"higher_is_better", "lower_is_better", "target"}

# чтение файла
def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def to_float(v):
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None

# валидация
def check_results(pub_dir):
    pub_dir = Path(pub_dir)
    results = read_csv(pub_dir / "results.csv")
    
    # Множества для быстрой проверки связей
    exps  = {r["experiment_id"] for r in read_csv(pub_dir / "experiments.csv")}
    cfgs  = {r["config_id"] for r in read_csv(pub_dir / "configurations.csv")}
    dsets = {r["dataset_id"] for r in read_csv(pub_dir / "datasets.csv")}
    tasks = {r["task_id"] for r in read_csv(pub_dir / "tasks.csv")}
    
    # Множество всех result_id для проверки duplicate_of
    all_rids = {r["result_id"] for r in results}

    issues = []

    def add(check, ids, desc, page="", frag=""):
        issues.append({
            "issue_id": f"ISS{len(issues)+1:04d}",
            "record_ids": ";".join(ids),
            "check_name": check,
            "description": desc,
            "source_page": page,
            "source_fragment": frag,
        })

    seen = {}

    # Единый цикл по всем результатам
    for r in results:
        rid = r["result_id"]
        page = r.get("source_page") or ""
        frag = r.get("source_fragment") or ""
        
        # Подготовка значений
        val = to_float(r.get("metric_value"))
        nval = to_float(r.get("normalized_value"))
        base = to_float(r.get("baseline_value"))
        k = to_float(r.get("metric_k"))
        
        unit = (r.get("metric_unit") or "").lower()
        nunit = (r.get("normalized_unit") or "").lower()
        name = (r.get("metric_name") or "").lower()
        direc = r.get("metric_direction") or ""

        # 1. Связи
        ex_id = r.get("experiment_id") or ""
        cf_id = r.get("config_id") or ""
        ds_id = r.get("dataset_id") or ""
        tk_id = r.get("task_id") or ""
        
        if ex_id not in exps or cf_id not in cfgs or ds_id not in dsets or tk_id not in tasks:
            add("protocol_mismatch", [rid],
                "Нет связанных записей в experiments/configurations/datasets/tasks",
                page, frag)

        # 2. Словари
        if r.get("result_origin") and r["result_origin"] not in RESULT_ORIGIN:
            add("normalization", [rid], f"result_origin='{r['result_origin']}'", page)
            
        if r.get("extraction_status") and r["extraction_status"] not in EXTRACTION_STATUS:
            add("normalization", [rid], f"extraction_status='{r['extraction_status']}'", page)
            
        if direc and direc not in DIRECTIONS:
            add("normalization", [rid],
                f"metric_direction='{direc}' не из рекомендуемых", page)

        # 3. Шкала и NaN/Inf
        if val is not None:
            if math.isnan(val) or math.isinf(val):
                add("normalization", [rid], "NaN/Inf в metric_value", page)
            else:
                if unit == "ratio" and (val < 0 or val > 1):
                    add("metric_scale", [rid],
                        f"{name}={val} вне допустимого для unit='{unit}'", page, frag)
                elif unit == "percent" and (val < 0 or val > 100):
                    add("metric_scale", [rid],
                        f"{name}={val} вне допустимого для unit='{unit}'", page, frag)

        # 4. metric_k
        needs_k = ("@" in name) or (name in {"ndcg", "recall", "precision", "mrr"})
        if needs_k:
            if k is None or math.isnan(k) or k < 0:
                add("metric_k", [rid], f"{name}: metric_k должно быть > 0", page)

        # 5. Нормализация
        if val is not None and nval is not None and not math.isnan(val) and not math.isnan(nval):
            expected = None
            if unit == "percent" and nunit == "ratio":
                expected = nval * 100
            elif unit == "ratio" and nunit == "percent":
                expected = nval / 100
                
            if expected is not None and abs(val - expected) > 1e-6:
                add("normalization", [rid],
                    f"metric_value={val} не согласован с normalized={nval}", page)

        # 6. baseline + direction
        if val is not None and base is not None and not math.isnan(val) and not math.isnan(base):
            if direc == "higher_is_better" and val < base:
                add("protocol_mismatch", [rid],
                    f"{name}: value={val} vs baseline={base} при {direc}", page)
            elif direc == "lower_is_better" and val > base:
                add("protocol_mismatch", [rid],
                    f"{name}: value={val} vs baseline={base} при {direc}", page)

        # 7. Дубли / противоречия
        dup = (r.get("duplicate_of") or "").strip()
        if dup:
            if dup == rid:
                add("contradiction", [rid], "duplicate_of ссылается на себя", page)
            elif dup not in all_rids:
                add("contradiction", [rid], f"duplicate_of={dup} не найден", page)
                
        key = (r.get("experiment_id"), r.get("config_id"), r.get("dataset_id"),
               r.get("task_id"), r.get("metric_name"), r.get("metric_variant"),
               r.get("metric_k"))
               
        if key in seen:
            a = seen[key]
            va = to_float(a.get("metric_value"))
            if va is not None and val is not None and abs(va - val) > 1e-9:
                add("contradiction", [a["result_id"], rid],
                    f"Одна метрика/эксперимент, но значения {va} и {val}", page)
        else:
            seen[key] = r

    # запись issues.csv
    out = pub_dir / "issues.csv"
    fields = ["issue_id", "record_ids", "check_name", "description",
              "source_page", "source_fragment"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(issues)

    return issues


if __name__ == "__main__":
    import sys
    pub = sys.argv[1] if len(sys.argv) > 1 else "submission/D001"
    issues = check_results(pub)
    print(f"Публикация {pub}: замечаний — {len(issues)}")
    for i in issues:
        print(f"  {i['issue_id']} [{i['check_name']}] {i['record_ids']}: {i['description']}")