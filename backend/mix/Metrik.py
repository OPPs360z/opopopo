import csv
import math
import numpy as np
from pathlib import Path

#справочники
RESULT_ORIGIN   = {"primary", "cited"}
EXTRACTION_STATUS = {"extracted", "present_not_extractable", "absent"}
DIRECTIONS = {"higher_is_better", "lower_is_better", "target"}

#чтение файла
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

#валидация
def check_results(pub_dir):
    pub_dir = Path(pub_dir)
    results = read_csv(pub_dir / "results.csv")
    exps  = {r["experiment_id"] for r in read_csv(pub_dir / "experiments.csv")}
    cfgs  = {r["config_id"]     for r in read_csv(pub_dir / "configurations.csv")}
    dsets = {r["dataset_id"]    for r in read_csv(pub_dir / "datasets.csv")}
    tasks = {r["task_id"]       for r in read_csv(pub_dir / "tasks.csv")}

    n = len(results)
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

    #numpy-векторы
    rid   = np.array([r["result_id"] for r in results], dtype=object)
    val   = np.array([to_float(r.get("metric_value")) for r in results], dtype=float)
    nval  = np.array([to_float(r.get("normalized_value")) for r in results], dtype=float)
    base  = np.array([to_float(r.get("baseline_value")) for r in results], dtype=float)
    k     = np.array([to_float(r.get("metric_k")) for r in results], dtype=float)
    unit  = np.array([(r.get("metric_unit") or "").lower() for r in results], dtype=object)
    nunit = np.array([(r.get("normalized_unit") or "").lower() for r in results], dtype=object)
    name  = np.array([(r.get("metric_name") or "").lower() for r in results], dtype=object)
    direc = np.array([(r.get("metric_direction") or "") for r in results], dtype=object)
    ex_id = np.array([r.get("experiment_id") or "" for r in results], dtype=object)
    cf_id = np.array([r.get("config_id") or "" for r in results], dtype=object)
    ds_id = np.array([r.get("dataset_id") or "" for r in results], dtype=object)
    tk_id = np.array([r.get("task_id") or "" for r in results], dtype=object)
    page  = np.array([r.get("source_page") or "" for r in results], dtype=object)
    frag  = np.array([r.get("source_fragment") or "" for r in results], dtype=object)

    #связи
    miss = (
        ~np.isin(ex_id, list(exps)) |
        ~np.isin(cf_id, list(cfgs)) |
        ~np.isin(ds_id, list(dsets)) |
        ~np.isin(tk_id, list(tasks))
    )

    for i in np.where(miss)[0]:
        add("protocol_mismatch", [rid[i]],
            "Нет связанных записей в experiments/configurations/datasets/tasks",
            page[i], frag[i])

    #словари
    for i, r in enumerate(results):
        if r.get("result_origin") and r["result_origin"] not in RESULT_ORIGIN:
            add("normalization", [rid[i]], f"result_origin='{r['result_origin']}'", page[i])
        if r.get("extraction_status") and r["extraction_status"] not in EXTRACTION_STATUS:
            add("normalization", [rid[i]], f"extraction_status='{r['extraction_status']}'", page[i])
    bad_dir = (direc != "") & ~np.isin(direc, list(DIRECTIONS))
    for i in np.where(bad_dir)[0]:
        add("normalization", [rid[i]],
            f"metric_direction='{direc[i]}' не из рекомендуемых", page[i])

    #шкала (numpy)
    finite = np.isfinite(val)
    ratio_bad   = (unit == "ratio")   & finite & ((val < 0) | (val > 1))
    percent_bad = (unit == "percent") & finite & ((val < 0) | (val > 100))
    for i in np.where(ratio_bad | percent_bad)[0]:
        add("metric_scale", [rid[i]],
            f"{name[i]}={val[i]} вне допустимого для unit='{unit[i]}'", page[i], frag[i])

    nan_bad = np.isnan(val) | np.isinf(val)
    for i in np.where(nan_bad)[0]:
        add("normalization", [rid[i]], "NaN/Inf в metric_value", page[i])

    #metric_k (numpy)
    needs_k = np.array([("@" in nm) or (nm in {"ndcg","recall","precision","mrr"})
                        for nm in name])
    k_bad = needs_k & (np.isnan(k) | (k < 0))
    for i in np.where(k_bad)[0]:
        add("metric_k", [rid[i]],
            f"{name[i]}: metric_k должно быть > 0", page[i])

    #нормализация (numpy)
    expected = np.where(
        (unit == "percent") & (nunit == "ratio"),  nval * 100,
        np.where((unit == "ratio") & (nunit == "percent"), nval / 100, np.nan)
    )
    norm_bad = ~np.isnan(expected) & np.isfinite(val) & (np.abs(val - expected) > 1e-6)
    for i in np.where(norm_bad)[0]:
        add("normalization", [rid[i]],
            f"metric_value={val[i]} не согласован с normalized={nval[i]}", page[i])

    #baseline + direction (numpy)
    hv = (direc == "higher_is_better") & np.isfinite(val) & np.isfinite(base) & (val < base)
    lv = (direc == "lower_is_better")  & np.isfinite(val) & np.isfinite(base) & (val > base)
    for i in np.where(hv | lv)[0]:
        add("protocol_mismatch", [rid[i]],
            f"{name[i]}: value={val[i]} vs baseline={base[i]} при {direc[i]}", page[i])

    #дубли/противоречия
    seen = {}
    for r in results:
        dup = (r.get("duplicate_of") or "").strip()
        if dup:
            if dup == r["result_id"]:
                add("contradiction", [r["result_id"]], "duplicate_of ссылается на себя", r.get("source_page",""))
            elif dup not in set(rid):
                add("contradiction", [r["result_id"]],
                    f"duplicate_of={dup} не найден", r.get("source_page",""))
        key = (r.get("experiment_id"), r.get("config_id"), r.get("dataset_id"),
               r.get("task_id"), r.get("metric_name"), r.get("metric_variant"),
               r.get("metric_k"))
        if key in seen:
            a = seen[key]
            va, vb = to_float(a.get("metric_value")), to_float(r.get("metric_value"))
            if va is not None and vb is not None and abs(va - vb) > 1e-9:
                add("contradiction", [a["result_id"], r["result_id"]],
                    f"Одна метрика/эксперимент, но значения {va} и {vb}",
                    r.get("source_page",""))
        else:
            seen[key] = r

    #запись issues.csv
    out = pub_dir / "issues.csv"
    fields = ["issue_id","record_ids","check_name","description",
              "source_page","source_fragment"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(issues)

    return issues


if __name__ == "__main__":
    import sys
    pub = sys.argv[1] if len(sys.argv) > 1 else "submission/D001"
    issues = check_results(pub)
    print(f"Публикация {pub}: замечаний — {len(issues)}")
    for i in issues:
        print(f"  {i['issue_id']} [{i['check_name']}] {i['record_ids']}: {i['description']}")