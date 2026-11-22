import csv
import math
from pathlib import Path

RESULT_ORIGIN = {"primary", "cited"}
EXTRACTION_STATUS = {"extracted", "present_not_extractable", "absent"}
DIRECTIONS = {"higher_is_better", "lower_is_better", "target"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v):
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def check_results(pub_dir: str | Path) -> list[dict]:
    pub_dir = Path(pub_dir)
    results = read_csv(pub_dir / "results.csv")
    out = pub_dir / "issues.csv"
    fields = [
        "issue_id",
        "publication_id",
        "record_ids",
        "check_name",
        "description",
        "source_page",
        "source_fragment",
    ]

    if not results:
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
        return []

    exps = {
        (r.get("publication_id", "").strip(), r.get("experiment_id", "").strip())
        for r in read_csv(pub_dir / "experiments.csv")
        if r.get("experiment_id")
    }
    cfgs = {
        (r.get("publication_id", "").strip(), r.get("config_id", "").strip())
        for r in read_csv(pub_dir / "configurations.csv")
        if r.get("config_id")
    }
    dsets = {
        (r.get("publication_id", "").strip(), r.get("dataset_id", "").strip())
        for r in read_csv(pub_dir / "datasets.csv")
        if r.get("dataset_id")
    }
    tasks = {
        (r.get("publication_id", "").strip(), r.get("task_id", "").strip())
        for r in read_csv(pub_dir / "tasks.csv")
        if r.get("task_id")
    }
    all_rids = {
        (r.get("publication_id", "").strip(), r.get("result_id", "").strip())
        for r in results
        if r.get("result_id")
    }

    issues = []

    def add(check: str, pid: str, ids: list[str], desc: str, page: str = "", frag: str = ""):
        issues.append({
            "issue_id": f"ISS{len(issues) + 1:04d}",
            "publication_id": pid,
            "record_ids": ";".join(ids),
            "check_name": check,
            "description": desc,
            "source_page": page,
            "source_fragment": frag,
        })

    seen = {}

    for r in results:
        pid = (r.get("publication_id") or "").strip()
        rid = (r.get("result_id") or "").strip()
        page = r.get("source_page") or ""
        frag = r.get("source_fragment") or ""

        val = to_float(r.get("metric_value"))
        nval = to_float(r.get("normalized_value"))
        base = to_float(r.get("baseline_value"))
        k = to_float(r.get("metric_k"))

        unit = (r.get("metric_unit") or "").lower().strip()
        nunit = (r.get("normalized_unit") or "").lower().strip()
        name = (r.get("metric_name") or "").lower().strip()
        direc = (r.get("metric_direction") or "").strip()

        # 1. Связи с учетом publication_id
        ex_id = (r.get("experiment_id") or "").strip()
        cf_id = (r.get("config_id") or "").strip()
        ds_id = (r.get("dataset_id") or "").strip()
        tk_id = (r.get("task_id") or "").strip()

        if (
            (pid, ex_id) not in exps
            or (pid, cf_id) not in cfgs
            or (pid, ds_id) not in dsets
            or (pid, tk_id) not in tasks
        ):
            add(
                "protocol_mismatch",
                pid,
                [rid],
                "Нет связанных записей в experiments/configurations/datasets/tasks",
                page,
                frag,
            )

        # 2. Словари
        origin = (r.get("result_origin") or "").strip()
        if origin and origin not in RESULT_ORIGIN:
            add("normalization", pid, [rid], f"result_origin='{origin}'", page, frag)

        ext_status = (r.get("extraction_status") or "").strip()
        if ext_status and ext_status not in EXTRACTION_STATUS:
            add("normalization", pid, [rid], f"extraction_status='{ext_status}'", page, frag)

        if direc and direc not in DIRECTIONS:
            add("normalization", pid, [rid], f"metric_direction='{direc}' не из рекомендуемых", page, frag)

        # 3. Шкала и NaN/Inf
        if val is not None:
            if math.isnan(val) or math.isinf(val):
                add("normalization", pid, [rid], "NaN/Inf в metric_value", page, frag)
            else:
                if unit == "ratio" and (val < 0 or val > 1):
                    add(
                        "metric_scale",
                        pid,
                        [rid],
                        f"{name}={val} вне допустимого диапазона [0, 1] для unit='{unit}'",
                        page,
                        frag,
                    )
                elif unit == "percent" and (val < 0 or val > 100):
                    add(
                        "metric_scale",
                        pid,
                        [rid],
                        f"{name}={val} вне допустимого диапазона [0, 100] для unit='{unit}'",
                        page,
                        frag,
                    )

        # 4. metric_k
        needs_k = ("@" in name) or (name in {"ndcg", "recall", "precision", "mrr"})
        if needs_k:
            if k is None or math.isnan(k) or k <= 0:
                add("metric_k", pid, [rid], f"{name}: metric_k должно быть > 0", page, frag)

        # 5. Нормализация
        if val is not None and nval is not None and not math.isnan(val) and not math.isnan(nval):
            expected = None
            if unit == "percent" and nunit == "ratio":
                expected = nval * 100
            elif unit == "ratio" and nunit == "percent":
                expected = nval / 100

            if expected is not None and abs(val - expected) > 1e-6:
                add(
                    "normalization",
                    pid,
                    [rid],
                    f"metric_value={val} не согласован с normalized={nval}",
                    page,
                    frag,
                )

        # 6. baseline + direction
        if val is not None and base is not None and not math.isnan(val) and not math.isnan(base):
            if direc == "higher_is_better" and val < base:
                add(
                    "protocol_mismatch",
                    pid,
                    [rid],
                    f"{name}: value={val} хуже baseline={base} при {direc}",
                    page,
                    frag,
                )
            elif direc == "lower_is_better" and val > base:
                add(
                    "protocol_mismatch",
                    pid,
                    [rid],
                    f"{name}: value={val} хуже baseline={base} при {direc}",
                    page,
                    frag,
                )

        # 7. Дубли / противоречия
        dup = (r.get("duplicate_of") or "").strip()
        if dup:
            if dup == rid:
                add("contradiction", pid, [rid], "duplicate_of ссылается на себя", page, frag)
            elif (pid, dup) not in all_rids:
                add("contradiction", pid, [rid], f"duplicate_of={dup} не найден", page, frag)

        key = (
            pid,
            ex_id,
            cf_id,
            ds_id,
            tk_id,
            r.get("metric_name"),
            r.get("metric_variant"),
            r.get("metric_k"),
        )

        if key in seen:
            a = seen[key]
            va = to_float(a.get("metric_value"))
            if va is not None and val is not None and abs(va - val) > 1e-9:
                add(
                    "contradiction",
                    pid,
                    [a.get("result_id", ""), rid],
                    f"Одна метрика/эксперимент, но разные значения {va} и {val}",
                    page,
                    frag,
                )
        else:
            seen[key] = r

    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(issues)

    return issues
