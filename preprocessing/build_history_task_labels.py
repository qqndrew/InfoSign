"""Audit and build seven exact-source history labels using raw OMOP evidence."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from datetime import date, datetime
import math
from pathlib import Path
import statistics
import tempfile

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def atomic_text(path, content):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(content)
        name = f.name
    os.replace(name, path)

def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def verify_files(root, hashes):
    root = Path(root).resolve()
    for name, digest in hashes.items():
        p = (root / name).resolve(strict=True)
        if root not in p.parents or sha256(p) != digest:
            raise ValueError(f"Artifact hash/path mismatch: {name}")


TASK_SOURCES = {
    "kidney_history_binary": ("mhoccur_rnl", 2005200017),
    "lung_history_binary": ("mhoccur_plm", 4186898),
    "cancer_history_binary": ("mhoccur_ca", 4194405),
    "arthritis_history_binary": ("mhoccur_ra", 4291025),
    "osteoporosis_history_binary": ("mhoccur_oa", 80502),
    "digestive_history_binary": ("mhoccur_gi", 4201745),
    "urinary_history_binary": ("mhoccur_ua", 81902),
}
TASKS = tuple(TASK_SOURCES)
SPEC_PATH = Path(__file__).resolve().parent/"configs/history_tasks.json"


def load_spec():
    spec = read_json(SPEC_PATH)
    if spec["schema_version"] != "ot2-history-task-spec-v1" or tuple(spec["tasks"]) != TASKS:
        raise ValueError("Unexpected history-task registry")
    for task, (source, condition_id) in TASK_SOURCES.items():
        item = spec["tasks"][task]
        if item["source_key"] != source or item["condition_concept_id"] != condition_id:
            raise ValueError("Changed source or disease meaning: "+task)
    if spec["observation_concept_id"] != 2005200014 or spec["yes_concept_ids"] != [4188539, 45877994] or spec["no_concept_id"] != 45878245 or spec["refused_concept_id"] != 36311156:
        raise ValueError("Changed questionnaire identity or response mapping")
    return spec


def task_hash(task):
    spec = load_spec()
    return canonical_hash({k: v for k, v in spec.items() if k != "tasks"} | {"task": task, "definition": spec["tasks"][task]})


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if str(path).endswith(".tsv") else ",")
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing/duplicate header: "+str(path))
        for row in reader:
            if None in row or None in row.values():
                raise ValueError("Malformed row: "+str(path))
            yield row


def write_rows(path, rows, columns):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def integer(value):
    try:
        n = float(str(value).strip())
        return int(n) if math.isfinite(n) and n.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def source_key(value):
    return str(value).split(",", 1)[0].strip().lower()


def parse_calendar_date(raw):
    """Accept ISO timestamps and the local M/D/YY[YY] [H:MM[:SS]] export."""
    text = str(raw).strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        if len(text) == 10:
            return date.fromisoformat(text)
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    if text.count("/") == 2:
        calendar, _, time = text.partition(" ")
        year = calendar.rsplit("/", 1)[1]
        if len(year) not in (2, 4):
            raise ValueError("invalid_date")
        fmt = "%m/%d/%y" if len(year) == 2 else "%m/%d/%Y"
        if time:
            fmt += " %H:%M:%S" if time.count(":") == 2 else " %H:%M"
        return datetime.strptime(text, fmt).date()
    raise ValueError("invalid_date")


def recorded_date(row, prefix):
    dates = []
    for field in (prefix+"_date", prefix+"_datetime"):
        raw = str(row.get(field, "")).strip()
        if raw.lower() in ("", "nan", "nat", "null", "none"):
            continue
        try:
            dates.append(parse_calendar_date(raw))
        except ValueError:
            raise ValueError("invalid_date") from None
    if not dates or len(set(dates)) != 1:
        raise ValueError("missing_or_conflicting_date")
    return dates[0]


def response(row, definition, spec):
    if integer(row.get("observation_concept_id")) != spec["observation_concept_id"]:
        raise ValueError("observation_concept_mismatch")
    if integer(row.get("qualifier_concept_id")) != definition["condition_concept_id"]:
        raise ValueError("disease_qualifier_mismatch")
    cid = integer(row.get("value_as_concept_id"))
    if cid in spec["yes_concept_ids"]:
        category, numeric = 1, 1
    elif cid == spec["no_concept_id"]:
        category, numeric = 0, 0
    elif cid == spec["refused_concept_id"]:
        category, numeric = None, 777
    else:
        raise ValueError("unrecognized_response_id")
    for field in ("value_as_number", "value_as_string"):
        value = str(row.get(field, "")).strip()
        if value.lower() not in ("", "nan", "none", "null") and integer(value) != numeric:
            raise ValueError("response_encoding_conflict")
    return category


def build(participants, observation, condition, out_dir):
    out = Path(out_dir).resolve()
    paths = {"participants": Path(participants).resolve(strict=True), "observation": Path(observation).resolve(strict=True),
             "condition": Path(condition).resolve(strict=True), "spec": SPEC_PATH.resolve(),
             "registry": Path(__file__).resolve()}
    for p in paths.values():
        if out == p or out in p.parents:
            raise ValueError("Output must not contain source files")
    hashes = {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()}
    if out.exists():
        receipt = read_json(out/"LABELS.json")
        verify_files(out, receipt["output_sha256"])
        if receipt["source_files"] != hashes or receipt["builder_sha256"] != sha256(__file__):
            raise ValueError("Existing labels belong to changed inputs/code; use a new output")
        return receipt
    spec = load_spec()
    people = list(read_rows(paths["participants"]))
    ids = [r["person_id"].strip() for r in people]
    if not ids or any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Participants require unique nonempty IDs")
    if any(r["recommended_split"] not in ("train", "val", "test") for r in people):
        raise ValueError("Invalid recommended split")
    by_source = {v["source_key"]: (k, v) for k, v in spec["tasks"].items()}
    known = set(ids)
    observations, conditions = defaultdict(list), defaultdict(list)
    encodings = {t: Counter() for t in TASKS}
    for kind, column in (("observation", "observation_source_value"), ("condition", "condition_source_value")):
        for row in read_rows(paths[kind]):
            source = source_key(row[column])
            if source not in by_source:
                continue
            pid = row["person_id"].strip()
            if pid not in known:
                raise ValueError("Target record has a person absent from participants.tsv")
            task, definition = by_source[source]
            (observations if kind == "observation" else conditions)[pid, task].append(row)
            if kind == "observation":
                encodings[task][(str(row.get("value_as_concept_id", "")), str(row.get("value_as_number", "")))]+=1
    audits, labels, summaries = {}, {}, []
    date_summaries = {}
    for task, definition in spec["tasks"].items():
        rows, valid, offsets = [], [], []
        for person in people:
            pid = person["person_id"].strip(); reasons = set(); answers = set(); dates = set()
            obs, cond = observations[pid, task], conditions[pid, task]
            for r in obs:
                try:
                    answers.add(response(r, definition, spec)); dates.add(recorded_date(r, "observation"))
                except ValueError as exc:
                    reasons.add(str(exc))
            for r in cond:
                if integer(r.get("condition_concept_id")) != definition["condition_concept_id"]:
                    reasons.add("condition_concept_mismatch")
            if not obs: reasons.add("missing_questionnaire_response")
            if len(dates) > 1: reasons.add("multiple_questionnaire_dates")
            if len(answers) > 1: reasons.add("conflicting_questionnaire_answers")
            if answers == {None}: reasons.add("refused_response")
            if bool(cond) != (answers == {1}): reasons.add("condition_observation_mismatch")
            if not answers and not reasons: reasons.add("no_usable_response")
            when = next(iter(dates)).isoformat() if len(dates)==1 else ""
            offset = (date.fromisoformat(when)-date.fromisoformat(person["study_visit_date"])).days if when else None
            if offset is not None: offsets.append(offset)
            y = next(iter(answers)) if not reasons and len(answers)==1 else None
            row = {"person_id": pid, "task": task, "split": person["recommended_split"],
                   "y": "" if y is None else y, "label_value": "" if y is None else y,
                   "label_datetime": when, "label_offset_days": "" if offset is None else offset,
                   "label_status": ";".join(sorted(reasons)) if reasons else "positive_verified" if y==1 else "negative_explicit_no",
                   "source_key": definition["source_key"], "observation_records": len(obs), "condition_records": len(cond),
                   "task_config_hash": task_hash(task)}
            rows.append(row)
            if y is not None: valid.append(row)
        audits[task], labels[task] = rows, valid
        date_summaries[task] = {"n":len(offsets), "min":min(offsets) if offsets else None,
            "median":statistics.median(offsets) if offsets else None, "max":max(offsets) if offsets else None,
            "after_visit":sum(x>0 for x in offsets), "at_or_after_day10":sum(x>=10 for x in offsets)}
        for split in ("train", "val", "test"):
            selected = [r for r in rows if r["split"]==split]
            positive = sum(r["y"]==1 for r in selected); negative=sum(r["y"]==0 for r in selected)
            summaries.append({"task":task, "label":definition["label"], "split":split, "n_total":len(selected),
                "n_positive":positive, "n_negative":negative, "n_valid":positive+negative,
                "n_excluded":len(selected)-positive-negative,
                "prevalence":positive/(positive+negative) if positive+negative else None,
                "status":"two_classes" if positive and negative else "insufficient_classes"})
    if any(sha256(p) != hashes[k]["sha256"] for k,p in paths.items()):
        raise ValueError("Source changed during label extraction")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".history-labels-",dir=out.parent) as temporary:
        ready=Path(temporary)/"ready";ready.mkdir()
        for task in TASKS:
            for kind, rows in (("labels",labels[task]),("audit",audits[task])):
                write_rows(ready/(task+"_"+kind+".csv"),rows,list(audits[task][0]))
        write_rows(ready/"cohort_counts.csv",summaries,list(summaries[0]))
        lines=["# Candidate history-task label audit", "", "Self-reported history of a clinician diagnosis; seven separate binary tasks. Counts describe label feasibility, not predictive performance.", "",
            "| Task | Split | Positive | Negative | Excluded | Positive / valid |", "|---|---|---:|---:|---:|---:|"]
        for r in summaries:
            rate=f"{r['prevalence']:.1%}" if r['prevalence'] is not None else "NA"
            lines.append(f"| {r['label']} | {r['split']} | {r['n_positive']} | {r['n_negative']} | {r['n_excluded']} | {rate} |")
        lines += ["", "Positive requires an explicit Yes and a matching source-specific condition record. Negative requires explicit No and no contradictory condition. Refusal, missing, ID/encoding conflicts or multiple response dates are excluded. Source key and disease qualifier are essential: all seven questions share observation_concept_id=2005200014.", "",
            "Official answer rows list Yes=45877994; the supplied raw data use Yes=4188539. Both explicitly listed Yes codes are accepted only with consistent numeric encodings. No=45878245; Prefer not to say=36311156 (777). No numeric-only fallback.", "",
            "Questionnaire dates are audited relative to study visit. No new fixed date window is imposed; the experiment retains the existing cross-sectional history scope. This is not prospective risk prediction. Test labels are counted only; there is no test model evaluation.", "",
            "| Task | Earliest offset (days) | Median | Latest | After visit N | At/after day 10 N |", "|---|---:|---:|---:|---:|---:|"]
        for task,d in date_summaries.items():
            lines.append(f"| {spec['tasks'][task]['label']} | {d['min']} | {d['median']} | {d['max']} | {d['after_visit']} | {d['at_or_after_day10']} |")
        atomic_text(ready/"SUMMARY_HISTORY_LABELS.md","\n".join(lines)+"\n")
        receipt={"schema_version":"ot2-history-labels-v1","source_files":hashes,"builder_sha256":sha256(__file__),
            "official_train_person_ids_hash":canonical_hash({"person_ids":sorted(r['person_id'].strip() for r in people if r['recommended_split']=='train')}),
            "n_participants":len(people),"specification":spec,"task_config_hashes":{t:task_hash(t) for t in TASKS},
            "counts":summaries,"date_offset_audit":date_summaries,"test_inference":False,
            "response_encodings":{t:[{"concept_id":a,"numeric_value":b,"n":n} for (a,b),n in encodings[t].items()] for t in TASKS},
            "exclusion_counts":{t:dict(Counter(r['label_status'] for r in audits[t] if r['y']=="")) for t in TASKS},
            "output_sha256":{p.name:sha256(p) for p in ready.iterdir()}}
        write_json(ready/"LABELS.json",receipt);ready.rename(out)
    print(f"HISTORY_LABELS_COMPLETE: {out/'SUMMARY_HISTORY_LABELS.md'}",flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participants",type=Path,required=True)
    p.add_argument("--clinical-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    build(a.participants,a.clinical_dir/"observation.csv",a.clinical_dir/"condition_occurrence.csv",a.out_dir)


if __name__=="__main__":main()
