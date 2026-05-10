#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage II pipeline orchestrator."""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from stage_II.config import Stage2Config, resolve_path
from stage_II.features.smart import build_smart_ir, infer_smart_columns
from stage_II.features.env import build_env_ir
from stage_II.features.workload import build_workload_ir
from stage_II.features.flash_type import build_flash_type_ir
from stage_II.features.algorithms import build_algorithms_ir
from stage_II.kg.data_kg import build_data_kg
from stage_II.kg.literature_kg import LiteratureKG
from stage_II.llm.openai_client import OpenAIChatClient
from stage_II.prompts.templates import (
    system_prompt,
    predictive_user_prompt,
    descriptive_user_prompt,
    prescriptive_user_prompt,
    whatif_user_prompt,
)
from stage_II.utils.io import ensure_dir, read_csv, write_json, append_jsonl, write_csv
from stage_II.utils.json_utils import extract_json_object
from stage_II.evaluation.metrics_predictive import confusion_from_labels, mse
from stage_II.evaluation.metrics_text import bleu4, rouge_l_f1
from stage_II.evaluation.grounding import faithfulness_precision, counterfactual_validity

def _load_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _resolve_first_existing(repo_root: Path, *candidates: Path) -> Path:
    for candidate in candidates:
        resolved = resolve_path(repo_root, candidate)
        if resolved.exists():
            return resolved
    return resolve_path(repo_root, candidates[0])


def _parse_extra_terms(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    parts = [p.strip() for p in s.replace("|", ";").split(";")]
    return [p for p in parts if p]


def _task_query_text(row: Dict[str, Any], task: str) -> Optional[str]:
    for key in (f"{task}_query", "query_text"):
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _row_tail_latency_target(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    for key, unit in (
        ("tail_latency_ms", "ms"),
        ("tail_latency", "ms"),
        ("tail_latency_change_pct", "pct_change"),
    ):
        if key not in row:
            continue
        val = row.get(key)
        if val is None or str(val).strip() == "":
            continue
        try:
            return float(val), unit
        except Exception:
            continue
    return None, None


def _out_tail_latency_prediction(out: Dict[str, Any], unit: Optional[str]) -> Optional[float]:
    candidates: List[str] = []
    if unit == "pct_change":
        candidates.extend(["predicted_tail_latency_pct", "predicted_tail_latency_ms"])
    else:
        candidates.extend(["predicted_tail_latency_ms", "predicted_tail_latency_pct"])
    for key in candidates:
        val = out.get(key)
        if val is None or str(val).strip() == "":
            continue
        try:
            return float(val)
        except Exception:
            continue
    return None


def _recommendations_to_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if not isinstance(value, list):
        text = str(value).strip()
        return text or None
    parts: List[str] = []
    for rec in value:
        if isinstance(rec, dict):
            action = str(rec.get("action", "")).strip()
            justification = str(rec.get("justification", "")).strip()
            priority = str(rec.get("priority", "")).strip()
            text = action
            if justification:
                text = f"{text}: {justification}" if text else justification
            if priority:
                text = f"[{priority}] {text}" if text else f"[{priority}]"
            if text:
                parts.append(text)
        else:
            text = str(rec).strip()
            if text:
                parts.append(text)
    return " ".join(parts) if parts else None


def _task_text_output(task: str, out: Dict[str, Any]) -> Optional[str]:
    if task == "descriptive":
        text = out.get("summary")
        return str(text).strip() if text is not None and str(text).strip() else None
    if task == "prescriptive":
        return _recommendations_to_text(out.get("recommendations"))
    if task == "whatif":
        for key in ("analysis", "summary"):
            text = out.get(key)
            if text is not None and str(text).strip():
                return str(text).strip()
        return None
    return None


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out

def _infer_query_terms(ir: Dict[str, Any]) -> List[str]:
    terms = []
    if "env" in ir:
        for k in ["temperature", "humidity", "vibration"]:
            terms.append(k)
    if "workload" in ir:
        terms.append("workload")
        w = ir.get("workload", {})
        if isinstance(w, dict):
            if w.get("type") == "app_tag":
                terms.append(str(w.get("value")))
            if w.get("type") == "fio":
                terms.append(str(w.get("rw","")))
    # SMART always: some common terms
    terms.extend(["SMART", "SSD", "wear", "uncorrectable", "ECC"])
    return [t for t in terms if t and str(t).strip()]

def _default_whatif_scenario(ir: Dict[str, Any]) -> str:
    # Try to pick a scenario compatible with modalities.
    if "env" in ir and isinstance(ir["env"], dict):
        return "If inlet temperature decreases by 5°C and relative humidity decreases by 10%, how do tail latency and failure risk change?"
    if "workload" in ir:
        return "If the workload shifts to higher write intensity (e.g., rwmixread decreases by 20 points), what happens to wear-related SMART signals and failure risk?"
    return "If background garbage collection aggressiveness increases and we throttle writes, how do failure risk and tail latency change?"

@dataclass
class RunOutputs:
    run_dir: Path
    responses_jsonl: Path
    metrics_csv: Path
    summary_json: Path
    data_kg_dir: Path

class Stage2Runner:
    def __init__(self, cfg: Stage2Config):
        self.cfg = cfg
        self.repo_root = cfg.repo_root

        # Load base system guidance (optional)
        cot_path = _resolve_first_existing(
            self.repo_root,
            Path("ssd_cot_prompt.txt"),
            Path("stage_I/ssd_cot_prompt.txt"),
        )
        self.base_cot = _load_text(cot_path)

        # LLM client
        self.llm = OpenAIChatClient(model=cfg.model)

        # Literature KG
        lit_path = _resolve_first_existing(
            self.repo_root,
            cfg.global_kg_ttl_path,
            Path("stage_I/out/global_knowledge_graph.ttl"),
            Path("stage_I/global_knowledge_graph.ttl"),
        )
        self.lit = LiteratureKG(lit_path)
        try:
            self.lit.load()
        except Exception:
            # Still usable in grep mode if rdflib missing
            pass

    def run(
        self,
        input_csv: Path,
        tasks: List[str],
        out_name: str,
        limit_rows: Optional[int] = None,
        seed: int = 7,
    ) -> RunOutputs:
        df = read_csv(input_csv)
        if limit_rows is not None:
            df = df.head(int(limit_rows)).copy()

        run_dir = ensure_dir(resolve_path(self.repo_root, self.cfg.runs_dir) / out_name)
        data_kg_dir = ensure_dir(run_dir / "data_kg_ttl")
        responses_jsonl = run_dir / "responses.jsonl"
        metrics_csv = run_dir / "metrics_per_sample.csv"
        summary_json = run_dir / "metrics_summary.json"

        # Keep a copy of input for reproducibility
        write_csv(run_dir / "input_samples.csv", df)

        # Determine SMART columns by header
        smart_cols = infer_smart_columns(list(df.columns))

        sys = system_prompt(self.base_cot)

        rows_out = []
        metrics_rows = []

        rng = seed

        for idx, r in df.iterrows():
            row = r.to_dict()

            sample_id = str(row.get("sample_id") or row.get("window_id") or row.get("id") or f"s{idx}")
            # Build IR
            ir: Dict[str, Any] = {}
            ir.update(build_smart_ir(row, smart_cols))
            ir.update(build_env_ir(row))
            ir.update(build_workload_ir(row))
            ir.update(build_flash_type_ir(row))
            ir.update(build_algorithms_ir(row))

            # Data KG
            dk = build_data_kg(sample_id, ir)
            if dk.ttl:
                (data_kg_dir / f"{sample_id}.ttl").write_text(dk.ttl, encoding="utf-8")

            # Literature retrieval
            terms = _unique_preserve_order(_infer_query_terms(ir) + _parse_extra_terms(row.get("retrieval_terms")))
            lit_evidence = self.lit.retrieve(terms, limit=8)
            lit_payload = [{"id": e.id, "text": e.text, "source": e.source} for e in lit_evidence]

            # Compose sample payload passed to prompt
            sample_payload = {
                "sample_id": sample_id,
                "meta": {
                    "dataset": row.get("dataset_type") or row.get("dataset") or None,
                    "disk_id": row.get("disk_id") or row.get("drive_id") or None,
                    "ds": row.get("ds") or row.get("date") or None,
                    "label": int(row.get("failure") or row.get("label") or 0) if str(row.get("failure") or row.get("label") or 0).strip() != "" else 0,
                    "ttf_days": row.get("ttf_days") if "ttf_days" in row else (row.get("ttf") if "ttf" in row else None),
                    "tail_latency_ms": row.get("tail_latency_ms") if "tail_latency_ms" in row else (row.get("tail_latency") if "tail_latency" in row else None),
                    "tail_latency_change_pct": row.get("tail_latency_change_pct") if "tail_latency_change_pct" in row else None,
                },
                "IR": ir,
                "DataKG_refs": sorted(list(dk.refs)),
                "Literature": lit_payload,
            }

            # For grounding, allow "IR:" refs too
            available_refs = set(dk.refs)
            for af in ir.get("smart", []):
                if isinstance(af, dict) and af.get("id"):
                    available_refs.add(af["id"])
            # Add literature ids as acceptable refs
            for e in lit_evidence:
                available_refs.add(e.id)

            # Execute tasks
            for task in tasks:
                question = _task_query_text(row, task)
                if task == "predictive":
                    user = predictive_user_prompt(sample_payload, question=question)
                elif task == "descriptive":
                    user = descriptive_user_prompt(sample_payload, question=question)
                elif task == "prescriptive":
                    user = prescriptive_user_prompt(sample_payload, question=question)
                elif task == "whatif":
                    scenario = str(row.get("whatif_scenario") or _default_whatif_scenario(ir))
                    user = whatif_user_prompt(sample_payload, scenario, question=question)
                else:
                    raise ValueError(f"Unknown task: {task}")

                resp = self.llm.chat(
                    system=sys,
                    user=user,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    seed=rng,
                )
                rng += 1

                parsed = extract_json_object(resp.text) or {"task": task, "sample_id": sample_id, "parse_error": True, "raw_text": resp.text}

                rows_out.append({
                    "sample_id": sample_id,
                    "task": task,
                    "prompt_terms": terms,
                    "response_text": resp.text,
                    "response_json": parsed,
                })

                # Metrics per task/sample (computed later more fully)
                m = {"sample_id": sample_id, "task": task}
                # FiP / CFV
                if task in ("descriptive", "prescriptive"):
                    fip = faithfulness_precision(parsed, available_refs)
                    m["FiP"] = fip
                    parsed["FiP"] = fip
                if task == "whatif":
                    cfv = counterfactual_validity(parsed, direction_lookup=None)
                    m["CFV"] = cfv
                    parsed["CFV"] = cfv
                metrics_rows.append(m)

                # polite pacing
                time.sleep(0.2)

        # Save responses
        append_jsonl(responses_jsonl, rows_out)
        mdf = pd.DataFrame(metrics_rows)
        write_csv(metrics_csv, mdf)

        # Aggregate metrics summary (requires joining predictions/references)
        summary = self._aggregate(df, rows_out)
        write_json(summary_json, summary)

        return RunOutputs(
            run_dir=run_dir,
            responses_jsonl=responses_jsonl,
            metrics_csv=metrics_csv,
            summary_json=summary_json,
            data_kg_dir=data_kg_dir,
        )

    def _aggregate(self, df_in: pd.DataFrame, rows_out: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across tasks. If reference columns are absent, metrics are skipped."""
        # Build lookup: (sample_id, task) -> parsed json
        lookup = {}
        for r in rows_out:
            key = (r["sample_id"], r["task"])
            lookup[key] = r.get("response_json", {})

        # Predictive: Precision/Recall/Accuracy if label exists
        y_true = []
        y_pred = []
        ttf_true = []
        ttf_pred = []
        tl_true = []
        tl_pred = []
        for _, row in df_in.iterrows():
            sid = str(row.get("sample_id") or row.get("window_id") or row.get("id") or f"s{_}")
            out = lookup.get((sid, "predictive"))
            if not out:
                continue
            gt = row.get("failure", None)
            if gt is None:
                gt = row.get("label", None)
            if gt is not None and str(gt).strip() != "":
                try:
                    yp = out.get("predicted_failure", None)
                    if yp is not None and str(yp).strip() != "":
                        y_true.append(int(gt))
                        y_pred.append(int(yp))
                except Exception:
                    pass

            ttf_gt = row.get("ttf_days", row.get("ttf", None))
            ttf_out = out.get("predicted_ttf_days", None)
            if ttf_gt is not None and str(ttf_gt).strip() != "" and ttf_out is not None and str(ttf_out).strip() != "":
                try:
                    ttf_true.append(float(ttf_gt))
                    ttf_pred.append(float(ttf_out))
                except Exception:
                    pass

            tl_gt, tl_unit = _row_tail_latency_target(row)
            tl_out = _out_tail_latency_prediction(out, tl_unit)
            if tl_gt is not None and tl_out is not None:
                tl_true.append(float(tl_gt))
                tl_pred.append(float(tl_out))

        pred_metrics = {}
        if y_true and y_pred:
            conf = confusion_from_labels(y_true, y_pred)
            pred_metrics = {
                "P": conf.precision(),
                "R": conf.recall(),
                "A": conf.accuracy(),
                "TP": conf.tp, "FP": conf.fp, "FN": conf.fn, "TN": conf.tn,
            }
        if ttf_true and ttf_pred:
            pred_metrics["TTF_MSE"] = mse(ttf_true, ttf_pred)
        if tl_true and tl_pred:
            pred_metrics["TL_MSE"] = mse(tl_true, tl_pred)

        # Text metrics: require reference columns
        def text_metrics(task: str, ref_col: str) -> Dict[str, Any]:
            b4s, rls = [], []
            fips, cfvs = [], []
            for _, row in df_in.iterrows():
                sid = str(row.get("sample_id") or row.get("window_id") or row.get("id") or f"s{_}")
                ref = row.get(ref_col, None)
                if ref is None or str(ref).strip() == "":
                    continue
                out = lookup.get((sid, task))
                if not out:
                    continue
                gen = _task_text_output(task, out)
                if gen is None:
                    continue
                b4s.append(bleu4(str(gen), str(ref)))
                rls.append(rouge_l_f1(str(gen), str(ref)))
                if task in ("descriptive","prescriptive"):
                    fips.append(float(out.get("FiP", 0.0)) if "FiP" in out else 0.0)
                if task == "whatif":
                    cfvs.append(float(out.get("CFV", 0.0)) if "CFV" in out else 0.0)

            outm = {}
            if b4s:
                outm["B4"] = float(sum(b4s)/len(b4s))
            if rls:
                outm["RL"] = float(sum(rls)/len(rls))
            if fips:
                outm["FiP"] = float(sum(fips)/len(fips))
            if cfvs:
                outm["CFV"] = float(sum(cfvs)/len(cfvs))
            outm["n_ref"] = int(len(b4s))
            return outm

        desc = text_metrics("descriptive", "ref_descriptive")
        pres = text_metrics("prescriptive", "ref_prescriptive")
        wif = text_metrics("whatif", "ref_whatif")

        return {
            "predictive": pred_metrics,
            "descriptive": desc,
            "prescriptive": pres,
            "whatif": wif,
            "notes": {
                "reference_columns_expected": ["ref_descriptive", "ref_prescriptive", "ref_whatif"],
                "if_missing": "Text overlap metrics are skipped; FiP/CFV still computed from evidence refs when available.",
            }
        }
