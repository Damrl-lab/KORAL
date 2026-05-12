#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Curated Table I benchmark materialization helpers."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from stage_II.features.smart import build_smart_ir, infer_smart_columns

QUERY_BANK_PATH = Path(__file__).resolve().parent / "query_bank.json"
_TEMPLATE_FIELD_PATTERN = re.compile(r"{([^{}]+)}")
_TEMPLATE_FIELD_ALIASES = {
    "smart_summary": "_smart_health_sentence",
}

WRITE_DOMINANT_APPS = {"WSM", "RM", "WS", "DAE", "NAS"}
READ_DOMINANT_APPS = {"WPS", "SS", "DB"}

SMART_LABELS = {
    "r_5": "reallocated sectors",
    "r_9": "power-on hours",
    "r_173": "wear leveling count",
    "r_177": "wear range delta",
    "r_187": "reported uncorrectable errors",
    "r_199": "UltraDMA CRC errors",
    "r_233": "media wearout indicator",
    "r_241": "blocks written",
    "r_242": "blocks read",
}


def load_query_bank(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    query_path = Path(path) if path is not None else QUERY_BANK_PATH
    data = json.loads(query_path.read_text(encoding="utf-8"))
    return list(data.get("queries", []))


def materialize_table1_task_inputs(
    dataset_type: str,
    input_df: pd.DataFrame,
    task: str,
    limit_rows: Optional[int] = None,
    query_bank_path: Optional[Path] = None,
) -> pd.DataFrame:
    base_df = input_df.head(int(limit_rows)).copy() if limit_rows is not None else input_df.copy()
    if base_df.empty:
        return base_df

    smart_cols = infer_smart_columns(list(base_df.columns))
    queries = [
        q for q in load_query_bank(query_bank_path)
        if q.get("task") == task and dataset_type in q.get("dataset_types", [])
    ]
    rows: List[Dict[str, Any]] = []

    for idx, record in enumerate(base_df.to_dict(orient="records")):
        context = _build_context(record, dataset_type, smart_cols)
        for query in queries:
            if not _query_applies(query, context):
                continue

            row = dict(record)
            base_sample_id = str(
                row.get("sample_id") or row.get("window_id") or row.get("id") or f"s{idx}"
            )
            row["base_sample_id"] = base_sample_id
            row["sample_id"] = f"{base_sample_id}__{query['id']}"
            row["dataset_type"] = dataset_type
            row["benchmark_task"] = task
            row["benchmark_query_id"] = query["id"]
            row["benchmark_query_group"] = query.get("group", "")
            row["retrieval_terms"] = json.dumps(
                _unique_preserve_order(query.get("retrieval_terms", []) + _contextual_terms(context)),
                ensure_ascii=False,
            )

            if task == "predictive":
                _attach_predictive_targets(row, context)
                row["predictive_query"] = str(query.get("question", "")).strip()
            elif task == "descriptive":
                row["descriptive_query"] = str(query.get("question", "")).strip()
                row["ref_descriptive"] = _render_reference(query, context, "descriptive")
            elif task == "prescriptive":
                row["prescriptive_query"] = str(query.get("question", "")).strip()
                row["ref_prescriptive"] = _render_reference(query, context, "prescriptive")
            elif task == "whatif":
                row["whatif_query"] = str(query.get("question", "")).strip()
                row["whatif_scenario"] = str(query.get("scenario", "")).strip()
                row["ref_whatif"] = _render_reference(query, context, "whatif")
            else:
                raise ValueError(f"Unsupported benchmark task: {task}")

            rows.append(row)

    return pd.DataFrame(rows)


def _query_applies(query: Dict[str, Any], context: Dict[str, Any]) -> bool:
    row_factors = query.get("row_factors_any")
    if not row_factors:
        return True
    factor_text = context.get("factor_text", "")
    return any(token in factor_text for token in row_factors)


def _build_context(row: Dict[str, Any], dataset_type: str, smart_cols: List[str]) -> Dict[str, Any]:
    smart_frames = build_smart_ir(row, smart_cols).get("smart", [])
    smart_by_attr = {frame.get("attribute"): frame for frame in smart_frames if isinstance(frame, dict)}

    factor_text = _lower_text(row.get("factor")) or _lower_text(row.get("condition")) or ""
    device_type = _text(row.get("device_type"))
    flash_type = _infer_flash_type(row, device_type)
    algorithms = _infer_algorithms(row)
    app_tag = _text(row.get("app"))
    workload = _text(row.get("workload_profile")) or _text(row.get("workload")) or app_tag
    workload_profile = _infer_workload_profile(app_tag, smart_by_attr, workload)
    age_years = _infer_age_years(smart_by_attr)
    smart_signals = _top_smart_signals(smart_by_attr, age_years, workload_profile)
    smart_risk = _smart_risk_label(smart_by_attr, age_years, workload_profile)

    return {
        "dataset_type": dataset_type,
        "factor_text": factor_text,
        "condition": _text(row.get("condition")),
        "exposure": _text(row.get("exposure")),
        "vibration_orientation": _text(row.get("vibration_orientation")),
        "direction": _lower_text(row.get("direction")),
        "metric": _lower_text(row.get("metric")),
        "metric_percentile": _text(row.get("metric_percentile")),
        "change_min": _to_float(row.get("change_pct_min")),
        "change_max": _to_float(row.get("change_pct_max")),
        "device_type": device_type,
        "flash_type": flash_type,
        "algorithms": algorithms,
        "workload": workload,
        "workload_profile": workload_profile,
        "age_years": age_years,
        "smart_signals": smart_signals,
        "smart_risk": smart_risk,
        "has_smart": bool(smart_cols),
    }


def _contextual_terms(context: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    if context.get("factor_text"):
        terms.extend(part.strip() for part in context["factor_text"].split("+"))
    if context.get("flash_type"):
        terms.append(str(context["flash_type"]))
    if context.get("algorithms"):
        terms.extend(str(policy) for policy in context["algorithms"])
    workload = context.get("workload")
    if workload:
        terms.append(str(workload))
    return [t for t in terms if t]


def _attach_predictive_targets(row: Dict[str, Any], context: Dict[str, Any]) -> None:
    metric = context.get("metric") or ""
    if "tail_latency" in metric and _text(row.get("tail_latency_change_pct")) is None:
        midpoint = _range_midpoint(context.get("change_min"), context.get("change_max"))
        if midpoint is not None:
            row["tail_latency_change_pct"] = midpoint


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _render_reference(query: Dict[str, Any], context: Dict[str, Any], task: str) -> str:
    template = _text(query.get("ground_truth"))
    if not template:
        raise ValueError(f"No ground_truth template defined for {task}:{query.get('id')}")
    field_names = _TEMPLATE_FIELD_PATTERN.findall(template)
    rendered = template.format_map(_SafeFormatDict(_template_values(context, field_names)))
    return " ".join(rendered.split())


def _template_values(context: Dict[str, Any], field_names: Iterable[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for name in _unique_preserve_order(field_names):
        values[name] = _render_template_field(name, context)
    return values


def _render_template_field(name: str, context: Dict[str, Any]) -> str:
    func_name = _TEMPLATE_FIELD_ALIASES.get(name, f"_{name}")
    func = globals().get(func_name)
    if not callable(func):
        raise ValueError(f"No template renderer defined for field: {name}")
    if len(inspect.signature(func).parameters) == 0:
        value = func()
    else:
        value = func(context)
    return " ".join(str(value).split()) if value else ""


def _render_smart_descriptive(query_id: str, context: Dict[str, Any]) -> str:
    pieces = [_smart_health_sentence(context)]
    if query_id in {"desc_smart_health_overview", "desc_smart_workload_risk"}:
        pieces.append(_smart_workload_sentence(context))
    if query_id in {"desc_smart_health_overview", "desc_smart_correlation_limits"}:
        pieces.append(_smart_limit_sentence())
    if query_id == "desc_smart_static_context":
        pieces.append(_smart_static_sentence(context))
    return " ".join(piece for piece in pieces if piece)


def _render_smart_prescriptive(query_id: str, context: Dict[str, Any]) -> str:
    if query_id == "pres_smart_monitor_replace":
        return (
            f"Prioritize { _monitoring_urgency(context) } monitoring of the SMART window, verify whether the abnormal counters keep rising, "
            "and migrate or replace the drive if media or interface errors continue to accumulate. "
            "Use the device-specific SMART evidence for local triage, but keep the explanation concise and evidence-backed."
        )
    if query_id == "pres_smart_reduce_correlated_risk":
        return (
            "Reduce correlated-failure exposure by avoiding dense placement of same-model SSDs in the same node or rack, "
            "using stronger fault-tolerance where possible, and preferring eager recovery over lazy recovery once failures are detected. "
            "Do not rely on SMART counters alone for node- or rack-level correlation decisions."
        )
    if query_id == "pres_smart_write_heavy_controls":
        return (
            f"If the workload remains {context.get('workload_profile')}, rebalance or throttle writes, move latency-sensitive traffic away from stressed drives, "
            "and shorten inspection intervals for media and interface counters. "
            "The Alibaba study indicates write-dominant workloads create higher overall SSD failure exposure than read-dominant ones."
        )
    if query_id == "pres_smart_static_policy_alignment":
        return (
            "Treat static device descriptors as operating context and align controller behavior conservatively: monitor garbage-collection pressure, wear leveling, "
            "refresh behavior, and read/write amplification, and tighten lifecycle thresholds for older or higher-density flash. "
            f"Current static context: {_smart_static_clause(context)}."
        )
    raise ValueError(f"Unhandled smart prescriptive query: {query_id}")


def _render_smart_whatif(query_id: str, context: Dict[str, Any]) -> str:
    if query_id == "whatif_smart_reduce_writes":
        return (
            f"If the workload becomes less write-intensive than the current {context.get('workload_profile')}, wear-related SMART pressure and overall failure exposure should decrease. "
            "The improvement is strongest when the starting point is clearly write-dominant."
        )
    if query_id == "whatif_smart_ageing_next_year":
        return (
            "If the drive accumulates more age and rated-life usage, correlated-failure risk should rise and short-interval failures become more plausible. "
            "The Alibaba study found older SSDs are more likely to exhibit intra-node and intra-rack failures within short time intervals."
        )
    if query_id == "whatif_smart_spread_same_model":
        return (
            "If same-model SSDs are spread across nodes and racks instead of densely colocated, correlated-failure exposure should decrease. "
            "The Alibaba study observed that placing many SSDs from the same drive model in the same scope raises intra-node and intra-rack failure risk."
        )
    if query_id == "whatif_smart_eager_recovery":
        return (
            "If the system switches from lazy recovery to eager recovery, reliability should improve under correlated failures because additional failures are more likely after initial ones. "
            "The Alibaba study explicitly concluded that lazy recovery is less suitable than eager recovery for this failure pattern."
        )
    raise ValueError(f"Unhandled smart what-if query: {query_id}")


def _render_env_descriptive(query_id: str, context: Dict[str, Any]) -> str:
    pieces = [_env_runtime_sentence(context)]
    if query_id == "desc_env_post_impact":
        pieces.append(_env_postimpact_sentence(context))
    if query_id == "desc_env_bandwidth_latency_tradeoff":
        pieces.append(_env_tradeoff_sentence(context))
    if query_id == "desc_env_device_sensitivity":
        pieces.append(_env_device_sensitivity_sentence(context))
    return " ".join(piece for piece in pieces if piece)


def _render_env_prescriptive(query_id: str, context: Dict[str, Any]) -> str:
    if query_id == "pres_env_humidity_mitigation":
        return (
            "Move the operating point back toward room conditions when humidity is high, avoid prolonged exposure, and treat high-humidity episodes as performance-risk events rather than harmless noise. "
            "Humidity reduction is the safest first mitigation because the paper reports better performance when humidity decreases."
        )
    if query_id == "pres_env_thermal_window":
        return (
            "Keep the device inside vendor limits, use any high-temperature operating point only deliberately and temporarily, and avoid trading a short-term bandwidth gain for longer-term retention or post-impact risk. "
            "When possible, schedule demanding work after the environment stabilizes."
        )
    if query_id == "pres_env_post_impact_monitoring":
        return (
            "After the exposure ends, continue monitoring tail latency, bandwidth, media-wear indicators, and unexpected failures because the paper reports post-impact after high humidity and some high-temperature exposures. "
            "Do not assume the drive is healthy simply because the chamber returned to nominal conditions."
        )
    if query_id == "pres_env_write_heavy_scheduling":
        return (
            "Protect write-heavy or latency-critical workloads by moving them away from the stressed interval, reducing burstiness, and preferring the cleanest thermal and humidity envelope available. "
            "The paper reports that write I/O is affected more strongly than read I/O under temperature and humidity stress."
        )
    raise ValueError(f"Unhandled env prescriptive query: {query_id}")


def _render_env_whatif(query_id: str, context: Dict[str, Any]) -> str:
    if query_id == "whatif_env_reduce_humidity":
        return (
            "If humidity is reduced toward room conditions, tail latency should improve and bandwidth should recover relative to high-humidity operation. "
            "The temperature-humidity paper found decreasing humidity improves SSD performance and that high humidity can create both runtime and post-impact degradation."
        )
    if query_id == "whatif_env_raise_temperature":
        return (
            "If temperature rises into the 50-60°C regime within specification, TLC devices may show higher average bandwidth, but operators should expect a tradeoff with higher endurance and post-impact risk. "
            "This is a performance optimization only if the environment remains controlled and the write path is watched carefully."
        )
    if query_id == "whatif_env_restore_room_conditions":
        return (
            "Returning to room conditions should remove the runtime component of the stress, but not all damage vanishes immediately. "
            "High humidity and some high-temperature exposures can leave post-impact, so partial recovery is expected rather than a guarantee of full recovery."
        )
    if query_id == "whatif_env_reverse_direction":
        return (
            "Reversing the environmental direction should reverse the performance trend in broad terms: more humidity generally worsens latency, lower humidity improves it, and temperature changes must be interpreted together with flash type and post-impact risk. "
            "The result should therefore flip direction but not necessarily with symmetric magnitude."
        )
    raise ValueError(f"Unhandled env what-if query: {query_id}")


def _render_vibration_descriptive(query_id: str, context: Dict[str, Any]) -> str:
    if query_id == "desc_vib_short_term_tail":
        return (
            "This vibration condition should be described as a tail-latency stressor even when mean performance looks normal. "
            "The paper reports short-term vibration can degrade read/write tail latency by more than 10% and up to roughly 30% in the worst case."
        )
    if query_id == "desc_vib_orientation_effect":
        return (
            "Orientation matters for vibration. Perpendicular vibration is typically worse than parallel vibration in the short term, and the difference between orientations can be as large as about 30% for tail-latency degradation."
        )
    if query_id == "desc_vib_long_term_degradation":
        return (
            "If vibration persists, the effect should be described as a long-term reliability and performance problem, not just a transient slowdown. "
            "The paper reports long-term tail-latency degradation up to about 45%, mean-bandwidth loss up to about 10%, and the possibility of silent or transient failures."
        )
    if query_id == "desc_vib_hidden_smart_gap":
        return (
            "This condition should be described as partially hidden from traditional health tools. "
            "The vibration paper found that mean performance and SMART-style counters may not show a clear signature even while tail latency degrades and fail-slow behavior develops."
        )
    raise ValueError(f"Unhandled vibration descriptive query: {query_id}")


def _render_vibration_prescriptive(query_id: str, context: Dict[str, Any]) -> str:
    if query_id == "pres_vib_isolate_latency_sensitive":
        return (
            "Move latency-sensitive or write-heavy workloads away from the vibrating device or isolate the drive from the source. "
            "Use tail-latency behavior, not just average throughput, to decide whether the workload still meets its service target."
        )
    if query_id == "pres_vib_orientation_mounting":
        return (
            "Reduce the harmful vibration orientation through better mounting, placement, and isolation, with particular attention to avoiding the most harmful perpendicular exposure. "
            "Treat physical layout as an operational control, not just a facilities issue."
        )
    if query_id == "pres_vib_monitor_tail_not_mean":
        return (
            "Monitor tail latency, bandwidth variability, and abrupt stop faults, because average latency or SMART counters alone can hide the impact of vibration. "
            "Escalate when tail metrics drift persistently even if the mean looks stable."
        )
    if query_id == "pres_vib_prepare_failslow_recovery":
        return (
            "Prepare for fail-slow and transient-fault behavior by scheduling rapid migration, replacement, and post-restart validation. "
            "A restart may temporarily restore functionality, but the paper reports continued degradation and eventual premature failure under long-term vibration."
        )
    raise ValueError(f"Unhandled vibration prescriptive query: {query_id}")


def _render_vibration_whatif(query_id: str, context: Dict[str, Any]) -> str:
    if query_id == "whatif_vib_remove_vibration":
        return (
            "If the vibration source is removed, tail latency should move back toward the no-vibration baseline and bandwidth should stabilize. "
            "The recovery is strongest for purely runtime effects, although prior exposure can still leave some post-effect."
        )
    if query_id == "whatif_vib_parallel_vs_perpendicular":
        return (
            "Switching between parallel and perpendicular vibration changes the severity profile. Perpendicular vibration is usually worse in the short term, while parallel vibration can still become harmful over long exposures."
        )
    if query_id == "whatif_vib_prolong_exposure":
        return (
            "If the exposure continues for much longer, expect substantially worse tail latency, measurable bandwidth loss, and a higher chance of silent or transient failures. "
            "The vibration paper shows long-term exposure is much more damaging than a brief episode."
        )
    if query_id == "whatif_vib_restart_after_transient_fault":
        return (
            "A restart after a transient fault may restore functionality temporarily, but it should not be treated as a full fix. "
            "The paper reports that SSDs can resume operation after restart while still showing degraded performance and recurring failure behavior."
        )
    raise ValueError(f"Unhandled vibration what-if query: {query_id}")


def _smart_interface_sentence(context: Dict[str, Any]) -> str:
    return (
        "Interface-path evidence should be interpreted conservatively: if UCRC or related link errors are active, the operator should suspect cabling, connector, controller-path, or host-link instability rather than assume a pure media failure."
    )


def _smart_endurance_sentence(context: Dict[str, Any]) -> str:
    return (
        "Endurance should be described through wear-oriented counters and age together. Rising media-wear or wear-leveling pressure lowers confidence even before the drive becomes an immediate hard failure."
    )


def _smart_write_pressure_sentence(context: Dict[str, Any]) -> str:
    return (
        "Write pressure matters because heavier write activity accelerates wear-related stress and is associated with higher overall failure exposure than read-dominant behavior in field studies."
    )


def _smart_age_sentence(context: Dict[str, Any]) -> str:
    age_years = context.get("age_years")
    if age_years is not None:
        return (
            f"Age should be treated as a real risk amplifier: this sample already reflects about {_fmt_num(age_years)} years of power-on time, and older SSDs are more likely to exhibit short-interval correlated failures."
        )
    return (
        "Age should be treated as a real risk amplifier when available, because older SSDs are more likely to exhibit short-interval correlated failures."
    )


def _smart_warning_sentence(context: Dict[str, Any]) -> str:
    risk = context.get("smart_risk")
    if risk == "elevated":
        return "This warning signal is actionable rather than merely weakly suggestive, because multiple media or interface indicators are already active."
    if risk == "moderate":
        return "This warning signal is meaningful but not definitive: it merits enhanced monitoring and contextual review rather than blind panic."
    return "This warning signal is weak or incomplete and should be treated as context for monitoring rather than proof of imminent failure."


def _smart_monitor_replace_recommendation(context: Dict[str, Any]) -> str:
    return (
        f"Prioritize {_monitoring_urgency(context)} monitoring of the SMART window, verify whether the abnormal counters keep rising, and migrate or replace the drive if media or interface errors continue to accumulate. "
        "Use the device-specific SMART evidence for local triage, but keep the explanation concise and evidence-backed."
    )


def _smart_reduce_correlated_risk_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Reduce correlated-failure exposure by avoiding dense placement of same-model SSDs in the same node or rack, using stronger fault tolerance where possible, and preferring eager recovery over lazy recovery once failures are detected. "
        "Do not rely on SMART counters alone for node- or rack-level correlation decisions."
    )


def _smart_write_heavy_controls_recommendation(context: Dict[str, Any]) -> str:
    profile = context.get("workload_profile", "mixed or unclear")
    if profile == "write-dominant":
        opening = "Because the workload is already write-dominant, rebalance or throttle writes"
    elif profile == "read-dominant":
        opening = "If the workload drifts away from its current read-dominant profile, rebalance or throttle writes early"
    else:
        opening = "If write pressure remains meaningful, rebalance or throttle writes"
    return (
        f"{opening}, move latency-sensitive traffic away from stressed drives, and shorten inspection intervals for media and interface counters. "
        "The Alibaba study indicates write-dominant workloads create higher overall SSD failure exposure than read-dominant ones."
    )


def _smart_static_policy_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Treat static device descriptors as operating context and align controller behavior conservatively: monitor garbage-collection pressure, wear leveling, refresh behavior, and read/write amplification, and tighten lifecycle thresholds for older or higher-density flash. "
        f"Current static context: {_smart_static_clause(context)}."
    )


def _smart_interface_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Inspect the interface path first: validate cabling, connectors, controller logs, and host-link integrity, then separate path issues from true media deterioration before escalating to replacement."
    )


def _smart_endurance_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Add endurance guardrails: reduce unnecessary writes, watch wear-related counters more frequently, and lower the replacement threshold once wear acceleration becomes clear."
    )


def _smart_migration_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Plan preemptive migration while the drive is still serving requests, validate that the warning is persistent across windows, and avoid waiting for a stronger but more disruptive failure signal."
    )


def _smart_placement_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Review rack and node placement for same-model concentration, especially when the fleet shows repeated issues on similar SSDs. Diversity in placement and model mix reduces correlated-failure exposure."
    )


def _smart_spare_capacity_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Increase operational safety margin by keeping spare capacity and recovery headroom available, and pair deteriorating drives with stronger redundancy or faster recovery policies instead of minimal protection."
    )


def _smart_trend_monitoring_recommendation(context: Dict[str, Any]) -> str:
    return (
        f"Monitor trends rather than snapshots. Track slopes, repeated outliers, and sustained counter growth, and escalate from {_monitoring_urgency(context)} monitoring if the same abnormal pattern persists across consecutive windows."
    )


def _smart_failslow_triage_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Use service-level symptoms together with SMART. If latency inflation or intermittent faults appear before a strong SMART spike, treat the case as fail-slow risk and prepare migration, replacement, and closer runtime monitoring."
    )


def _smart_counter_priority_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Prioritize counters tied to media errors, reallocation, interface reliability, age, and wear. Read and write volume metrics should be used as supporting workload context rather than the only trigger."
    )


def _smart_reduce_writes_analysis(context: Dict[str, Any]) -> str:
    profile = context.get("workload_profile", "mixed or unclear")
    if profile == "write-dominant":
        baseline = "the current write-dominant profile"
    elif profile == "read-dominant":
        baseline = "the current read-dominant profile"
    else:
        baseline = "the current mixed or unclear profile"
    return (
        f"If the workload becomes less write-intensive than {baseline}, wear-related SMART pressure and overall failure exposure should decrease. "
        "The improvement is strongest when the starting point is clearly write-dominant."
    )


def _smart_ageing_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the drive accumulates more age and rated-life usage, correlated-failure risk should rise and short-interval failures become more plausible. "
        "Field evidence shows older SSDs are more likely to exhibit intra-node and intra-rack failures within short time intervals."
    )


def _smart_spread_model_analysis(context: Dict[str, Any]) -> str:
    return (
        "If same-model SSDs are spread across nodes and racks instead of densely colocated, correlated-failure exposure should decrease. "
        "Field studies observed that placing many SSDs from the same drive model in the same scope raises intra-node and intra-rack failure risk."
    )


def _smart_eager_recovery_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the system switches from lazy recovery to eager recovery, reliability should improve under correlated failures because additional failures are more likely after initial ones. "
        "Correlated-failure studies explicitly concluded that lazy recovery is less suitable than eager recovery for this pattern."
    )


def _smart_fix_interface_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the interface or host-link instability is removed, the warning should weaken substantially for path-driven anomalies, although any genuine media wear signal would remain."
    )


def _smart_refresh_policy_analysis(context: Dict[str, Any]) -> str:
    return (
        "If refresh and background maintenance become more conservative, near-term reliability confidence should improve modestly, especially for aging or wear-stressed drives, but this does not replace the need to manage workload and placement risk."
    )


def _smart_preemptive_replacement_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the operator replaces the drive proactively, local risk drops immediately and correlated exposure can also fall if the troubled drive is part of a concentrated same-model cluster."
    )


def _smart_health_sentence(context: Dict[str, Any]) -> str:
    signals = context.get("smart_signals") or ["the available SMART window does not expose a single dominant media-error spike"]
    return (
        f"This window should be described as {_smart_risk_phrase(context)}. "
        f"Key evidence includes {signals[0]}"
        + (f", {signals[1]}" if len(signals) > 1 else "")
        + (f", and {signals[2]}" if len(signals) > 2 else "")
        + "."
    )


def _smart_workload_sentence(context: Dict[str, Any]) -> str:
    profile = context.get("workload_profile", "mixed or unclear")
    if profile == "write-dominant":
        return (
            "The workload context is write-dominant, which the Alibaba study associates with higher overall SSD failure exposure than read-dominant workloads."
        )
    if profile == "read-dominant":
        return (
            "The workload context is read-dominant, which generally corresponds to lower overall SSD failure exposure than write-dominant workloads in the Alibaba study."
        )
    return (
        "The workload contribution should be described cautiously because the read/write dominance is mixed or unclear in the available evidence."
    )


def _smart_limit_sentence() -> str:
    return (
        "SMART evidence alone should not be over-interpreted for node- or rack-level correlated failures because the Alibaba study found SMART attributes are limited indicators for detecting intra-node and intra-rack failure correlation without placement and workload context."
    )


def _smart_static_sentence(context: Dict[str, Any]) -> str:
    return f"Static context such as {_smart_static_clause(context)} should be treated as supporting operational context rather than proof of imminent failure."


def _smart_static_clause(context: Dict[str, Any]) -> str:
    parts = []
    if context.get("flash_type"):
        parts.append(f"flash type {context['flash_type']}")
    if context.get("algorithms"):
        parts.append("controller policies " + ", ".join(context["algorithms"]))
    return " and ".join(parts) if parts else "the available flash/controller descriptors"


def _smart_risk_phrase(context: Dict[str, Any]) -> str:
    risk = str(context.get("smart_risk") or "").strip()
    if risk == "elevated":
        return "an elevated reliability risk pattern"
    if risk == "moderate":
        return "a moderate reliability risk pattern"
    if risk:
        return f"a {risk} reliability picture"
    return "an uncertain reliability picture"


def _smart_trend_sentence(context: Dict[str, Any]) -> str:
    return (
        "A temporal SMART trend is more informative than a single snapshot because slopes, repeated outliers, and sustained counter growth show deterioration more reliably than one isolated reading."
    )


def _smart_failslow_sentence(context: Dict[str, Any]) -> str:
    return (
        "SMART should not be treated as a perfect fail-slow detector. An SSD can show latency inflation, intermittent faults, or service degradation before a dramatic SMART collapse appears."
    )


def _smart_feature_priority_sentence(context: Dict[str, Any]) -> str:
    return (
        "The highest-priority SMART evidence usually comes from media errors, reallocation, interface reliability, age, and wear. Read and write volume counters provide supporting workload context rather than a standalone trigger."
    )


def _smart_relieve_write_pressure_analysis(context: Dict[str, Any]) -> str:
    return (
        "If write pressure is relieved but not eliminated, the SMART trajectory should look healthier over time and aging pressure should ease, even if the drive does not become fully read-dominant."
    )


def _smart_model_mix_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the fleet mixes more SSD models instead of concentrating one model heavily, common-mode failure exposure should decrease because fewer identical drives share the same failure mode in the same scope."
    )


def _smart_error_counters_rise_analysis(context: Dict[str, Any]) -> str:
    urgency = _monitoring_urgency(context)
    return (
        f"If media or interface error counters continue to rise across future windows, the current warning should be treated as strengthening evidence rather than a transient anomaly, and {urgency} monitoring should escalate toward migration or replacement."
    )


def _smart_failslow_no_hard_failure_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the SSD enters fail-slow behavior without an immediate hard fail-stop, user-visible latency risk should increase first, and the operator should not wait for a dramatic SMART collapse before acting."
    )


def _workload_app_interpretation_sentence(context: Dict[str, Any]) -> str:
    workload = context.get("workload")
    profile = context.get("workload_profile", "mixed or unclear")
    if workload:
        return (
            f"The workload tag {workload} should be used as an operational proxy for a {profile} service pattern so the SMART evidence is interpreted with the correct read/write context."
        )
    return (
        f"The workload should be interpreted as {profile}, using application or read/write evidence as a proxy so the SMART window is not judged in isolation."
    )


def _workload_write_service_risk_sentence(context: Dict[str, Any]) -> str:
    profile = context.get("workload_profile", "mixed or unclear")
    if profile == "write-dominant":
        return (
            "Because this sample already looks write-dominant, it should be treated as higher service risk than a comparable read-oriented workload. Extra write stress accelerates wear and makes abnormal SMART trends more operationally important."
        )
    if profile == "read-dominant":
        return (
            "A more write-heavy service would be riskier than this read-dominant profile because sustained writes accelerate wear and make abnormal SMART trends more consequential."
        )
    return (
        "A sustained write-heavy service should be treated as riskier than a read-oriented one because extra write stress accelerates wear and makes future SMART anomalies more consequential."
    )


def _workload_rebalance_recommendation(context: Dict[str, Any]) -> str:
    workload = context.get("workload")
    profile = context.get("workload_profile", "mixed or unclear")
    if workload:
        return (
            f"Use workload class {workload} to rebalance risk: move the most write-intensive or latency-sensitive traffic away first, especially when the current profile looks {profile}."
        )
    return (
        f"Use the workload profile to rebalance risk: move the most write-intensive or latency-sensitive traffic away first when the current service pattern looks {profile}."
    )


def _workload_reduce_bursts_recommendation(context: Dict[str, Any]) -> str:
    profile = context.get("workload_profile", "mixed or unclear")
    return (
        f"Reduce burstiness and write amplification where possible. Smooth the write path, avoid unnecessary background work, and shorten inspection intervals when the workload remains {profile}."
    )


def _workload_switch_write_heavy_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the workload becomes more write-heavy than it is today, endurance pressure and the operational importance of abnormal SMART trends should increase."
    )


def _workload_shift_read_heavy_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the workload shifts toward a more read-dominant pattern, the SSD should see lower wear pressure and a safer reliability outlook than under an equally intense write-heavy service."
    )


def _smart_env_compound_sentence(context: Dict[str, Any]) -> str:
    return (
        f"When SMART warnings appear together with {_env_condition_clause(context)}, the case should be described as compound risk. The SMART view captures device health, while the environment adds extra performance and reliability pressure."
    )


def _workload_env_compound_sentence(context: Dict[str, Any]) -> str:
    return (
        f"When a {context.get('workload_profile', 'mixed or unclear')} workload runs during {_env_condition_clause(context)}, the combined penalty should be treated as worse than reading either signal alone."
    )


def _smart_env_compound_recommendation(context: Dict[str, Any]) -> str:
    return (
        f"Treat the case as compound risk. Tighten monitoring, reduce workload pressure, avoid prolonged exposure to {_env_condition_clause(context)}, and lower the threshold for migration or replacement compared with a SMART-only case."
    )


def _workload_env_schedule_recommendation(context: Dict[str, Any]) -> str:
    return (
        f"Protect demanding workloads by moving them away from intervals with {_env_condition_clause(context)}, reducing burstiness, and preferring the cleanest operating envelope available."
    )


def _smart_env_hot_humid_analysis(context: Dict[str, Any]) -> str:
    return (
        f"If SMART warnings persist while {_env_condition_clause(context)} becomes harsher, the case should be treated as a compounded reliability and performance problem with lower tolerance for waiting or partial mitigation."
    )


def _workload_env_burst_analysis(context: Dict[str, Any]) -> str:
    return (
        f"If a write-heavy burst arrives during {_env_condition_clause(context)}, service risk should rise more sharply than under either condition alone because the write path is the more vulnerable side."
    )


def _flash_tradeoff_sentence(context: Dict[str, Any]) -> str:
    flash = _flash_label(context)
    return (
        f"Flash type changes the reliability tradeoff. {flash} should be interpreted as a capacity-versus-endurance choice rather than a neutral detail, with denser media generally carrying tighter endurance and error margins."
    )


def _flash_temperature_sensitivity_sentence(context: Dict[str, Any]) -> str:
    flash = _flash_label(context)
    return (
        f"Flash type also changes environmental sensitivity. {flash} should be evaluated with flash-specific temperature and humidity guardrails instead of assuming every SSD reacts the same way."
    )


def _flash_lifecycle_recommendation(context: Dict[str, Any]) -> str:
    flash = _flash_label(context)
    return (
        f"Adjust lifecycle and replacement policy for {flash}. Higher-density flash should be managed with tighter endurance guardrails and more conservative replacement thresholds than more robust low-density media."
    )


def _flash_guardrails_recommendation(context: Dict[str, Any]) -> str:
    flash = _flash_label(context)
    return (
        f"Use flash-specific environmental guardrails for {flash}. Apply tighter temperature and humidity discipline for more sensitive media instead of assuming one shared margin is safe for every SSD."
    )


def _flash_switch_to_mlc_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the same workload or environment is experienced by a more robust flash type such as MLC instead of TLC-like media, endurance risk should ease, although temperature and humidity sensitivity will still depend on the exact condition."
    )


def _flash_move_to_qlc_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the SSD moves to a denser flash type such as QLC, capacity efficiency improves but the reliability margin becomes tighter, so endurance and error sensitivity should be treated more conservatively."
    )


def _alg_gc_tradeoff_sentence(context: Dict[str, Any]) -> str:
    return (
        f"Garbage-collection policy should be described as a performance-lifetime tradeoff. The current context ({_algorithms_label(context)}) may relieve space pressure and some latency spikes, but aggressive background work can also increase write amplification."
    )


def _alg_wear_leveling_tradeoff_sentence(context: Dict[str, Any]) -> str:
    return (
        f"Wear-leveling policy should be described as a balancing problem. In the current controller context ({_algorithms_label(context)}), stronger leveling can improve lifetime fairness across blocks but may also introduce extra movement and latency overhead."
    )


def _alg_policy_review_recommendation(context: Dict[str, Any]) -> str:
    return (
        f"Review controller policy whenever latency spikes, write amplification, or wear imbalance become visible. In the current setup ({_algorithms_label(context)}), garbage collection, refresh, mapping, and retry settings should be revisited as a group rather than tuned in isolation."
    )


def _alg_gc_wl_balance_recommendation(context: Dict[str, Any]) -> str:
    return (
        f"Balance garbage collection and wear leveling conservatively in the current controller setup ({_algorithms_label(context)}). Avoid policy extremes that suppress one problem only by creating another, such as lowering latency at the cost of excessive write amplification."
    )


def _alg_more_aggressive_gc_analysis(context: Dict[str, Any]) -> str:
    return (
        "If garbage collection becomes more aggressive, some latency spikes and space-pressure problems may ease, but write amplification and endurance cost can rise."
    )


def _alg_static_to_hybrid_wl_analysis(context: Dict[str, Any]) -> str:
    return (
        "If wear leveling moves toward a better balanced hybrid policy, lifetime fairness should improve, but the operator should still watch for extra background movement and latency overhead."
    )


def _alg_page_to_block_analysis(context: Dict[str, Any]) -> str:
    return (
        "If mapping becomes coarser, controller memory cost may fall, but the system should expect less flexibility and potentially worse fine-grained performance behavior under mixed workloads."
    )


def _flash_label(context: Dict[str, Any]) -> str:
    return str(context.get("flash_type") or "the current flash type")


def _algorithms_label(context: Dict[str, Any]) -> str:
    algs = context.get("algorithms") or []
    return ", ".join(algs) if algs else "the current controller-policy mix"


def _env_condition_clause(context: Dict[str, Any]) -> str:
    factor = str(context.get("factor_text") or "").strip()
    if factor:
        return factor.replace("+", " and ")
    condition = _text(context.get("condition"))
    if condition:
        return condition
    return "the current environmental stress"


def _env_runtime_sentence(context: Dict[str, Any]) -> str:
    factor = context.get("factor_text") or "environmental"
    metric = context.get("metric") or "performance"
    return (
        f"This {factor} condition should be described as a {metric} stressor: { _metric_effect_phrase(context) }. "
        "The explanation should stay grounded in the published experimental evidence rather than claim a direct SMART-derived failure event."
    )


def _env_postimpact_sentence(context: Dict[str, Any]) -> str:
    factor = context.get("factor_text") or ""
    if "humidity" in factor:
        return (
            "High-humidity exposure can leave post-impact after the environment returns to nominal conditions, with particularly strong tail-latency degradation reported for TLC devices."
        )
    if "temperature" in factor:
        return (
            "Temperature exposure is not only a runtime effect: the paper reports negative post-impact from high-temperature exposure mainly for TLC SSDs, even after returning to room conditions."
        )
    return (
        "Abrupt temperature and humidity changes can leave post-impact, so the reference answer should distinguish immediate runtime effects from lingering degradation after the exposure ends."
    )


def _env_tradeoff_sentence(context: Dict[str, Any]) -> str:
    return (
        "The main tradeoff is that some high-temperature points can improve average bandwidth, while humidity increases and post-impact episodes can hurt tail latency and make the write path more fragile than the read path."
    )


def _env_device_sensitivity_sentence(context: Dict[str, Any]) -> str:
    device = context.get("device_type") or "the tested SSD types"
    return (
        f"Sensitivity should be described in device terms: {device} should be interpreted through the paper's finding that TLC devices react strongly to temperature, MLC tail latency can degrade under humidity increases, and write I/O is usually more affected than read I/O."
    )


def _env_high_temp_sentence(context: Dict[str, Any]) -> str:
    return (
        "A high-temperature operating point can improve average bandwidth for some TLC cases, but that should be interpreted as a conditional throughput gain rather than a blanket statement that heat is harmless."
    )


def _env_high_humidity_sentence(context: Dict[str, Any]) -> str:
    return (
        "High humidity is the clearest environmental penalty in this study: it worsens tail latency, can leave post-impact after the exposure ends, and is more dangerous for the write path than for the read path."
    )


def _env_write_read_sentence(context: Dict[str, Any]) -> str:
    return (
        "This condition should be described as more damaging to write I/O than to read I/O, because the study reports stronger write-path degradation under temperature and humidity stress."
    )


def _env_abrupt_change_sentence(context: Dict[str, Any]) -> str:
    return (
        "Abrupt environmental transitions should be treated as first-class events: even when a transition temporarily improves one metric, it still changes SSD behavior enough to justify explicit operational attention."
    )


def _env_recovery_sentence(context: Dict[str, Any]) -> str:
    return (
        "Recovery after the exposure is partial rather than guaranteed. Runtime stress can fade quickly, but humidity- and heat-related post-impact may remain visible in latency, bandwidth, or later failures."
    )


def _env_humidity_mitigation_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Move the operating point back toward room conditions when humidity is high, avoid prolonged exposure, and treat high-humidity episodes as performance-risk events rather than harmless noise. "
        "Humidity reduction is the safest first mitigation because the experiments report better performance when humidity decreases."
    )


def _env_thermal_window_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Keep the device inside vendor limits, use any high-temperature operating point only deliberately and temporarily, and avoid trading a short-term bandwidth gain for longer-term retention or post-impact risk. "
        "When possible, schedule demanding work after the environment stabilizes."
    )


def _env_post_impact_monitoring_recommendation(context: Dict[str, Any]) -> str:
    return (
        "After the exposure ends, continue monitoring tail latency, bandwidth, media-wear indicators, and unexpected failures because the experiments report post-impact after high humidity and some high-temperature exposures. "
        "Do not assume the drive is healthy simply because the chamber returned to nominal conditions."
    )


def _env_write_heavy_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Protect write-heavy or latency-critical workloads by moving them away from the stressed interval, reducing burstiness, and preferring the cleanest thermal and humidity envelope available. "
        "The experiments show that write I/O is affected more strongly than read I/O under environmental stress."
    )


def _env_room_recovery_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Return the system gradually toward room conditions, then validate that both runtime metrics and post-exposure metrics have stabilized before restoring full trust in the SSD."
    )


def _env_high_temp_guardrail_recommendation(context: Dict[str, Any]) -> str:
    return (
        "If a high-temperature point is used for throughput reasons, treat it as a controlled exception: cap duration, watch tail latency and wear indicators, and avoid pairing it with high humidity or write-heavy spikes."
    )


def _env_humidity_limit_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Limit the duration and blast radius of high-humidity exposure by compartmentalizing affected equipment, rerouting sensitive work, and prioritizing environmental correction over tuning the SSD in place."
    )


def _env_flash_type_policy_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Adjust policy by flash type: treat TLC devices as especially temperature-sensitive and MLC devices as particularly vulnerable to humidity-driven tail-latency penalties, rather than using a single uniform rule."
    )


def _env_post_event_validation_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Before fully trusting the SSD again, validate tail-latency behavior, bandwidth, error counters, and any evidence of post-impact. A nominal room-condition reading alone is not enough."
    )


def _env_reduce_humidity_analysis(context: Dict[str, Any]) -> str:
    return (
        "If humidity is reduced toward room conditions, tail latency should improve and bandwidth should recover relative to high-humidity operation. "
        "The temperature-humidity study found decreasing humidity improves SSD performance and that high humidity can create both runtime and post-impact degradation."
    )


def _env_raise_temperature_analysis(context: Dict[str, Any]) -> str:
    return (
        "If temperature rises into the 50-60°C regime within specification, TLC devices may show higher average bandwidth, but operators should expect a tradeoff with higher endurance and post-impact risk. "
        "This is a performance optimization only if the environment remains controlled and the write path is watched carefully."
    )


def _env_restore_room_analysis(context: Dict[str, Any]) -> str:
    return (
        "Returning to room conditions should remove the runtime component of the stress, but not all damage vanishes immediately. "
        "High humidity and some high-temperature exposures can leave post-impact, so partial recovery is expected rather than a guarantee of full recovery."
    )


def _env_reverse_direction_analysis(context: Dict[str, Any]) -> str:
    return (
        "Reversing the environmental direction should reverse the performance trend in broad terms: more humidity generally worsens latency, lower humidity improves it, and temperature changes must be interpreted together with flash type and post-impact risk. "
        "The result should therefore flip direction but not necessarily with symmetric magnitude."
    )


def _env_increase_humidity_analysis(context: Dict[str, Any]) -> str:
    return (
        "If humidity increases further, expect worse tail latency, a stronger chance of post-impact, and higher concern for write-path degradation."
    )


def _env_drop_temperature_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the device is cooled while humidity stays comparable, any high-temperature throughput gain should weaken, and the result should be interpreted primarily through the tradeoff between immediate bandwidth and longer-term stress."
    )


def _env_extend_exposure_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the same environmental exposure lasts much longer, operator confidence should decrease because post-impact and eventual failure behavior become more plausible than in the short runtime experiments."
    )


def _env_switch_flash_type_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the same condition is applied to a different flash type, sensitivity should shift. TLC tends to be more temperature-sensitive, while MLC can show especially visible humidity-driven tail-latency penalties."
    )


def _env_shift_write_read_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the workload becomes less write-heavy under the same environmental condition, the performance penalty should ease because the experiments indicate write I/O is more affected than read I/O."
    )


def _vib_short_term_sentence(context: Dict[str, Any]) -> str:
    orientation = _vib_orientation_kind(context)
    orientation_clause = f" under {orientation} vibration" if orientation else ""
    return (
        f"This should be described as a short-term tail-latency stressor{orientation_clause}: read/write tail latency can degrade by {_range_phrase(context.get('change_min'), context.get('change_max'))}, "
        "and worst-case short episodes reported in the study reach roughly 30%."
    )


def _vib_orientation_sentence(context: Dict[str, Any]) -> str:
    orientation = _vib_orientation_kind(context)
    if orientation == "parallel":
        return (
            "This row already reflects a parallel-vibration case. Parallel vibration is still harmful, especially as exposure length grows, but perpendicular vibration is usually the more damaging short-term direction."
        )
    if orientation == "perpendicular":
        return (
            "This row should be read as a high-risk orientation case. Perpendicular vibration is typically worse than parallel vibration, and the short-term gap between the two directions can approach about 30%."
        )
    return (
        "Orientation matters for vibration. Perpendicular vibration is typically worse than parallel vibration in the short term, and the performance gap between orientations can reach about 30% for tail-latency degradation."
    )


def _vib_long_term_sentence(context: Dict[str, Any]) -> str:
    return (
        "If vibration persists, the condition should be described as a long-term reliability and performance problem, not just a transient slowdown. The study reports tail-latency degradation above 30% and up to about 45%, mean-bandwidth loss around 10% in some cases, and the possibility of fail-slow, silent, or transient failures."
    )


def _vib_hidden_gap_sentence(context: Dict[str, Any]) -> str:
    return (
        "This condition should be described as partially hidden from traditional health tools. Mean performance and SMART-style indicators may look only mildly affected even while tail latency degrades significantly and fail-slow behavior develops."
    )


def _vib_mean_vs_tail_sentence(context: Dict[str, Any]) -> str:
    return (
        "The important distinction is that tail latency worsens much more clearly than mean behavior. Average throughput or average latency can understate service risk, so tail metrics are the better early signal."
    )


def _vib_vendor_variability_sentence(context: Dict[str, Any]) -> str:
    device_type = _text(context.get("device_type")) or "the tested SSDs"
    return (
        f"Vendor and implementation sensitivity should be treated as real. {device_type} should be interpreted against the study's observation that some vendors show small impact in one regime while others, especially on the write path, degrade much more noticeably under the same vibration class."
    )


def _vib_post_effect_sentence(context: Dict[str, Any]) -> str:
    return (
        "The impact can persist after the strongest vibration interval ends. Recovery toward baseline is possible, but long or repeated exposure can leave post-effects in bandwidth stability, tail latency, or recurrent transient faults."
    )


def _vib_transient_fault_sentence(context: Dict[str, Any]) -> str:
    return (
        "The failure pattern should be described as fail-slow or transient rather than purely crash-stop. A device may stop serving correctly, recover after restart, and still continue with degraded performance or recurring faults."
    )


def _vib_bandwidth_sentence(context: Dict[str, Any]) -> str:
    return (
        "As exposure length grows, mean bandwidth degradation becomes easier to see. The paper reports about a 10% drop in one long-term parallel-vibration regime and more than 20% loss in a case involving transient failures between quiet phases."
    )


def _vib_isolate_recommendation(context: Dict[str, Any]) -> str:
    workload = _text(context.get("workload")) or "latency-sensitive"
    return (
        f"Move {workload} or other latency-sensitive traffic away from the vibrating device when possible, or isolate the drive from the source. Use tail-latency behavior rather than average throughput alone to decide whether the SSD still meets its service target."
    )


def _vib_orientation_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Reduce the harmful vibration direction through better mounting, placement, and isolation, with special attention to avoiding perpendicular exposure. Treat chassis orientation and mechanical layout as operational reliability controls, not just facilities details."
    )


def _vib_monitor_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Monitor tail latency, bandwidth variability, restart events, and abrupt stop faults, because mean performance or SMART counters alone can hide the impact of vibration. Escalate when tail metrics drift persistently even if the average still looks acceptable."
    )


def _vib_failslow_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Prepare for fail-slow and transient-fault behavior by scheduling rapid migration, replacement, and post-restart validation. A restart may temporarily restore functionality, but it should not be treated as a durable fix for a drive that remains under harmful vibration."
    )


def _vib_exposure_duration_recommendation(context: Dict[str, Any]) -> str:
    return (
        "If the source cannot be removed immediately, shorten effective exposure duration: reduce time spent under the strongest vibration, insert quiet intervals when operationally possible, and move the most sensitive workloads out first."
    )


def _vib_restart_validation_recommendation(context: Dict[str, Any]) -> str:
    return (
        "After restart, validate tail-latency recovery, bandwidth stability, filesystem or controller errors, and whether the fault recurs under comparable load. Functional recovery alone is not enough to declare the SSD healthy."
    )


def _vib_vendor_qualification_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Qualification should include model-specific vibration tests across orientation, exposure length, and read/write mix instead of relying only on no-vibration averages. The paper shows vendor sensitivity is uneven enough that platform qualification needs per-model evidence."
    )


def _vib_workload_derating_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Derate the workload while the condition persists: move latency-sensitive or write-heavy traffic away first, reduce burstiness, and lower service expectations on the affected SSD instead of assuming the nominal SLO still holds."
    )


def _vib_source_audit_recommendation(context: Dict[str, Any]) -> str:
    return (
        "Audit likely vibration sources systematically: inspect mounting hardware, nearby fans, pumps, rotating equipment, enclosure resonance, and neighboring drive activity. The goal is to localize the mechanical path so mitigation can target the source instead of only the symptom."
    )


def _vib_remove_vibration_analysis(context: Dict[str, Any]) -> str:
    exposure = _vib_exposure_kind(context)
    recovery_clause = (
        " Recovery should be strongest if the issue was a short-term runtime effect."
        if exposure == "short-term"
        else " Recovery may be only partial if the device already experienced long-term or repeated exposure."
    )
    return (
        "If the vibration source is removed, tail latency should move back toward the no-vibration baseline and bandwidth should stabilize." + recovery_clause
    )


def _vib_orientation_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the same vibration exposure switches between parallel and perpendicular orientation, severity should shift with it. Perpendicular vibration is usually worse in the short term, while parallel vibration can still become harmful once exposure is prolonged."
    )


def _vib_prolong_exposure_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the exposure persists much longer, expect substantially worse tail latency, clearer mean-bandwidth loss, and a higher chance of fail-slow, silent, or transient-fault behavior. Long-term vibration is materially more dangerous than a brief episode."
    )


def _vib_restart_analysis(context: Dict[str, Any]) -> str:
    return (
        "A restart after a transient vibration-related fault may restore functionality in the near term, but it should not be treated as a full fix. Longer-term performance risk remains elevated, and recurring faults are still plausible if the vibration source persists."
    )


def _vib_lower_intensity_analysis(context: Dict[str, Any]) -> str:
    return (
        "If vibration intensity is reduced but not eliminated, the severity profile should improve but not disappear. Tail-latency risk should ease first, while long-term reliability concerns remain until the mechanical stress is fully controlled."
    )


def _vib_shift_parallel_analysis(context: Dict[str, Any]) -> str:
    return (
        "If the dominant vibration direction moves closer to the less harmful parallel case, performance risk should decrease, especially for short-term tail latency. The SSD still remains vulnerable if exposure continues for a long duration."
    )


def _vib_move_latency_workload_analysis(context: Dict[str, Any]) -> str:
    return (
        "If latency-sensitive or write-heavy traffic is moved away while vibration persists, user-visible service risk should drop immediately even though the SSD itself remains mechanically stressed. This is a mitigation of workload impact, not a cure for the device condition."
    )


def _vib_repeat_bursts_analysis(context: Dict[str, Any]) -> str:
    return (
        "Repeated short bursts should be treated as more dangerous than one isolated blip because they can create intermittent post-effects that are hard to catch with average metrics. Risk evolves toward recurring service instability rather than a single clean event."
    )


def _vib_no_action_analysis(context: Dict[str, Any]) -> str:
    return (
        "If no mitigation is applied while vibration continues, near-term tail-latency degradation should remain visible and the longer-term outlook should worsen toward bandwidth loss, fail-slow behavior, and premature failure."
    )


def _vib_exposure_kind(context: Dict[str, Any]) -> Optional[str]:
    exposure = _lower_text(context.get("exposure")) or _lower_text(context.get("condition")) or ""
    if "long-term" in exposure or "long term" in exposure:
        return "long-term"
    if "short-term" in exposure or "short term" in exposure:
        return "short-term"
    return None


def _vib_orientation_kind(context: Dict[str, Any]) -> Optional[str]:
    orientation = _lower_text(context.get("vibration_orientation")) or _lower_text(context.get("condition")) or ""
    if "perpendicular" in orientation and "parallel" not in orientation:
        return "perpendicular"
    if "parallel" in orientation and "perpendicular" not in orientation:
        return "parallel"
    return None


def _metric_effect_phrase(context: Dict[str, Any]) -> str:
    metric = context.get("metric") or "performance"
    direction = context.get("direction") or "changes"
    rng = _range_phrase(context.get("change_min"), context.get("change_max"))
    if "tail_latency" in metric:
        if direction == "improves":
            return f"tail latency is expected to improve, potentially by {rng}"
        if direction == "degrades":
            return f"tail latency is expected to worsen by {rng}"
    if "bandwidth" in metric:
        if direction == "improves":
            return f"bandwidth is expected to increase by {rng}"
        if direction == "degrades":
            return f"bandwidth is expected to drop by {rng}"
    return f"{metric.replace('_', ' ')} is expected to {direction} by about {rng}"


def _range_phrase(vmin: Optional[float], vmax: Optional[float]) -> str:
    if vmin is None and vmax is None:
        return "a meaningful amount"
    if vmin is not None and vmax is not None:
        if abs(vmin - vmax) < 1e-9:
            return f"about {_fmt_num(vmax)}%"
        return f"about {_fmt_num(vmin)}-{_fmt_num(vmax)}%"
    if vmax is not None:
        return f"up to {_fmt_num(vmax)}%"
    return f"more than {_fmt_num(vmin)}%"


def _range_midpoint(vmin: Optional[float], vmax: Optional[float]) -> Optional[float]:
    if vmin is None and vmax is None:
        return None
    if vmin is None:
        return float(vmax)
    if vmax is None:
        return float(vmin)
    return (float(vmin) + float(vmax)) / 2.0


def _top_smart_signals(
    smart_by_attr: Dict[str, Dict[str, Any]],
    age_years: Optional[float],
    workload_profile: str,
) -> List[str]:
    signals: List[str] = []
    for attr in ("r_187", "r_5", "r_199", "r_233", "r_173", "r_177"):
        frame = smart_by_attr.get(attr)
        if not frame:
            continue
        if _nonzero(frame.get("max")) or _nonzero(frame.get("slope")) or int(frame.get("outliers") or 0) > 0:
            label = SMART_LABELS.get(attr, attr)
            signals.append(f"{label} shows activity or upward pressure")
    if "r_241" in smart_by_attr and "r_242" in smart_by_attr:
        w = smart_by_attr["r_241"].get("median")
        r = smart_by_attr["r_242"].get("median")
        if _nonzero(w) or _nonzero(r):
            if _to_float(w) is not None and _to_float(r) is not None and float(w) > float(r):
                signals.append("blocks written exceed blocks read, indicating write pressure")
    if age_years is not None and age_years >= 3.0:
        signals.append(f"power-on age is already about {_fmt_num(age_years)} years")
    if workload_profile == "write-dominant":
        signals.append("the workload profile is write-dominant")
    return signals[:4]


def _smart_risk_label(
    smart_by_attr: Dict[str, Dict[str, Any]],
    age_years: Optional[float],
    workload_profile: str,
) -> str:
    for attr in ("r_187", "r_5", "r_199"):
        frame = smart_by_attr.get(attr)
        if not frame:
            continue
        if _nonzero(frame.get("max")) or _nonzero(frame.get("slope")) or int(frame.get("outliers") or 0) > 0:
            return "elevated"
    if (age_years is not None and age_years >= 3.0) or workload_profile == "write-dominant":
        return "moderate"
    return "low-to-moderate with uncertainty"


def _monitoring_urgency(context: Dict[str, Any]) -> str:
    risk = context.get("smart_risk")
    if risk == "elevated":
        return "high-priority"
    if risk == "moderate":
        return "enhanced"
    return "routine but evidence-backed"


def _infer_workload_profile(
    app_tag: Optional[str],
    smart_by_attr: Dict[str, Dict[str, Any]],
    workload_text: Optional[str] = None,
) -> str:
    if app_tag:
        tag = app_tag.strip().upper()
        if tag in WRITE_DOMINANT_APPS:
            return "write-dominant"
        if tag in READ_DOMINANT_APPS:
            return "read-dominant"
    if workload_text:
        text = workload_text.strip().lower()
        if "write-dominant" in text:
            return "write-dominant"
        if "read-dominant" in text:
            return "read-dominant"
        if "mixed" in text or "unclear" in text:
            return "mixed or unclear"
    written = _to_float((smart_by_attr.get("r_241") or {}).get("median"))
    read = _to_float((smart_by_attr.get("r_242") or {}).get("median"))
    if written is not None and read is not None:
        if written > read * 1.1:
            return "write-dominant"
        if read > written * 1.1:
            return "read-dominant"
    return "mixed or unclear"


def _infer_age_years(smart_by_attr: Dict[str, Dict[str, Any]]) -> Optional[float]:
    power_on = _to_float((smart_by_attr.get("r_9") or {}).get("median"))
    if power_on is None:
        return None
    return power_on / 24.0 / 365.0


def _infer_flash_type(row: Dict[str, Any], device_type: Optional[str]) -> Optional[str]:
    explicit = _text(row.get("flash_type") or row.get("ft") or row.get("FlashType"))
    if explicit:
        return explicit
    upper = (device_type or "").upper()
    for token in ("SLC", "MLC", "TLC", "QLC", "PLC"):
        if token in upper:
            return token
    return None


def _infer_algorithms(row: Dict[str, Any]) -> List[str]:
    raw = (
        row.get("algorithms")
        or row.get("policies")
        or row.get("controller_policies")
        or row.get("policy_string")
    )
    text = _text(raw)
    parts: List[str] = [part.strip() for part in text.split(";") if part.strip()] if text else []

    derived = []
    for key, prefix in (
        ("gc_algo", "gc"),
        ("wear_leveling", "wl"),
        ("ftl_mapping", "ftl"),
        ("refresh_policy", "refresh"),
        ("ecc_scheme", "ecc"),
        ("read_retry", "retry"),
        ("overprovision_pct", "op"),
    ):
        value = _text(row.get(key))
        if not value:
            continue
        suffix = f"{value}%" if key == "overprovision_pct" else value
        derived.append(f"{prefix}={suffix}")

    for item in derived:
        if item not in parts:
            parts.append(item)
    return parts


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _lower_text(value: Any) -> Optional[str]:
    text = _text(value)
    return text.lower() if text else None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _nonzero(value: Any) -> bool:
    num = _to_float(value)
    return num is not None and abs(num) > 1e-9


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")
