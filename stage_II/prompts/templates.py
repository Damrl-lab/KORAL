#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt templates for Stage II tasks."""

from __future__ import annotations
from typing import Any, Dict, Optional


def _question_block(question: Optional[str]) -> str:
    if question is None:
        return ""
    text = str(question).strip()
    if not text:
        return ""
    return f"\nUser question: {text}\n"

def system_prompt(base_cot: str | None = None) -> str:
    # We keep it tight: ask for JSON output only.
    core = """You are KORAL Stage II for SSD operational analysis.
You MUST return a single valid JSON object and nothing else.
Do NOT wrap JSON in markdown fences.
If something is unknown, use null and explain in fields like notes or uncertainty.

Definitions:
- Supported claim: verifiable from provided Intermediate Representation (IR) or DataKG references.
- Evidence citations: use the provided reference IDs (IR:..., ENV:..., LIT:...).

Output MUST follow the schema requested in the user prompt.
"""
    if base_cot and base_cot.strip():
        return core + "\n\nAdditional guidance:\n" + base_cot.strip()
    return core

def predictive_user_prompt(sample: Dict[str, Any], question: Optional[str] = None) -> str:
    return f"""Task: Predictive analysis for one SSD window.
Input JSON (sample): {sample}
{_question_block(question)}

Produce JSON with:
{{
  "task": "predictive",
  "sample_id": <string>,
  "predicted_failure": <0|1|null>,
  "predicted_ttf_days": <number|null>,
  "predicted_tail_latency_ms": <number|null>,
  "predicted_tail_latency_pct": <number|null>,
  "tail_latency_unit": <ms|pct_change|null>,
  "rationale": <short text>,
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Keep rationale brief.
- If the sample includes a tail-latency percentage-change target, use `predicted_tail_latency_pct`.
- If the sample includes an absolute tail-latency target in milliseconds, use `predicted_tail_latency_ms`.
- Every atomic claim MUST cite at least one ref_id from IR/DataKG (e.g., "IR:AF_r_233") or literature (e.g., "LIT_3").
- Prefer IR/DataKG refs for device-specific statements.
"""

def descriptive_user_prompt(sample: Dict[str, Any], question: Optional[str] = None) -> str:
    return f"""Task: Descriptive analysis for one SSD window.
Input JSON (sample): {sample}
{_question_block(question)}

Produce JSON with:
{{
  "task": "descriptive",
  "sample_id": <string>,
  "summary": <text>,
  "key_risks": [<text>, ...],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Summary should describe health + performance signals seen in IR.
- Answer the user question directly when one is provided.
- Atomic claims must be grounded with refs.
"""

def prescriptive_user_prompt(sample: Dict[str, Any], question: Optional[str] = None) -> str:
    return f"""Task: Prescriptive analysis (actions) for one SSD window.
Input JSON (sample): {sample}
{_question_block(question)}

Produce JSON with:
{{
  "task": "prescriptive",
  "sample_id": <string>,
  "recommendations": [{{"action": <text>, "priority": <low|med|high>, "justification": <text>, "support": [<ref_id>, ...]}}],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Recommendations should be feasible operational actions (monitoring, scrubbing, throttling, migration, etc.).
- Answer the user question directly when one is provided.
- Cite IR for why you recommend each action; cite LIT for general justifications.
"""

def whatif_user_prompt(sample: Dict[str, Any], scenario: str, question: Optional[str] = None) -> str:
    return f"""Task: What-if analysis for one SSD window.
Input JSON (sample): {sample}
{_question_block(question)}

Counterfactual scenario: {scenario}

Produce JSON with:
{{
  "task": "whatif",
  "sample_id": <string>,
  "scenario": <text>,
  "analysis": <text>,
  "counterfactual_statements": [
    {{
      "statement": <text>,
      "variable": <text>,
      "delta": <number|null>,
      "effect": <text>,
      "effect_direction": <increase|decrease|unclear>,
      "evidence": [<ref_id>, ...]
    }}
  ]
}}

Rules:
- Each counterfactual statement MUST include evidence refs (IR/ENV/LIT).
- effect_direction should be consistent with evidence when possible.
- Answer the user question directly when one is provided.
"""
