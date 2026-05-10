#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flash controller algorithm/policy parsing."""

from __future__ import annotations
from typing import Any, Dict, List

POLICY_PART_KEYS = (
    "gc_algo",
    "wear_leveling",
    "ftl_mapping",
    "refresh_policy",
    "ecc_scheme",
    "read_retry",
    "overprovision_pct",
)


def _collect_policy_parts(row: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for key in POLICY_PART_KEYS:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if key == "overprovision_pct":
            parts.append(f"op={text}%")
        elif key == "gc_algo":
            parts.append(f"gc={text}")
        elif key == "wear_leveling":
            parts.append(f"wl={text}")
        elif key == "ftl_mapping":
            parts.append(f"ftl={text}")
        elif key == "refresh_policy":
            parts.append(f"refresh={text}")
        elif key == "ecc_scheme":
            parts.append(f"ecc={text}")
        elif key == "read_retry":
            parts.append(f"retry={text}")
    return parts

def build_algorithms_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    al = (
        row.get("algorithms")
        or row.get("policies")
        or row.get("controller_policies")
        or row.get("policy_string")
    )
    if al is None:
        policy_parts = _collect_policy_parts(row)
        if not policy_parts:
            return {}
        return {"algorithms": {"id": "AL_1", "policies": policy_parts}}
    if isinstance(al, list):
        policies = [str(x).strip() for x in al if str(x).strip()]
    else:
        s = str(al).strip()
        if not s:
            policy_parts = _collect_policy_parts(row)
            if not policy_parts:
                return {}
            return {"algorithms": {"id": "AL_1", "policies": policy_parts}}
        # allow semicolon separated
        policies = [p.strip() for p in s.split(";") if p.strip()]
    policies.extend(p for p in _collect_policy_parts(row) if p not in policies)
    if not policies:
        return {}
    return {"algorithms": {"id": "AL_1", "policies": policies}}
