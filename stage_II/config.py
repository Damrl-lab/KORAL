#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage II configuration and dataset-type defaults."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

@dataclass
class Stage2Config:
    # Paths (relative to repo root by default)
    repo_root: Path = Path(".")
    rule_base_path: Path = Path("rule_base.json")
    taxonomy_path: Path = Path("taxonomy.json")
    global_kg_ttl_path: Path = Path("global_knowledge_graph.ttl")

    # Output
    runs_dir: Path = Path("stage_II/runs")

    # LLM
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 900

    # Evaluation horizon (days) for label semantics (if you include it in your prompt)
    fail_horizon_days: int = 30

    # Default dataset-type → input CSV guess (override with --input_csv if you want)
    dataset_type_to_csv: Dict[str, Path] = None

    def __post_init__(self):
        if self.dataset_type_to_csv is None:
            self.dataset_type_to_csv = {
                # SMART-only
                "SMART_ALIBABA": Path("dataset/alibaba/test_data/smart.csv"),
                "SMART_GOOGLE": Path("dataset/google/test_data/smart.csv"),

                # Environmental-only
                "ENV": Path("dataset/env/env_effects.csv"),
                "ENV_FT": Path("dataset/env/env_ft.csv"),
                "ENV_AL": Path("dataset/env/env_al.csv"),

                # Paired/tripled modalities
                "SMART_WORKLOAD": Path("dataset/alibaba/test_data/smart_workload.csv"),
                "SMART_ENV": Path("dataset/alibaba/test_data/smart_env.csv"),
                "WORKLOAD_ENV": Path("dataset/fio_workload/test_data/workload_env.csv"),
                "SMART_ENV_WORKLOAD": Path("dataset/alibaba/test_data/smart_env_workload.csv"),

                # Augmentations
                "SMART_FT": Path("dataset/alibaba/test_data/smart_ft.csv"),
                "SMART_AL": Path("dataset/alibaba/test_data/smart_al.csv"),
                "SMART_FT_ENV_WORKLOAD": Path("dataset/alibaba/test_data/smart_ft_env_workload.csv"),
            }

def resolve_path(repo_root: Path, p: Path) -> Path:
    return (repo_root / p).resolve()
