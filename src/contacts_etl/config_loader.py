from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore[import-untyped]


@dataclass
class OutputsConfig:
    dir: Path


@dataclass
class NormalizationConfig:
    default_phone_country: str = "US"
    keep_generational_suffixes: Optional[list[str]] = None
    professional_suffixes: Optional[list[str]] = None
    name_prefixes: Optional[list[str]] = None


@dataclass
class DedupeConfig:
    enable_nickname_equivalence: bool = True
    first_name_similarity_threshold: float = 0.88
    merge_score_threshold: float = 1.2
    relaxed_merge_threshold: float = 0.6
    require_corroborator: bool = False


@dataclass
class ValidationConfig:
    email_dns_mx_check: bool = False


@dataclass
class TaggingConfig:
    prior_companies: list[str]
    prior_domains: list[str]
    local_cities: list[str]


@dataclass
class LoggingConfig:
    level: str = "WARNING"


@dataclass
class PipelineConfig:
    inputs: Dict[str, Optional[str]]
    outputs: OutputsConfig
    normalization: NormalizationConfig
    dedupe: DedupeConfig
    validation: ValidationConfig
    tagging: TaggingConfig
    logging: LoggingConfig


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    config_data = _load_yaml(getattr(args, "config", None))
    inputs = config_data.get("inputs", {})
    outputs_cfg = config_data.get("outputs", {})
    normalization_cfg = config_data.get("normalization", {})
    dedupe_cfg = config_data.get("dedupe", {})
    validation_cfg = config_data.get("validation", {})
    tagging_cfg = config_data.get("tagging", {})
    logging_cfg = config_data.get("logging", {})

    outputs_dir = Path(getattr(args, "out_dir", None) or outputs_cfg.get("dir") or os.getcwd())
    outputs = OutputsConfig(dir=outputs_dir)

    normalization = NormalizationConfig(
        default_phone_country=getattr(args, "default_phone_country", None)
        or normalization_cfg.get("default_phone_country", "US"),
        keep_generational_suffixes=getattr(args, "keep_generational_suffixes", None)
        or normalization_cfg.get("keep_generational_suffixes"),
        professional_suffixes=getattr(args, "professional_suffixes", None)
        or normalization_cfg.get("professional_suffixes"),
        name_prefixes=getattr(args, "name_prefixes", None)
        or normalization_cfg.get("name_prefixes"),
    )

    dedupe = DedupeConfig(
        enable_nickname_equivalence=(
            dedupe_cfg.get("enable_nickname_equivalence", True)
            if getattr(args, "enable_nickname_equivalence", None) is None
            else bool(getattr(args, "enable_nickname_equivalence"))
        ),
        first_name_similarity_threshold=getattr(args, "first_name_similarity_threshold", None)
        or dedupe_cfg.get("first_name_similarity_threshold", 0.88),
        merge_score_threshold=getattr(args, "merge_score_threshold", None)
        or dedupe_cfg.get("merge_score_threshold", 1.2),
        relaxed_merge_threshold=getattr(args, "relaxed_merge_threshold", None)
        or dedupe_cfg.get("relaxed_merge_threshold", 0.6),
        require_corroborator=getattr(args, "require_corroborator", None)
        or dedupe_cfg.get("require_corroborator", False),
    )

    validation = ValidationConfig(
        email_dns_mx_check=getattr(args, "email_dns_mx", None)
        or validation_cfg.get("email_dns_mx_check", False),
    )

    tagging = TaggingConfig(
        prior_companies=tagging_cfg.get("prior_companies", []),
        prior_domains=tagging_cfg.get("prior_domains", []),
        local_cities=tagging_cfg.get("local_cities", []),
    )

    arg_level = getattr(args, "log_level", None)
    effective_level = (arg_level or logging_cfg.get("level") or "WARNING").upper()
    logging_config = LoggingConfig(level=effective_level)

    resolved_inputs = {
        "linkedin_csv": getattr(args, "linkedin_csv", None) or inputs.get("linkedin_csv"),
        "gmail_csv": getattr(args, "gmail_csv", None) or inputs.get("gmail_csv"),
        "mac_vcf": getattr(args, "mac_vcf", None) or inputs.get("mac_vcf"),
        "contacts_csv": getattr(args, "contacts_csv", None),
        "lineage_csv": getattr(args, "lineage_csv", None),
    }

    return PipelineConfig(
        inputs=resolved_inputs,
        outputs=outputs,
        normalization=normalization,
        dedupe=dedupe,
        validation=validation,
        tagging=tagging,
        logging=logging_config,
    )
