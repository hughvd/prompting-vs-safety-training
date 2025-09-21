from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(output_dir: str | Path) -> Dict[str, Path]:
    """Ensure the standard output directory structure exists."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    subdirs = {
        "base": base,
        "results": base / "results",
        "figures": base / "figures",
        "logs": base / "logs",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_spec_texts(directory: str | Path, entries: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """Load spec text files into memory keyed by spec id."""
    base = Path(directory)
    specs: Dict[str, str] = {}
    for spec in entries:
        spec_id = spec["id"]
        filename = spec["filename"]
        text = (base / filename).read_text(encoding="utf-8").strip()
        specs[spec_id] = text
    return specs


def append_jsonl(records: Iterable[Dict[str, Any]], destination: Path) -> None:
    """Append multiple JSON objects to a JSONL file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_env_file(path: str | Path = ".env", override: bool = False) -> Dict[str, str]:
    """Load environment variables from a .env-style file."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: Dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded
