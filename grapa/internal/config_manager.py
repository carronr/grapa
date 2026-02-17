"""
Config loading and caching for grapa.

Provides a small config store with path resolution, cache, and type-aware value access.
Each Graph object may specify its own configuration file.
keys are lower-case.

@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, Optional, Union, List
import warnings

from grapa.shared.error_management import issue_warning
from grapa.shared.string_manipulations import strToVar

logger = logging.getLogger(__name__)

CONFIG_FILENAME_FALLBACK = "config.txt"
CONFIG_DEFAULT_VALUE = ""


@dataclass
class ConfigStore:
    """Config snapshot with filename metadata and type-aware lookup."""

    filename: str
    attributes: Dict[str, Any]

    def get(
        self,
        key: str,
        default: Any = CONFIG_DEFAULT_VALUE,  #  MetadataContainer.VALUE_DEFAULT
        astype: str = "auto",  # or "str"
    ) -> Any:
        """Return value for key with type handling (mirrors Graph.config)."""
        key = key.lower()
        if key in self.attributes:
            out = self.attributes.get(key)
        else:
            out = default
        if astype in [str, "str"]:
            return str(out)
        return out

    def all(self) -> Dict[str, Any]:
        """Return full attributes dict (copy)."""
        return dict(self.attributes)


class ConfigManager:
    """Resolves, loads, and caches config files and exposes typed access helpers.
    Each Graph may specify its own config file. If "auto", the previous one is reused.
    """

    def __init__(self, filename_fallback: str = CONFIG_FILENAME_FALLBACK):
        self.filename_default = None
        self.filename_fallback = filename_fallback
        self._cache: Dict[str, ConfigStore] = {}

    def _resolve_abs_filename(self, config: Union[None, str]) -> Optional[str]:
        """
        Resolve "auto" / relative paths to absolute.
        Returns None if config is None.
        """
        if config is None:
            return None
        if config == "auto":
            config = self.filename_default
            if config is None:
                config = self.filename_fallback
        if config is None:
            return None
        if not os.path.isabs(config):
            dir_name = os.path.dirname(os.path.dirname(__file__))
            config = os.path.join(dir_name, config)
        return config

    def _load(self, config: Union[None, str]) -> Optional[ConfigStore]:
        """
        Resolve and load config, returning cached ConfigStore if available.
        Handles IO errors; returns None if missing or invalid.
        """
        config_path = self._resolve_abs_filename(config)
        if config_path is None:
            return None
        if config_path in self._cache:
            return self._cache[config_path]

        content = _read_file_content(config_path)
        if content is None:
            return None
        attributes = _parse_config_content(content)
        store = ConfigStore(config_path, attributes)
        self._cache[config_path] = store
        if config == "auto" and self.filename_default is None:
            self.filename_default = config_path

        print(f"Config Manager: load file {config_path}.")
        return store

    def get(
        self,
        config: Union[None, str],  # e.g. "auto"
        key: str,
        default: Any = CONFIG_DEFAULT_VALUE,
        astype: str = "auto",
    ) -> Any:
        """Convenience: load + get, for Graph.config()."""
        store = self._load(config)
        if store is None:
            return default
        return store.get(key, default=default, astype=astype)

    def all(self, config: Union[None, str]) -> Dict[str, Any]:
        """Return {"attributes": ..., "filename": ...} shape"""
        store = self._load(config)
        if store is None:
            return {"attributes": {}, "filename": None}
        return {"attributes": store.all(), "filename": store.filename}


def _read_file_content(filename: str) -> Optional[list]:
    """Read config file content into a list of tab-split lines."""
    try:
        with open(filename, "r", encoding="utf-8", errors="backslashreplace") as file:
            return [line.strip(":\r\n").split("\t") for line in file]
    # except (UnicodeError, OSError):
    #     try:
    #         with open(
    #             filename, "r", encoding="ascii", errors="backslashreplace"
    #         ) as file:
    #             return [line.strip(":\r\n").split("\t") for line in file]
    except (UnicodeError, OSError) as e:
        msg = "ConfigManager: cannot open config file %s: %s."
        issue_warning(logger, msg, filename, e)
        return None


def _safe_str_to_var(value):
    if not isinstance(value, str):
        return value
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        out = strToVar(value)
        if any(w.category is SyntaxWarning for w in caught):
            return value
        return out


def _parse_config_content(content: List[str]) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}

    for line in content:
        if not line or line[0] == "":
            continue

        key = line[0].lower()
        if len(line) == 1:
            value = ""
        elif len(line) == 2:
            value = _safe_str_to_var(line[1])
        else:
            value = _safe_str_to_var("\t".join(line[1:]))
        attributes[key] = value
    return attributes
