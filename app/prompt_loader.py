"""プロンプトテンプレートの読み込みとプレースホルダ置換"""

from __future__ import annotations

import os
from functools import lru_cache
from string import Formatter
from typing import Any


class _SafeFormatter(Formatter):
    """欠落キーはそのまま残す Formatter"""

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)


_formatter = _SafeFormatter()


@lru_cache(maxsize=128)
def _read_prompt_file(relative_path: str) -> str:
    if not os.path.exists(relative_path):
        raise FileNotFoundError(f"プロンプトファイルが見つかりません: {relative_path}")
    with open(relative_path, "r", encoding="utf-8") as prompt_file:
        return prompt_file.read()


def load_prompt(relative_path: str, **placeholders: Any) -> str:
    """相対パスのテンプレートを読み込み、プレースホルダを置換する"""
    template = _read_prompt_file(relative_path)
    if not placeholders:
        return template
    return _formatter.format(template, **placeholders)


def clear_prompt_cache() -> None:
    """テンプレートキャッシュをクリアする（主にテスト用）"""
    _read_prompt_file.cache_clear()
