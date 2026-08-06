from __future__ import annotations

import re
import unicodedata


KATAKANA_START = ord("ァ")
KATAKANA_END = ord("ヶ")
HIRAGANA_START = ord("ぁ")


def normalize_surface(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def katakana_to_hiragana(value: str) -> str:
    result: list[str] = []
    for char in value:
        if char == "ヵ":
            result.append("か")
            continue
        if char == "ヶ":
            result.append("け")
            continue
        codepoint = ord(char)
        if KATAKANA_START <= codepoint <= KATAKANA_END:
            result.append(chr(codepoint - KATAKANA_START + HIRAGANA_START))
        else:
            result.append(char)
    return "".join(result)


def normalize_reading(value: str) -> str:
    normalized = katakana_to_hiragana(unicodedata.normalize("NFKC", value))
    return re.sub(r"[\s・･]", "", normalized).strip()


def is_valid_reading(value: str) -> bool:
    if not value or len(value) < 2 or len(value) > 60:
        return False
    return re.fullmatch(r"[ぁ-ゖー]+", value) is not None


def is_meaningful_surface(value: str) -> bool:
    if not value or len(value) > 80:
        return False
    if value in {"以下に掲載がない場合", "一円", "番地がくる場合"}:
        return False
    return any("一" <= char <= "鿿" for char in value)
