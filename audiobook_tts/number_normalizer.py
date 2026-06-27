from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

from num2words import num2words


MONTHS_GENITIVE = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
MONTH_NUMBER = {name: index for index, name in enumerate(MONTHS_GENITIVE) if name}

HEADING_GENDERS = {
    "глава": "f",
    "часть": "f",
    "книга": "f",
    "том": "m",
    "раздел": "m",
}

HEADING_ARABIC_RE = re.compile(
    r"(?P<label>\b(?:глава|часть|книга|том|раздел))\s+"
    r"(?P<number>\d+)(?![\w-])",
    re.IGNORECASE,
)
HEADING_ROMAN_RE = re.compile(
    r"(?P<label>\b(?:глава|часть|книга|том|раздел))\s+"
    r"(?P<number>[IVXLCDM]+)\b",
    re.IGNORECASE,
)
ISO_DATE_RE = re.compile(
    r"(?<!\w)(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})(?!\w)"
)
DATE_RE = re.compile(
    r"(?<!\w)(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-]"
    r"(?P<year>\d{2,4})(?!\w)"
)
TEXT_DATE_RE = re.compile(
    r"(?<!\w)(?P<day>\d{1,2})\s+"
    r"(?P<month>января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря)\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(
    r"(?P<prefix>\b(?:в|во|к|ко|с|со|до|от|после)\s+)?"
    r"(?P<year>\d{3,4})\s*"
    r"(?P<word>году|годом|года|год|г\.)(?!\w)",
    re.IGNORECASE,
)
TIME_RE = re.compile(
    r"(?<!\d)(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)(?!\d)"
)
NUMBER_TOKEN = r"[+-]?(?:\d{1,3}(?:[\u00a0 ]\d{3})+|\d+)(?:[.,]\d{1,2})?"
CURRENCY_PREFIX_RE = re.compile(
    rf"(?P<currency>[$€₽])\s*(?P<amount>{NUMBER_TOKEN})(?![\w\d])"
)
CURRENCY_SUFFIX_RE = re.compile(
    rf"(?<![\w.,\d])(?P<amount>{NUMBER_TOKEN})\s*"
    r"(?P<currency>₽|\$|€|руб(?:\.|ль|ля|лей)?|доллар(?:а|ов)?|"
    r"евро|RUB|USD|EUR)(?!\w)",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(
    rf"(?<![\w.,\d])(?P<amount>{NUMBER_TOKEN})\s*%(?!\w)"
)
DECADE_RE = re.compile(
    r"(?<!\w)(?P<number>\d+)\s*[-‑–—]\s*(?P<suffix>е|х|ми)\s+"
    r"(?P<word>годы|годов|годах|годами)\b",
    re.IGNORECASE,
)
EXPLICIT_ORDINAL_RE = re.compile(
    r"(?<!\w)(?P<number>\d+)\s*[-‑–—]\s*"
    r"(?P<suffix>ый|ой|ая|яя|ое|ее|ые|ие|ого|его|ому|ему|ыми|ими|"
    r"ых|их|ую|юю|й|я|е|го|му|ми|м|ю|х)\b",
    re.IGNORECASE,
)
IDENTIFIER_CONTEXT_RE = re.compile(
    r"(?P<prefix>\b(?:позвонить|телефон|тел\.|код|индекс|номер)\s+|№\s*)"
    r"(?P<value>\+?\d(?:[\d() -]*\d)?)",
    re.IGNORECASE,
)
SEPARATED_NUMBER_RE = re.compile(r"(?<!\w)\+?\d(?:[\d() -]*\d)?(?!\w)")
LONG_IDENTIFIER_RE = re.compile(r"(?<!\w)\d{10,}(?!\w)")
RANGE_RE = re.compile(
    r"(?<!\w)(?P<start>[+-]?\d+(?:[.,]\d+)?)\s*[-–—]\s*"
    r"(?P<end>[+-]?\d+(?:[.,]\d+)?)(?!\w)"
)
REMAINING_NUMBER_RE = re.compile(
    r"(?<!\w)(?P<number>[+-]?\d+(?:[.,]\d+)?)(?!\w)"
)

ORDINAL_SUFFIXES: dict[str, dict[str, object]] = {
    "й": {"gender": "m"},
    "ый": {"gender": "m"},
    "ой": {"gender": "m"},
    "я": {"gender": "f"},
    "ая": {"gender": "f"},
    "яя": {"gender": "f"},
    "е": {"gender": "n"},
    "ое": {"gender": "n"},
    "ее": {"gender": "n"},
    "ые": {"plural": True},
    "ие": {"plural": True},
    "го": {"gender": "m", "case": "g"},
    "ого": {"gender": "m", "case": "g"},
    "его": {"gender": "m", "case": "g"},
    "му": {"gender": "m", "case": "d"},
    "ому": {"gender": "m", "case": "d"},
    "ему": {"gender": "m", "case": "d"},
    "м": {"gender": "m", "case": "p"},
    "ю": {"gender": "f", "case": "a"},
    "ую": {"gender": "f", "case": "a"},
    "юю": {"gender": "f", "case": "a"},
    "х": {"plural": True, "case": "g"},
    "ых": {"plural": True, "case": "g"},
    "их": {"plural": True, "case": "g"},
    "ми": {"plural": True, "case": "i"},
    "ыми": {"plural": True, "case": "i"},
    "ими": {"plural": True, "case": "i"},
}

CURRENCY_CODES = {
    "$": "USD",
    "USD": "USD",
    "ДОЛЛАР": "USD",
    "ДОЛЛАРА": "USD",
    "ДОЛЛАРОВ": "USD",
    "€": "EUR",
    "EUR": "EUR",
    "ЕВРО": "EUR",
    "₽": "RUB",
    "RUB": "RUB",
    "РУБ": "RUB",
    "РУБ.": "RUB",
    "РУБЛЬ": "RUB",
    "РУБЛЯ": "RUB",
    "РУБЛЕЙ": "RUB",
}

ZERO_MINOR_UNITS = {
    "RUB": re.compile(r",\s*ноль копеек$"),
    "USD": re.compile(r",\s*ноль центов$"),
    "EUR": re.compile(r",\s*ноль центов$"),
}


def normalize_numbers_for_tts(text: str) -> str:
    """Expand common Russian numeric notation into words for Silero TTS."""
    if not text or (
        not any(char.isdigit() for char in text) and not HEADING_ROMAN_RE.search(text)
    ):
        return text

    text = _canonicalize_decimal_digits(text)
    text = HEADING_ROMAN_RE.sub(_replace_roman_heading, text)
    text = HEADING_ARABIC_RE.sub(_replace_arabic_heading, text)
    text = ISO_DATE_RE.sub(_replace_iso_date, text)
    text = DATE_RE.sub(_replace_date, text)
    text = TEXT_DATE_RE.sub(_replace_text_date, text)
    text = YEAR_RE.sub(_replace_year, text)
    text = TIME_RE.sub(_replace_time, text)
    text = CURRENCY_PREFIX_RE.sub(_replace_currency, text)
    text = CURRENCY_SUFFIX_RE.sub(_replace_currency, text)
    text = PERCENT_RE.sub(_replace_percent, text)
    text = DECADE_RE.sub(_replace_decade, text)
    text = EXPLICIT_ORDINAL_RE.sub(_replace_explicit_ordinal, text)
    text = IDENTIFIER_CONTEXT_RE.sub(_replace_context_identifier, text)
    text = SEPARATED_NUMBER_RE.sub(_replace_separated_identifier, text)
    text = LONG_IDENTIFIER_RE.sub(lambda match: _digits_as_words(match.group(0)), text)
    text = RANGE_RE.sub(_replace_range, text)
    text = REMAINING_NUMBER_RE.sub(_replace_remaining_number, text)
    return re.sub(r"\d+", lambda match: _digits_as_words(match.group(0)), text)


def _canonicalize_decimal_digits(text: str) -> str:
    result: list[str] = []
    for char in text:
        if unicodedata.category(char) == "Nd":
            try:
                result.append(str(unicodedata.digit(char)))
            except (TypeError, ValueError):
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def _replace_arabic_heading(match: re.Match[str]) -> str:
    label = match.group("label")
    gender = HEADING_GENDERS[label.lower()]
    return f"{label} {_ordinal_words(int(match.group('number')), gender=gender)}"


def _replace_roman_heading(match: re.Match[str]) -> str:
    value = _roman_to_int(match.group("number"))
    if value is None:
        return match.group(0)
    label = match.group("label")
    return f"{label} {_ordinal_words(value, gender=HEADING_GENDERS[label.lower()])}"


def _replace_iso_date(match: re.Match[str]) -> str:
    return _date_words(
        int(match.group("day")),
        int(match.group("month")),
        int(match.group("year")),
        match.group(0),
    )


def _replace_date(match: re.Match[str]) -> str:
    year = int(match.group("year"))
    if year < 100:
        year += 2000 if year <= 69 else 1900
    return _date_words(
        int(match.group("day")),
        int(match.group("month")),
        year,
        match.group(0),
    )


def _date_words(day: int, month: int, year: int, fallback: str) -> str:
    try:
        date(year, month, day)
    except ValueError:
        return fallback
    return (
        f"{_ordinal_words(day, gender='n')} {MONTHS_GENITIVE[month]} "
        f"{_ordinal_words(year, gender='m', case='g')} года"
    )


def _replace_text_date(match: re.Match[str]) -> str:
    day = int(match.group("day"))
    month_text = match.group("month")
    month = MONTH_NUMBER[month_text.lower()]
    try:
        date(2000, month, day)
    except ValueError:
        return match.group(0)
    return f"{_ordinal_words(day, gender='n')} {month_text}"


def _replace_year(match: re.Match[str]) -> str:
    prefix = match.group("prefix") or ""
    word = match.group("word")
    word_lower = word.lower()
    prefix_lower = prefix.strip().lower()
    case = "n"
    spoken_word = word
    if word_lower == "года":
        case = "g"
    elif word_lower == "годом":
        case = "i"
    elif word_lower == "году":
        case = "p" if prefix_lower in {"в", "во"} else "d"
    elif word_lower == "г.":
        if prefix_lower in {"в", "во"}:
            case, spoken_word = "p", "году"
        elif prefix_lower in {"к", "ко"}:
            case, spoken_word = "d", "году"
        elif prefix_lower in {"с", "со", "до", "от", "после"}:
            case, spoken_word = "g", "года"
        else:
            spoken_word = "год"
        remainder = match.string[match.end() :].lstrip()
        if not remainder or remainder[0].isupper():
            spoken_word += "."
    words = _ordinal_words(int(match.group("year")), gender="m", case=case)
    return f"{prefix}{words} {spoken_word}"


def _replace_time(match: re.Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    result = f"{_number_words(str(hour))} {_plural_form(hour, ('час', 'часа', 'часов'))}"
    if minute:
        result += (
            f" {_number_words(str(minute))} "
            f"{_plural_form(minute, ('минута', 'минуты', 'минут'))}"
        )
    return result


def _replace_currency(match: re.Match[str]) -> str:
    raw_currency = match.group("currency")
    code = CURRENCY_CODES[raw_currency.upper()]
    return _currency_words(match.group("amount"), code)


def _currency_words(raw_amount: str, code: str) -> str:
    compact = raw_amount.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    try:
        amount = Decimal(compact)
        result = num2words(amount, lang="ru", to="currency", currency=code)
    except (InvalidOperation, OverflowError, TypeError, ValueError):
        return _number_words(compact)
    if amount == amount.to_integral_value():
        result = ZERO_MINOR_UNITS[code].sub("", result)
    return result


def _replace_percent(match: re.Match[str]) -> str:
    raw = match.group("amount")
    compact = raw.replace(" ", "").replace("\u00a0", "")
    words = _number_words(compact)
    if "." in compact or "," in compact:
        return f"{words} процента"
    value = abs(int(compact))
    return f"{words} {_plural_form(value, ('процент', 'процента', 'процентов'))}"


def _replace_decade(match: re.Match[str]) -> str:
    case = {"е": "n", "х": "g", "ми": "i"}[match.group("suffix").lower()]
    words = _ordinal_words(int(match.group("number")), plural=True, case=case)
    return f"{words} {match.group('word')}"


def _replace_explicit_ordinal(match: re.Match[str]) -> str:
    kwargs = ORDINAL_SUFFIXES[match.group("suffix").lower()]
    return _ordinal_words(int(match.group("number")), **kwargs)


def _replace_context_identifier(match: re.Match[str]) -> str:
    prefix = match.group("prefix")
    value = match.group("value")
    spoken_prefix = "номер " if prefix.strip() == "№" else prefix
    return f"{spoken_prefix}{_digits_as_words(value)}"


def _replace_separated_identifier(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = re.findall(r"\d", raw)
    groups = re.findall(r"\d+", raw)
    is_identifier = (
        raw.startswith("+")
        or "(" in raw
        or ")" in raw
        or len(groups) >= 3 and len(digits) >= 7
        or len(groups) == 2 and len(digits) >= 7 and not all(len(group) == 4 for group in groups)
    )
    return _digits_as_words(raw) if is_identifier else raw


def _replace_range(match: re.Match[str]) -> str:
    start = _number_words(match.group("start"), case="g")
    end = _number_words(match.group("end"), case="g")
    return f"от {start} до {end}"


def _replace_remaining_number(match: re.Match[str]) -> str:
    raw = match.group("number")
    unsigned = raw.lstrip("+-")
    integer_part = re.split(r"[.,]", unsigned, maxsplit=1)[0]
    if len(integer_part) > 1 and integer_part.startswith("0"):
        return _digits_as_words(raw)
    return _number_words(raw)


def _number_words(raw: str, *, case: str = "n") -> str:
    compact = raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    try:
        Decimal(compact)
        return num2words(compact, lang="ru", case=case)
    except (InvalidOperation, OverflowError, TypeError, ValueError):
        return _digits_as_words(raw)


def _ordinal_words(
    value: int,
    *,
    gender: str = "m",
    case: str = "n",
    plural: bool = False,
) -> str:
    try:
        return num2words(
            value,
            lang="ru",
            to="ordinal",
            gender=gender,
            case=case,
            plural=plural,
        )
    except (OverflowError, TypeError, ValueError):
        return _digits_as_words(str(value))


def _digits_as_words(raw: str) -> str:
    parts: list[str] = []
    if raw.strip().startswith("+"):
        parts.append("плюс")
    elif raw.strip().startswith("-"):
        parts.append("минус")
    for char in raw:
        if char.isdigit():
            parts.append(num2words(int(char), lang="ru"))
    return " ".join(parts)


def _plural_form(value: int, forms: tuple[str, str, str]) -> str:
    value = abs(value) % 100
    if 11 <= value <= 19:
        return forms[2]
    last = value % 10
    if last == 1:
        return forms[0]
    if 2 <= last <= 4:
        return forms[1]
    return forms[2]


def _roman_to_int(raw: str) -> int | None:
    roman = raw.upper()
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(roman):
        value = values[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    if not 1 <= total <= 3999 or _int_to_roman(total) != roman:
        return None
    return total


def _int_to_roman(value: int) -> str:
    parts: list[str] = []
    for number, token in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        count, value = divmod(value, number)
        parts.append(token * count)
    return "".join(parts)
