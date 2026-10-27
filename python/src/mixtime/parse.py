"""Parsing helpers for Gregorian time strings and Python date objects."""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

_EPOCH_DATE = _dt.date(1970, 1, 1)
_EPOCH_DATETIME = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}



def parse_year(data: Any) -> np.ndarray:
    return _parse_each(data, _parse_year_one)


def parse_quarter(data: Any) -> np.ndarray:
    return _parse_each(data, _parse_quarter_one)


def parse_month(data: Any) -> np.ndarray:
    return _parse_each(data, _parse_month_one)


def parse_date(data: Any, *, tz: str | None = None) -> np.ndarray:
    return _parse_each(data, lambda value: _parse_date_one(value, tz=tz))


def parse_week(data: Any, *, tz: str | None = None) -> np.ndarray:
    return _parse_each(data, lambda value: _parse_week_one(value, tz=tz))


def parse_datetime(data: Any, *, tz: str | None = None) -> np.ndarray:
    return _parse_each(data, lambda value: _parse_datetime_one(value, tz=tz))


def parse_time(data: Any, *, tz: str | None = None) -> np.ndarray:
    return _parse_each(data, lambda value: _parse_time_one(value, tz=tz))


def date_to_days(value: Any, *, tz: str | None = None) -> float:
    if _is_missing(value):
        return float("nan")
    value = _coerce_temporal(value, tz=tz)
    if isinstance(value, _dt.datetime):
        value = _local_date(value, tz=tz)
    return float((value - _EPOCH_DATE).days)


def datetime_to_seconds(value: Any, *, tz: str | None = None) -> float:
    if _is_missing(value):
        return float("nan")
    value = _coerce_temporal(value, tz=tz)
    if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        value = _dt.datetime.combine(value, _dt.time())
    if value.tzinfo is None:
        value = value.replace(tzinfo=_zone(tz))
    return float(value.timestamp())


def _parse_each(data: Any, parser) -> np.ndarray:
    values = _as_sequence(data)
    out: list[float] = []
    for value in values:
        if _is_missing(value):
            out.append(float("nan"))
            continue
        out.append(float(parser(value)))
    return np.asarray(out, dtype=np.float64)


def _as_sequence(data: Any) -> list[Any]:
    if isinstance(data, (str, bytes)):
        return [data]
    if isinstance(data, (list, tuple, np.ndarray)):
        return list(data)
    if _is_temporal_scalar(data):
        return [data]
    return [data]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    if value is np.nan:
        return True
    if isinstance(value, np.datetime64) and np.isnat(value):
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, str) and value.strip().upper() in {"", "NA", "NAN", "NONE"}:
        return True
    return False


def _parse_year_one(value: Any) -> int:
    if _is_temporal_scalar(value):
        dt = _coerce_temporal(value)
        return dt.year - 1970
    if isinstance(value, (int, np.integer)):
        return int(value) - 1970
    text = str(value).strip()
    match = re.fullmatch(r"(?P<year>[+-]?\d{1,6})", text)
    if match:
        return int(match.group("year")) - 1970
    # Accept a full date string for convenience, even though R's year()
    # parser is deliberately narrower.
    parsed = _try_parse_date(text)
    if parsed is not None:
        return parsed.year - 1970
    raise ValueError(f"Could not parse year from {value!r}")


def _parse_quarter_one(value: Any) -> int:
    if _is_temporal_scalar(value):
        dt = _coerce_temporal(value)
        quarter = (dt.month - 1) // 3 + 1
        return (dt.year - 1970) * 4 + quarter - 1
    text = str(value).strip()

    match = re.fullmatch(
        r"(?P<year>[+-]?\d{1,6})\s*[-/ ]?\s*[Qq](?P<q>[1-4])", text
    )
    if match:
        return _quarter_value(int(match.group("year")), int(match.group("q")))

    match = re.fullmatch(
        r"[Qq](?P<q>[1-4])\s*[-/ ]?\s*(?P<year>[+-]?\d{1,6})", text
    )
    if match:
        return _quarter_value(int(match.group("year")), int(match.group("q")))

    match = re.fullmatch(
        r"(?P<year>[+-]?\d{1,6})\s*[-/ ]\s*(?P<month>\d{1,2})", text
    )
    if match:
        month = int(match.group("month"))
        if 1 <= month <= 12:
            return _quarter_value(int(match.group("year")), (month - 1) // 3 + 1)

    match = re.fullmatch(
        r"(?P<year>[+-]?\d{1,6})\s*[-/ ]?\s*(?P<month>[A-Za-z]{3,9})", text
    )
    if match:
        month = _month_number(match.group("month"))
        if month is not None:
            return _quarter_value(int(match.group("year")), (month - 1) // 3 + 1)

    raise ValueError(f"Could not parse quarter from {value!r}")


def _parse_month_one(value: Any) -> int:
    if _is_temporal_scalar(value):
        dt = _coerce_temporal(value)
        return (dt.year - 1970) * 12 + dt.month - 1
    text = str(value).strip()

    match = re.fullmatch(
        r"(?P<year>[+-]?\d{1,6})\s*[-/ .]?\s*(?P<month>\d{1,2})", text
    )
    if match:
        month = int(match.group("month"))
        if 1 <= month <= 12:
            return (int(match.group("year")) - 1970) * 12 + month - 1

    match = re.fullmatch(
        r"(?P<year>[+-]?\d{1,6})\s*[-/ .]?\s*(?P<month>[A-Za-z]{3,9})", text
    )
    if match:
        month = _month_number(match.group("month"))
        if month is not None:
            return (int(match.group("year")) - 1970) * 12 + month - 1

    match = re.fullmatch(
        r"(?P<month>[A-Za-z]{3,9})\.?\s*[-/ ,]?\s*(?P<year>[+-]?\d{1,6})", text
    )
    if match:
        month = _month_number(match.group("month"))
        if month is not None:
            return (int(match.group("year")) - 1970) * 12 + month - 1

    match = re.fullmatch(
        r"(?P<month>\d{1,2})\s*[-/ .]\s*(?P<year>[+-]?\d{1,6})", text
    )
    if match:
        month = int(match.group("month"))
        if 1 <= month <= 12:
            return (int(match.group("year")) - 1970) * 12 + month - 1

    # Accept a full date string and keep its month.
    parsed = _try_parse_date(text)
    if parsed is not None:
        return (parsed.year - 1970) * 12 + parsed.month - 1

    raise ValueError(f"Could not parse month from {value!r}")


def _parse_date_one(value: Any, *, tz: str | None) -> int:
    if _is_temporal_scalar(value):
        dt = _coerce_temporal(value, tz=tz)
        if isinstance(dt, _dt.datetime):
            dt = _local_date(dt, tz=tz)
        return (dt - _EPOCH_DATE).days

    text = str(value).strip()
    parsed_dt = _try_parse_datetime(text)
    if parsed_dt is not None:
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=_zone(tz))
        return (_local_date(parsed_dt, tz=tz) - _EPOCH_DATE).days
    parsed = _try_parse_date(text)
    if parsed is None:
        raise ValueError(f"Could not parse date from {value!r}")
    return (parsed - _EPOCH_DATE).days


def _parse_week_one(value: Any, *, tz: str | None) -> int:
    if _is_temporal_scalar(value):
        dt = _coerce_temporal(value, tz=tz)
        if isinstance(dt, _dt.datetime):
            dt = _local_date(dt, tz=tz)
        days = (dt - _EPOCH_DATE).days
        return _day_to_week(days)

    text = str(value).strip()
    match = re.fullmatch(
        r"(?P<year>[+-]?\d{1,6})\s*[-/ ]?[Ww](?P<week>\d{1,2})", text
    )
    if match:
        year = int(match.group("year"))
        week = int(match.group("week"))
        # ISO week 53 is valid only in some years; fromisocalendar raises
        # ValueError for invalid combinations.
        thursday = _dt.date.fromisocalendar(year, week, 4)
        return (thursday - _EPOCH_DATE).days // 7

    parsed_dt = _try_parse_datetime(text)
    if parsed_dt is not None:
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=_zone(tz))
        return _day_to_week((_local_date(parsed_dt, tz=tz) - _EPOCH_DATE).days)
    parsed = _try_parse_date(text)
    if parsed is not None:
        return _day_to_week((parsed - _EPOCH_DATE).days)
    raise ValueError(f"Could not parse ISO week from {value!r}")


def _day_to_week(days: int) -> int:
    # Week chronons are 7-day blocks anchored on Thursday, matching the ISO
    # week's Thursday and R mixtime's week values.
    return (days + 3) // 7


def _parse_datetime_one(value: Any, *, tz: str | None) -> float:
    if _is_temporal_scalar(value):
        return datetime_to_seconds(value, tz=tz)

    text = str(value).strip()
    parsed = _try_parse_datetime(text)
    if parsed is None:
        parsed_date = _try_parse_date(text)
        if parsed_date is not None:
            parsed = _dt.datetime.combine(parsed_date, _dt.time())
        else:
            raise ValueError(f"Could not parse datetime from {value!r}")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_zone(tz))
    return float(parsed.timestamp())


def _parse_time_one(value: Any, *, tz: str | None) -> float:
    if _is_temporal_scalar(value):
        return datetime_to_seconds(value, tz=tz)

    text = str(value).strip()
    parsed_dt = _try_parse_datetime(text)
    if parsed_dt is not None:
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=_zone(tz))
        return float(parsed_dt.timestamp())

    parsed_date = _try_parse_date(text)
    if parsed_date is not None:
        parsed_dt = _dt.datetime.combine(parsed_date, _dt.time()).replace(
            tzinfo=_zone(tz)
        )
        return float(parsed_dt.timestamp())

    match = re.fullmatch(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.(?P<frac>\d{1,9}))?)?"
        r"\s*(?P<ampm>[AaPp][Mm])?",
        text,
    )
    if not match:
        raise ValueError(f"Could not parse time from {value!r}")
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second") or 0)
    frac = match.group("frac") or ""
    microsecond = int((frac + "000000")[:6]) if frac else 0
    ampm = match.group("ampm")
    if ampm:
        if not 1 <= hour <= 12:
            raise ValueError(f"Invalid 12-hour time: {value!r}")
        hour = hour % 12 + (12 if ampm.lower() == "pm" else 0)
    parsed = _dt.datetime(1970, 1, 1, hour, minute, second, microsecond, tzinfo=_zone(tz))
    return float(parsed.timestamp())


def _try_parse_date(text: str) -> _dt.date | None:
    text = text.strip()
    patterns = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y%m%d",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%B %d %Y",
    )
    for pattern in patterns:
        try:
            return _dt.datetime.strptime(text, pattern).date()
        except ValueError:
            pass

    # ISO datetime strings also carry a date.
    parsed_dt = _try_parse_datetime(text)
    if parsed_dt is not None:
        return parsed_dt.date()
    return None


def _try_parse_datetime(text: str) -> _dt.datetime | None:
    text = text.strip()
    if not text:
        return None

    iso = text.replace(" ", "T", 1)
    try:
        return _dt.datetime.fromisoformat(iso)
    except ValueError:
        pass

    patterns = (
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d %b %Y %H:%M:%S",
        "%d %B %Y %H:%M:%S",
    )
    for pattern in patterns:
        try:
            return _dt.datetime.strptime(text, pattern)
        except ValueError:
            pass
    return None


def _month_number(name: str) -> int | None:
    return _MONTHS.get(name.strip().lower().rstrip("."))


def _quarter_value(year: int, quarter: int) -> int:
    return (year - 1970) * 4 + quarter - 1


def _is_temporal_scalar(value: Any) -> bool:
    if isinstance(value, (_dt.date, _dt.datetime, np.datetime64)):
        return True
    # pandas Timestamp / Period are optional; detect without importing pandas.
    module = type(value).__module__
    return module.startswith("pandas") and type(value).__name__ in {
        "Timestamp",
        "Period",
    }


def _coerce_temporal(value: Any, *, tz: str | None = None) -> _dt.date | _dt.datetime:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]").item()
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return value
    if type(value).__name__ == "Timestamp":
        return value.to_pydatetime()
    if type(value).__name__ == "Period":
        # A pandas Period is not a point; use its start timestamp.
        return value.start_time.to_pydatetime()
    raise TypeError(f"Unsupported temporal value: {type(value)!r}")


def _local_date(value: _dt.datetime, *, tz: str | None) -> _dt.date:
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(_zone(tz)).date()


def _zone(tz: str | None) -> _dt.tzinfo:
    if tz in (None, "", "UTC", "Etc/UTC"):
        return _dt.timezone.utc
    return ZoneInfo(tz)
