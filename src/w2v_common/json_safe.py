"""JSON safety helpers shared by every w2v service.

``json.dumps`` defaults to ``allow_nan=True``, which serialises a non-finite
float to the bare tokens ``NaN`` / ``Infinity`` / ``-Infinity``. Those are NOT
valid JSON per RFC 8259, and Postgres rejects them outright::

    invalid input syntax for type json ... Token "-Infinity" is invalid

For a JSONB column that means the row never writes and the surrounding
transaction aborts — a failure that surfaces far from its cause (a
divide-by-zero ratio, a silent audio track measured by ``loudnorm`` as
``-inf``) and typically looks like a mysterious retry storm.

The fix is to coerce non-finite floats to JSON ``null`` before dumping.
Deliberately NOT ``allow_nan=False``: that only trades a database error for a
``ValueError`` in the caller, which is no better for anyone downstream.
"""

from __future__ import annotations

import json
import math

__all__ = ["finite_json_safe", "json_serializer"]


def finite_json_safe(obj: object) -> object:
    """Recursively map non-finite floats (NaN/Infinity) to ``None`` (JSON null).

    Walks ``dict`` / ``list`` / ``tuple`` containers. Finite values (ints,
    finite floats, strings, bools, ``None``) pass through unchanged — this
    reformats nothing.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: finite_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json_safe(item) for item in obj]
    return obj


def json_serializer(obj: object) -> str:
    """Serialise ``obj`` to JSON that Postgres will accept.

    Wired into every engine as SQLAlchemy's ``json_serializer`` so that a
    non-finite float anywhere inside a ``JSON``/``JSONB`` value becomes
    ``null`` instead of an invalid ``-Infinity``/``NaN`` token.
    """
    return json.dumps(finite_json_safe(obj))
