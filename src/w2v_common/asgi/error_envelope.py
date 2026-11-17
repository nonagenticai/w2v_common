"""Canonical error-envelope shape for FastAPI / Starlette services.

All w2v services emit a uniform error body so clients (FE, peer services,
external tenants) can branch on a single typed shape regardless of which
service produced the error::

    {
        "error": {
            "code": "HTTP_404",
            "message": "Run not found",
            "status": 404,
            "request_id": "...",        # mirrors X-Request-Id (generated if absent)
            "timestamp": "2026-05-25T12:34:56Z",
            "details": ...              # 422: ALWAYS a list of scrubbed
                                        # {loc, msg, type} field errors;
                                        # 500: debug-only exception text
        }
    }

This is the shape the fleet standardised on organically (it was hand-rolled,
identically, in every service's ``asgi.py``). It is lifted here verbatim so
each service deletes its copy and imports one implementation — the wire
format is unchanged, so no consumer (FE, peer service, external tenant) sees
a difference. ``code`` is ``HTTP_<status>`` for routed HTTPExceptions,
``VALIDATION_ERROR`` for request-validation failures, and
``INTERNAL_SERVER_ERROR`` for uncaught exceptions.

The helpers register the three exception handlers every service maps:

- ``HTTPException`` (FastAPI/Starlette routes, incl. the 404 for unknown paths)
- ``RequestValidationError`` (FastAPI pydantic body/path/query validation)
- bare ``Exception`` (catch-all so an uncaught error never leaks a stack trace)

``register_error_envelope_handlers(app, debug=...)`` — when ``debug`` is true,
500 responses carry a ``details`` string (exception text) to aid local
debugging; in production it is omitted. ``debug`` does NOT gate 422 detail.

A 422 tells the CALLER why, in every environment (see
``_public_validation_errors``): ``message`` is the first field error's ``msg``
(so a validator's explicit, user-facing reason reaches the FE toast, which
renders ``error.message``), and ``details`` is the structured
``[{loc, msg, type}]`` list. Until 0.76.1 the 422 said only "Request
validation failed" in production, so a user whose order was rejected by a
deliberate ``model_validator`` message could not learn why. ``input``, ``ctx``
and ``url`` never reach the wire. pydantic's own ``msg`` never does either for a
built-in error type — it interpolates the submitted value — so that text is
rebuilt from ``type`` via a fixed table (``_BUILTIN_MESSAGES``). Only a
``value_error`` / ``assertion_error`` carries the service validator's own text,
and **validators must therefore never interpolate secrets** (``_author_msg``).

A 422 additionally emits a WARNING log line carrying a **redacted projection**
of the pydantic errors — ``type`` plus ``loc``, with the ``loc`` positions the
caller controls masked to a 3-character prefix and a length (see
``_redact_validation_errors``) — so a failed request is diagnosable from the
logs. The log never carries ``msg`` at all: the log is a different trust
boundary from the response, which only ever goes back to the submitter.
"""

from __future__ import annotations

import logging
import re
import time as _time
import uuid as _uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    """ISO-8601 UTC timestamp (second precision) matching the fleet format."""
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


def make_error_payload(
    *,
    code: str,
    message: str,
    status: int,
    request_id: str,
    details: Any | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the canonical error-envelope dict ready for ``JSONResponse``.

    Args:
        code: Machine-readable code — ``HTTP_<status>`` / ``VALIDATION_ERROR``
            / ``INTERNAL_SERVER_ERROR``.
        message: Human-readable summary.
        status: HTTP status code (mirrored inside the envelope body).
        request_id: Correlation id (mirror of ``X-Request-Id``).
        details: Optional extra context; the key is omitted when ``None``.
        timestamp: ISO-8601 UTC; generated when ``None``.
    """
    error_body: dict[str, Any] = {
        "code": code,
        "message": message,
        "status": status,
        "request_id": request_id,
        "timestamp": timestamp or _utc_timestamp(),
    }
    if details is not None:
        error_body["details"] = details
    return {"error": error_body}


def _request_id_from(request: Request) -> str:
    """Request id from RequestIdMiddleware state, the inbound header, or a fresh uuid7."""
    rid = getattr(request.state, "request_id", None)
    if rid:
        return str(rid)
    return request.headers.get("x-request-id") or str(_uuid.uuid7())


def _find_http_exception(exc: BaseException) -> StarletteHTTPException | None:
    """Unwrap nested ExceptionGroups to find the first HTTPException."""
    if isinstance(exc, StarletteHTTPException):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            found = _find_http_exception(sub)
            if found is not None:
                return found
    return None


def _http_response(exc: StarletteHTTPException, request_id: str) -> JSONResponse:
    # FastAPI's HTTPException allows an arbitrary ``detail``; keep ``message`` a
    # string by stringifying when it isn't one already.
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=make_error_payload(
            code=f"HTTP_{exc.status_code}",
            message=message,
            status=exc.status_code,
            request_id=request_id,
        ),
        headers=dict(exc.headers or {}),
    )


#: Max number of pydantic errors projected into one log line — a hostile body
#: can carry thousands of fields; the log must not be floodable.
_MAX_LOGGED_VALIDATION_ERRORS = 20
#: Max characters kept per ``loc`` element. A bound on LOG VOLUME only — never
#: rely on it for secrecy; see ``_redact_validation_errors`` for why.
_MAX_LOC_ELEMENT_CHARS = 32

#: pydantic appends this literal to ``loc`` when the failure is in a mapping's
#: KEY rather than its value, e.g. ``('data', 'somekey', '[key]')``.
_DICT_KEY_MARKER = "[key]"


def _mask(text: str) -> str:
    """Render a caller-supplied ``loc`` element without disclosing it.

    Keeps the first three characters and the length — enough for an operator to
    correlate "an unexpected field starting ``AKI``, 20 characters" with a
    request they can find, and far too little to reconstruct a credential.
    """
    return f"{text[:3]}…({len(text)})"


def _redact_validation_errors(
    errors: Sequence[Mapping[str, object]],
) -> list[str]:
    """Project pydantic validation errors down to what is safe to log.

    Each pydantic error mapping has the keys ``type``, ``loc``, ``msg``,
    ``input``, and optionally ``ctx`` / ``url``. Only two are safe:

    - ``type`` — kept **verbatim**. It is a fixed pydantic enum
      (``missing``, ``int_parsing``, ``extra_forbidden``, ``value_error``, …)
      and never contains caller-submitted data.
    - ``loc`` — kept, except at the positions the CALLER controls, which are
      masked to ``<first-3-chars>…(<length>)``.

      An earlier version of this function truncated every ``loc`` element to
      32 characters and argued that "a 32-character cap cannot carry a
      credential". That was **refuted by measurement** and is why the masking
      below exists rather than a length cap::

          len=20 -> extra_forbidden at body.<a 20-char AWS-style key id>  # IN FULL
          len=32 -> extra_forbidden at body.aaaa...(32 hex)        # token, IN FULL

      Plenty of real credentials are at or under 32 characters, so a cap that
      only fires above 32 protects nothing. Masking is applied by POSITION,
      not by string content, because there is no reliable way to look at a
      string and tell a field name from a secret:

      - the **leaf of an** ``extra_forbidden`` **error** is, by definition,
        a key the caller invented: posting ``{"<a 20-char access-key id>": 1}`` to
        a model with ``extra="forbid"`` yields
        ``loc=('body', '<that same 20-char id>')``.
      - the element **before a literal** ``'[key]'`` **marker** is a
        caller-supplied mapping key (pydantic emits
        ``('data', 'somekey', '[key]')`` for a bad ``dict`` key).

      Every other element is schema-derived — a declared field name or an
      integer list index — and is kept verbatim so the path stays actionable.
      Elements are still capped at ``_MAX_LOC_ELEMENT_CHARS`` as a bound on
      log volume, not as a secrecy measure.

      Residual, accepted: a ``dict[str, X]`` **value** error puts the caller's
      key in ``loc`` with no marker to detect it. Keys there are field names
      chosen by the API's own contract (e.g. ``auth_config``'s
      ``login_url``/``username``/``password``), and the *values* — which are
      the secrets — are never logged.

    The rest are **never** included:

    - ``msg`` carries submitted values. Proven against pydantic 2.13.3: a
      ``field_validator`` raising ``ValueError(f"bad: {v}")`` yields
      ``msg='Value error, bad: sk-live-ABC123SECRET'``, and a
      discriminated-union tag error yields ``msg="Input tag
      'sk-live-ABC123SECRET' found using 'kind' does not match…"``. 24 live
      validators across the fleet interpolate their input, one echoing a whole
      ``target_url`` — and URLs in this fleet are a known credential sink.
    - ``input`` is the submitted value itself; ``ctx`` and ``url`` can embed
      it too. All three are unconditionally dropped.

    Args:
        errors: the sequence returned by ``RequestValidationError.errors()``.

    Returns:
        At most ``_MAX_LOGGED_VALIDATION_ERRORS`` strings of the form
        ``"<type> at <loc.path>"``.
    """
    projection: list[str] = []
    for error in errors[:_MAX_LOGGED_VALIDATION_ERRORS]:
        error_type = str(error.get("type", "unknown"))
        raw_loc = error.get("loc")
        elements = [str(part) for part in raw_loc] if isinstance(raw_loc, tuple | list) else []
        # Positions the CALLER chose, computed before rendering so masking is
        # driven by structure rather than by guessing at the string.
        caller_controlled = {
            index for index, text in enumerate(elements) if text == _DICT_KEY_MARKER and index > 0
        }
        caller_controlled = {index - 1 for index in caller_controlled}
        if error_type == "extra_forbidden" and elements:
            caller_controlled.add(len(elements) - 1)
        loc_parts = [
            _mask(text) if index in caller_controlled else text[:_MAX_LOC_ELEMENT_CHARS]
            for index, text in enumerate(elements)
        ]
        projection.append(f"{error_type} at {'.'.join(loc_parts) or '<root>'}")
    return projection


#: Fallback wire text for any error type not in ``_BUILTIN_MESSAGES``.
_GENERIC_MSG = "Invalid value"
#: Max characters of one ``msg`` put on the wire (a validator could build a huge one).
_MAX_PUBLIC_MSG_CHARS = 512
#: Error types whose text is written by SERVICE code (a validator's
#: ``raise ValueError(...)`` / ``assert ..., "..."``), not by pydantic.
_AUTHOR_MSG_TYPES = frozenset({"value_error", "assertion_error"})
#: pydantic's prefix on a ``ValueError`` / ``AssertionError`` raised by a validator.
_VALIDATOR_MSG_PREFIX = re.compile(r"^(Value error|Assertion failed), ")
#: Placeholder substituted for a submitted string an author ``msg`` echoed verbatim.
_REDACTED = "[redacted]"
#: Max string leaves scanned out of one error's ``input`` for the author-text scrub.
#: Past it the author text is withheld (fail closed).
_MAX_SCRUB_LEAVES = 256
#: Submitted strings shorter than this are not scrubbed out of author text.
_MIN_SCRUB_CHARS = 4
#: A lowercase snake_case token WITH an underscore (``website_review``,
#: ``title_card``) is enum/Literal vocabulary a deliberate reason names, so the
#: scrub leaves it; a plain word (``correcthorsebattery``) is still scrubbed
#: (see ``_looks_secret_shaped`` for the length/shape floor on the latter).
_ENUM_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")
#: Below this length, a value is scrubbed only if it is shaped like a secret
#: (``_looks_secret_shaped``); at or above it, any verbatim match is scrubbed
#: regardless of shape — length alone is enough signal for a real credential.
_MIN_SECRET_SHAPE_CHARS = 12
#: A single lowercase-alphabetic word, any length. Ordinary English/schema
#: vocabulary is never secret-shaped, no matter how it got into the payload.
_LOWERCASE_WORD = re.compile(r"[a-z]+")
#: Common schema/field words short enough to collide with real field names
#: but not alphabetic-only in every submitted form seen in prod (kept as an
#: explicit allow-list rather than relying solely on ``_LOWERCASE_WORD`` so a
#: capitalized or mixed-case submission of the same word, e.g. ``Title``, is
#: still spared).
_COMMON_SCHEMA_WORDS = frozenset(
    {
        "title",
        "name",
        "url",
        "type",
        "id",
        "image",
        "video",
        "brand",
        "voice",
        "theme",
        "style",
        "email",
        "phone",
        "color",
        "status",
        "format",
        "action",
        "reason",
        "target",
        "source",
        "label",
        "value",
        "card",
        "intro",
        "cards",
        "review",
    }
)
#: A declared field name / alias. Any other ``loc`` string is caller-shaped.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
#: FastAPI names a declared header/cookie by its hyphenated alias (``x-request-id``).
_HEADER_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
_HEADER_PARTS = frozenset({"header", "cookie"})
#: FastAPI's first ``loc`` element names the request part, not a field.
_REQUEST_PARTS = frozenset({"body", "query", "path", "header", "cookie"})

#: Wire text for pydantic's BUILT-IN error types, keyed by ``type``.
#:
#: pydantic's own ``msg`` is NEVER passed through for these: it interpolates the
#: submitted value (``union_tag_invalid`` quotes the caller's tag; every
#: ``*_parsing`` type appends a parser ``error`` such as "found `S` at 1";
#: ``too_short`` echoes the input's length). Each entry is ``(template,
#: ctx_keys)``: the template is fixed text, and the only placeholders it may use
#: are the listed ``ctx`` keys, each of which was VERIFIED against pydantic
#: 2.13.3 to be schema-derived (a declared constraint, literal, discriminator or
#: scheme list) — never ``tag``, ``error``, ``actual_length`` or ``input``. When a
#: listed key is absent the template is not used and ``_GENERIC_MSG`` is.
_BUILTIN_MESSAGES: dict[str, tuple[str, tuple[str, ...]]] = {
    "missing": ("Field required", ()),
    "extra_forbidden": ("Extra inputs are not permitted", ()),
    "json_invalid": ("Invalid JSON", ()),
    "json_type": ("Input should be a valid JSON string", ()),
    "frozen_field": ("Field is frozen", ()),
    "frozen_instance": ("Instance is frozen", ()),
    "none_required": ("Input should be null", ()),
    "is_instance_of": ("Input is not of the expected type", ()),
    "model_type": ("Input should be an object", ()),
    "model_attributes_type": ("Input should be an object", ()),
    "dataclass_type": ("Input should be an object", ()),
    "dict_type": ("Input should be a valid dictionary", ()),
    "list_type": ("Input should be a valid list", ()),
    "tuple_type": ("Input should be a valid tuple", ()),
    "set_type": ("Input should be a valid set", ()),
    "frozen_set_type": ("Input should be a valid frozenset", ()),
    "iterable_type": ("Input should be iterable", ()),
    "string_type": ("Input should be a valid string", ()),
    "string_unicode": ("Input should be a valid string", ()),
    "string_sub_type": ("Input should be a valid string", ()),
    "bytes_type": ("Input should be valid bytes", ()),
    "bool_type": ("Input should be a valid boolean", ()),
    "bool_parsing": ("Input should be a valid boolean", ()),
    "int_type": ("Input should be a valid integer", ()),
    "int_parsing": ("Input should be a valid integer", ()),
    "int_parsing_size": ("Input should be a valid integer", ()),
    "int_from_float": ("Input should be a whole number", ()),
    "float_type": ("Input should be a valid number", ()),
    "float_parsing": ("Input should be a valid number", ()),
    "finite_number": ("Input should be a finite number", ()),
    "decimal_type": ("Input should be a valid decimal", ()),
    "decimal_parsing": ("Input should be a valid decimal", ()),
    "decimal_max_digits": ("Decimal should have at most {max_digits} digits", ("max_digits",)),
    "decimal_max_places": (
        "Decimal should have at most {decimal_places} decimal places",
        ("decimal_places",),
    ),
    "uuid_type": ("Input should be a valid UUID", ()),
    "uuid_parsing": ("Input should be a valid UUID", ()),
    "date_type": ("Input should be a valid date", ()),
    "date_parsing": ("Input should be a valid date", ()),
    "date_from_datetime_parsing": ("Input should be a valid date", ()),
    "date_from_datetime_inexact": ("Input should be a date with no time part", ()),
    "date_past": ("Date should be in the past", ()),
    "date_future": ("Date should be in the future", ()),
    "datetime_type": ("Input should be a valid datetime", ()),
    "datetime_parsing": ("Input should be a valid datetime", ()),
    "datetime_from_date_parsing": ("Input should be a valid datetime", ()),
    "datetime_past": ("Datetime should be in the past", ()),
    "datetime_future": ("Datetime should be in the future", ()),
    "timezone_naive": ("Datetime should not have timezone info", ()),
    "timezone_aware": ("Datetime should have timezone info", ()),
    "time_type": ("Input should be a valid time", ()),
    "time_parsing": ("Input should be a valid time", ()),
    "time_delta_type": ("Input should be a valid duration", ()),
    "time_delta_parsing": ("Input should be a valid duration", ()),
    "url_type": ("Input should be a valid URL", ()),
    "url_parsing": ("Input should be a valid URL", ()),
    "url_syntax_violation": ("Input should be a valid URL", ()),
    "url_scheme": ("URL scheme should be {expected_schemes}", ("expected_schemes",)),
    "url_too_long": ("URL should have at most {max_length} characters", ("max_length",)),
    "enum": ("Input should be {expected}", ("expected",)),
    "literal_error": ("Input should be {expected}", ("expected",)),
    "string_too_short": ("String should have at least {min_length} characters", ("min_length",)),
    "string_too_long": ("String should have at most {max_length} characters", ("max_length",)),
    "string_pattern_mismatch": ("String should match pattern {pattern}", ("pattern",)),
    "bytes_too_short": ("Data should have at least {min_length} bytes", ("min_length",)),
    "bytes_too_long": ("Data should have at most {max_length} bytes", ("max_length",)),
    "too_short": ("Should have at least {min_length} items", ("min_length",)),
    "too_long": ("Should have at most {max_length} items", ("max_length",)),
    "greater_than": ("Input should be greater than {gt}", ("gt",)),
    "greater_than_equal": ("Input should be greater than or equal to {ge}", ("ge",)),
    "less_than": ("Input should be less than {lt}", ("lt",)),
    "less_than_equal": ("Input should be less than or equal to {le}", ("le",)),
    "multiple_of": ("Input should be a multiple of {multiple_of}", ("multiple_of",)),
    "union_tag_invalid": (
        "Unknown {discriminator}; expected one of {expected_tags}",
        ("discriminator", "expected_tags"),
    ),
    "union_tag_not_found": ("Missing discriminator {discriminator}", ("discriminator",)),
}


def _looks_secret_shaped(item: str) -> bool:
    """Whether a submitted string is worth redacting out of author text.

    Live prod defect (fixed here): the scrub was over-eager on short/ordinary
    words. A ``website_review`` order with a submitted field value of
    ``"title"`` turned "the shortest possible review is 7.2s (title_card
    intro + ...)" into "...([redacted]_card intro + ...)" — ``title`` isn't a
    secret, it collided with the validator's OWN vocabulary and made the
    message unreadable. ``_ENUM_TOKEN`` already spares multi-word
    ``snake_case`` tokens; this closes the gap for single common words.

    At or above ``_MIN_SECRET_SHAPE_CHARS`` any verbatim match is treated as
    secret-shaped — length alone is real signal (API keys, tokens, long
    identifiers). Below that floor, a value is only secret-shaped if it does
    NOT look like ordinary dictionary vocabulary: a bare lowercase word or a
    small stoplist entry is spared even if it happens to appear in the
    message; anything else short (mixed case + digits, punctuation, ``@``,
    ``://``, ...) is still redacted, because THAT shape is what a short
    secret (an email, a short token, a URL fragment) actually looks like.
    """
    if len(item) >= _MIN_SECRET_SHAPE_CHARS:
        return True
    if item.lower() in _COMMON_SCHEMA_WORDS:
        return False
    return not _LOWERCASE_WORD.fullmatch(item)


def _submitted_strings(value: object) -> list[str] | None:
    """Every secret-shaped string (dict keys included) of ``_MIN_SCRUB_CHARS``+ in a value.

    Excludes ``_ENUM_TOKEN``-shaped ``snake_case`` vocabulary and, per
    ``_looks_secret_shaped``, short ordinary words that only collide with a
    validator's own wording rather than leaking anything.

    Returns ``None`` past ``_MAX_SCRUB_LEAVES`` — the caller then withholds.
    """
    found: list[str] = []
    stack: list[object] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list | tuple | set | frozenset):
            stack.extend(item)
        elif (
            isinstance(item, str)
            and len(item) >= _MIN_SCRUB_CHARS
            and not _ENUM_TOKEN.fullmatch(item)
            and _looks_secret_shaped(item)
        ):
            found.append(item)
            if len(found) > _MAX_SCRUB_LEAVES:
                return None
    return found


def _author_msg(error: Mapping[str, object]) -> str:
    """The text a SERVICE validator raised, prefix stripped.

    This is service-code-controlled text and it is passed through BY CONTRACT:
    it is the only place a deliberate, user-facing reason (e.g. the
    orchestrator's website_review minimum-order message) can live.

    ⚠️ CONTRACT FOR VALIDATOR AUTHORS: the text of a ``ValueError`` /
    ``AssertionError`` raised in a validator is returned to the caller in every
    environment. It MUST NOT interpolate secrets or submitted credentials —
    not whole, not truncated (``v[:10]``), not ``!r``-quoted, not case-changed,
    not another field's value (``info.data``). Interpolating a harmless scalar
    the caller just sent (a duration, an enum value) is fine.

    As defense in depth ONLY, any submitted string of this field's ``input``
    that appears VERBATIM in the text is replaced by ``[redacted]``. This is
    NOT a guarantee and must not be relied on — it cannot catch a transformed
    or cross-field value (``tests/test_asgi_factory.py`` pins exactly which
    transforms it misses).
    """
    msg: str = _VALIDATOR_MSG_PREFIX.sub("", str(error.get("msg", "")))
    submitted = _submitted_strings(error.get("input"))
    if submitted is None:
        return _GENERIC_MSG
    # Longest first, so a string containing a shorter submitted one goes whole.
    ordered: list[str] = sorted(set(submitted), key=lambda item: len(item), reverse=True)
    for text in ordered:
        msg = msg.replace(text, _REDACTED)
    if len(msg) > _MAX_PUBLIC_MSG_CHARS:
        msg = msg[: _MAX_PUBLIC_MSG_CHARS - 1] + "…"
    return msg or _GENERIC_MSG


def _builtin_msg(error_type: str, ctx: object) -> str:
    """Input-free wire text for a pydantic built-in error type (see ``_BUILTIN_MESSAGES``)."""
    entry = _BUILTIN_MESSAGES.get(error_type)
    if entry is None:
        return _GENERIC_MSG
    template, keys = entry
    if not keys:
        return template
    if not isinstance(ctx, Mapping):
        return _GENERIC_MSG
    typed_ctx = cast("Mapping[str, object]", ctx)
    values: dict[str, str] = {}
    for key in keys:
        if key not in typed_ctx:
            return _GENERIC_MSG
        values[key] = str(typed_ctx[key])
    rendered = template.format(**values)
    if len(rendered) > _MAX_PUBLIC_MSG_CHARS:
        return _GENERIC_MSG
    return rendered


def _public_msg(error: Mapping[str, object]) -> str:
    """Wire ``msg`` for one error: author text for validator errors, else a fixed template.

    pydantic's own ``msg`` is never forwarded for a built-in type; an unknown
    type (incl. a custom ``PydanticCustomError``) gets ``_GENERIC_MSG``.
    """
    error_type = str(error.get("type", ""))
    if error_type in _AUTHOR_MSG_TYPES:
        return _author_msg(error)
    return _builtin_msg(error_type, error.get("ctx"))


def _public_validation_errors(
    errors: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Project pydantic errors down to what may go back to the CALLER.

    Keys: ``type`` (fixed pydantic enum, verbatim), ``loc`` and ``msg``.
    ``input``, ``ctx`` and ``url`` are never included — they carry the
    submitted value (``ctx`` holds the raised exception object).

    - ``loc``: ints kept; strings kept when they are a declared-identifier
      shape AND not at a caller-controlled position (the ``extra_forbidden``
      leaf, the element before ``'[key]'`` — see ``_redact_validation_errors``);
      otherwise replaced by ``[redacted:<length>]``.
    - ``msg``: see ``_public_msg`` — built-in types get fixed, input-free text
      from ``_BUILTIN_MESSAGES``; only ``value_error`` / ``assertion_error``
      carry the service validator's own text (``_author_msg``).

    Capped at ``_MAX_LOGGED_VALIDATION_ERRORS`` entries.
    """
    public: list[dict[str, Any]] = []
    for error in errors[:_MAX_LOGGED_VALIDATION_ERRORS]:
        error_type = str(error.get("type", "unknown"))
        raw_loc = error.get("loc")
        parts: list[object] = list(raw_loc) if isinstance(raw_loc, tuple | list) else []
        caller_controlled = {
            index - 1 for index, part in enumerate(parts) if part == _DICT_KEY_MARKER and index > 0
        }
        if error_type == "extra_forbidden" and parts:
            caller_controlled.add(len(parts) - 1)
        loc: list[object] = []
        for index, part in enumerate(parts):
            if isinstance(part, int) and not isinstance(part, bool):
                loc.append(part)
                continue
            text = str(part)
            shape = _HEADER_IDENTIFIER if parts[0] in _HEADER_PARTS else _IDENTIFIER
            if index in caller_controlled or not (
                text == _DICT_KEY_MARKER or shape.fullmatch(text)
            ):
                # The wire gets the LENGTH only — no prefix, unlike the log's
                # ``_mask``: a caller-shaped key may itself be a credential.
                loc.append(f"[redacted:{len(text)}]")
            else:
                loc.append(text)
        public.append({"loc": loc, "msg": _public_msg(error), "type": error_type})
    return public


def _validation_message(public: Sequence[Mapping[str, Any]], total: int) -> str:
    """One-line human summary of the (already scrubbed) field errors."""
    if not public:
        return "Request validation failed"
    first = public[0]
    loc = [str(part) for part in first["loc"]]
    if loc and loc[0] in _REQUEST_PARTS:
        loc = loc[1:]
    message = f"{'.'.join(loc)}: {first['msg']}" if loc else str(first["msg"])
    if total > 1:
        message += f" (and {total - 1} more)"
    return message


def register_error_envelope_handlers(app: FastAPI, *, debug: bool = False) -> None:
    """Wire the three canonical exception handlers onto ``app``.

    Args:
        app: the FastAPI application.
        debug: when true, 500 responses include a ``details`` string
            (exception text). Off in production so the wire never leaks
            internals. ``debug`` does not affect a 422: its scrubbed
            ``{loc, msg, type}`` field errors are returned in every
            environment (``_public_validation_errors``) — a 422 is the
            caller's fault and the caller must be able to learn why. A 422
            also logs a redacted ``type``/``loc`` projection at WARNING.

    Idempotent: re-registering replaces the handler for the same class.
    ``create_app()`` calls this; services can register extra handlers after.
    """

    async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Starlette widens the handler's exception parameter to ``Exception``;
        # narrow back here. A non-HTTP exception reaching this is a wiring bug.
        if not isinstance(exc, StarletteHTTPException):
            raise TypeError(f"_http_exception_handler got non-HTTP exception: {type(exc).__name__}")
        return _http_response(exc, _request_id_from(request))

    async def _validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, RequestValidationError):
            raise TypeError(
                f"_validation_exception_handler got non-validation exception: {type(exc).__name__}"
            )
        request_id = _request_id_from(request)
        # WARNING, not ERROR: a 422 is a caller fault. ERROR stays reserved for
        # the catch-all 500 below so the two remain distinguishable in logs.
        logger.warning(
            "request validation failed for %s %s (request_id=%s): %s",
            request.method,
            request.url.path,
            request_id,
            _redact_validation_errors(exc.errors()),
        )
        errors = exc.errors()
        public = _public_validation_errors(errors)
        return JSONResponse(
            status_code=422,
            content=make_error_payload(
                code="VALIDATION_ERROR",
                message=_validation_message(public, len(errors)),
                status=422,
                request_id=request_id,
                details=public,
            ),
        )

    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all — never leak a stack trace to the wire."""
        request_id = _request_id_from(request)
        # An HTTPException can arrive wrapped in an ExceptionGroup (e.g. raised
        # inside a TaskGroup); render it as the HTTP error it is, not a 500.
        http_exc = _find_http_exception(exc)
        if http_exc is not None:
            return _http_response(http_exc, request_id)
        logger.exception(
            "unhandled exception in %s %s (request_id=%s): %s",
            request.method,
            request.url.path,
            request_id,
            exc,
        )
        return JSONResponse(
            status_code=500,
            content=make_error_payload(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred",
                status=500,
                request_id=request_id,
                details=str(exc) if debug else None,
            ),
        )

    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
