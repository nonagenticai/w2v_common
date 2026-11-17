"""Tests for `w2v_common.asgi.create_app`."""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Annotated, Literal

import pytest
from fastapi import APIRouter, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from pydantic_core import PydanticCustomError

from w2v_common.asgi import create_app, make_error_payload
from w2v_common.asgi.error_envelope import _MAX_LOGGED_VALIDATION_ERRORS
from w2v_common.config import APISettings, AuthSettings

_ENVELOPE_LOGGER = "w2v_common.asgi.error_envelope"


class StrictPayload(BaseModel):
    """Mirrors the live request models that both forbid extras AND interpolate
    the submitted value into a validator message — the two proven leak paths."""

    model_config = ConfigDict(extra="forbid")

    name: str

    @field_validator("name")
    @classmethod
    def _always_rejects(cls, value: str) -> str:
        raise ValueError(f"bad: {value}")


class OrderPayload(BaseModel):
    """Mirrors w2v_orchestrator ``ProjectCreate``: a ``model_validator`` raising
    an explicit, user-facing reason (the website_review minimum-order floor)."""

    type: str
    target_duration_seconds: float
    target_url: str = ""

    @model_validator(mode="after")
    def _minimum_order(self) -> OrderPayload:
        if self.type == "website_review" and self.target_duration_seconds < 18:
            raise ValueError(
                f"website_review ordered {self.target_duration_seconds:g}s, but the "
                "shortest possible review is 18s. Order at least 18s or disable title_card."
            )
        return self


class Target(BaseModel):
    """A validator that interpolates a submitted credential-bearing URL."""

    target_url: str

    @field_validator("target_url")
    @classmethod
    def _reject(cls, value: str) -> str:
        raise ValueError(f"cannot reach {value}")


class _AuthorPayload(BaseModel):
    """Validators that interpolate a TRANSFORMED submitted value (auditor probe)."""

    token: str | None = None
    note: str | None = None
    email: str | None = None
    card: int | None = None
    nickname: str | None = None

    @field_validator("token")
    @classmethod
    def _token(cls, value: str) -> str:
        raise ValueError(f"token prefix {value[:10]} unknown")

    @field_validator("note")
    @classmethod
    def _note(cls, value: str) -> str:
        raise ValueError(f"bad note {value!r}")

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        raise ValueError(f"{value.lower()} already registered")

    @field_validator("card")
    @classmethod
    def _card(cls, value: int) -> int:
        raise ValueError(f"card {value} declined")

    @field_validator("nickname")
    @classmethod
    def _nickname(cls, value: str) -> str:
        raise ValueError(f"nickname {value} is taken")


class _CustomErrorPayload(BaseModel):
    field: str

    @field_validator("field")
    @classmethod
    def _custom(cls, value: str) -> str:
        raise PydanticCustomError("my_custom", "rejected {v}", {"v": value})


def _post_router(path: str, model: type[BaseModel]) -> APIRouter:
    """One POST route taking ``model`` as body, annotated with the real class
    (``from __future__ import annotations`` would make a local class unresolvable)."""
    router = APIRouter()

    async def endpoint(payload: BaseModel) -> dict[str, str]:
        return {}

    endpoint.__annotations__["payload"] = model
    router.add_api_route(path, endpoint, methods=["POST"])
    return router


class _TagA(BaseModel):
    kind: Literal["a"]


class _TagB(BaseModel):
    kind: Literal["b"]


@pytest.fixture
def settings() -> APISettings:
    return APISettings(port=8080, host="127.0.0.1")


@pytest.fixture
def hello_router() -> APIRouter:
    router = APIRouter()

    @router.get("/hello")
    async def hello() -> dict[str, str]:
        return {"message": "world"}

    @router.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("kaboom")

    @router.get("/forbidden")
    async def forbidden() -> dict[str, str]:
        raise HTTPException(status_code=403, detail="nope")

    @router.post("/echo")
    async def echo(payload: dict) -> dict:
        return payload

    @router.post("/strict")
    async def strict(payload: StrictPayload) -> dict[str, str]:
        return {"name": payload.name}

    @router.post("/order")
    async def order(payload: OrderPayload) -> dict[str, str]:
        return {"type": payload.type}

    @router.post("/explicit-422")
    async def explicit_422() -> dict[str, str]:
        raise HTTPException(status_code=422, detail="Order at least 18s")

    return router


def _envelope_warnings(caplog) -> list[logging.LogRecord]:
    """WARNING records emitted by the error-envelope module itself."""
    return [
        record
        for record in caplog.records
        if record.name == _ENVELOPE_LOGGER and record.levelno == logging.WARNING
    ]


async def _client(app):
    # raise_app_exceptions=False: keep ASGI exceptions inside the app so
    # the registered Exception handler can convert them to 500 responses
    # (without it, httpx re-raises the exception client-side and our
    # canonical envelope handler appears to "not fire").
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    )


class TestCreateApp:
    async def test_happy_path_routes_respond(self, settings, hello_router) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.get("/hello")
        assert resp.status_code == 200
        assert resp.json() == {"message": "world"}

    async def test_request_id_header_set(self, settings, hello_router) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.get("/hello")
        assert "x-request-id" in {k.lower() for k in resp.headers}

    async def test_security_headers_set(self, settings, hello_router) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.get("/hello")
        lower_keys = {k.lower() for k in resp.headers}
        assert "x-content-type-options" in lower_keys
        assert "x-frame-options" in lower_keys

    async def test_uncaught_exception_returns_canonical_envelope(
        self,
        settings,
        hello_router,
    ) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.get("/boom", headers={"X-Request-Id": "test-id-abc"})
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
        assert body["error"]["message"] == "An unexpected error occurred"
        assert body["error"]["status"] == 500
        assert body["error"]["timestamp"]
        # request_id mirrored from the header (via RequestIdMiddleware putting
        # it on request.state) — lives inside the error body in shape #2.
        assert body["error"]["request_id"] == "test-id-abc"

    async def test_http_exception_returns_canonical_envelope(
        self,
        settings,
        hello_router,
    ) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.get("/forbidden")
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "HTTP_403"
        assert body["error"]["message"] == "nope"
        assert body["error"]["status"] == 403
        assert body["error"]["request_id"]
        assert body["error"]["timestamp"]

    async def test_validation_error_returns_canonical_envelope(
        self,
        settings,
        hello_router,
    ) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            # /echo expects a JSON body; sending a non-dict trips pydantic.
            resp = await client.post("/echo", json=["not", "a", "dict"])
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Input should be a valid dictionary"
        assert body["error"]["details"] == [
            {"loc": ["body"], "msg": "Input should be a valid dictionary", "type": "dict_type"}
        ]
        assert body["error"]["status"] == 422
        assert body["error"]["request_id"]

    async def test_validation_error_logs_redacted_projection(
        self,
        settings,
        hello_router,
        caplog,
    ) -> None:
        """The 422 must be diagnosable from the logs (field paths + error types)
        WITHOUT leaking submitted values: pydantic's ``msg`` interpolates the
        input, so only ``type`` + a truncated ``loc`` may be projected."""
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        body_payload = {
            "name": "sk-live-VALUE-SECRET",  # trips the field_validator -> value_error
            "sk-live-EXTRA-SECRET": 1,  # unexpected key -> extra_forbidden
        }
        with caplog.at_level(logging.WARNING, logger=_ENVELOPE_LOGGER):
            async with await _client(app) as client:
                resp = await client.post("/strict", json=body_payload)

        # The value that pydantic would have echoed through `msg` must not appear.
        assert "sk-live-VALUE-SECRET" not in caplog.text
        # NOR may the unexpected KEY. For `extra_forbidden` the leaf of `loc` is
        # a string the caller invented, so it is masked by position. An earlier
        # revision only capped it at 32 chars and this assertion was missing --
        # so the test tacitly accepted the very leak it was written to guard.
        assert "sk-live-EXTRA-SECRET" not in caplog.text
        # The mask keeps just enough to correlate with a request.
        assert "sk-…(20)" in caplog.text
        # ...but the error types must appear, so the log is actionable.
        assert "extra_forbidden" in caplog.text
        assert "value_error" in caplog.text

        assert len(_envelope_warnings(caplog)) == 1

        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        # The response DOES carry field detail now, but neither submitted secret:
        # the value `msg` interpolated is scrubbed, the invented key is masked.
        assert "sk-live-VALUE-SECRET" not in resp.text
        assert "sk-live-EXTRA-SECRET" not in resp.text
        details = body["error"]["details"]
        assert {"loc": ["body", "name"], "msg": "bad: [redacted]", "type": "value_error"} in details
        assert {
            "loc": ["body", "[redacted:20]"],
            "msg": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        } in details
        assert all(set(entry) == {"loc", "msg", "type"} for entry in details)

    async def test_model_validator_reason_reaches_caller_in_production(
        self,
        settings,
        hello_router,
    ) -> None:
        """Live prod defect: a too-short website_review order got a bare
        "Request validation failed" — the orchestrator's explicit reason was
        dropped, so the FE toast (which renders ``error.message``) could not say
        why. ``settings`` is production (``debug`` off)."""
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        payload = {
            "type": "website_review",
            "target_duration_seconds": 10,
            "target_url": "https://example.com/private-path-MARKER-7731",
        }
        async with await _client(app) as client:
            resp = await client.post("/order", json=payload)
        assert resp.status_code == 422
        error = resp.json()["error"]
        expected = (
            "website_review ordered 10s, but the shortest possible review is 18s. "
            "Order at least 18s or disable title_card."
        )
        assert error["code"] == "VALIDATION_ERROR"
        assert error["message"] == expected
        assert error["details"] == [{"loc": ["body"], "msg": expected, "type": "value_error"}]
        # The model-level `input` is the whole body; none of it is echoed.
        assert "MARKER-7731" not in resp.text
        assert "input" not in error["details"][0]
        assert "ctx" not in error["details"][0]

    async def test_validator_echoing_submitted_url_is_scrubbed(self, hello_router) -> None:
        router = APIRouter()

        @router.post("/target")
        async def target(payload: Target) -> dict[str, str]:
            return {"url": payload.target_url}

        # debug on too: debug must not re-open a raw dump of the errors.
        debug_settings = APISettings(port=8080, host="127.0.0.1", debug=True)
        app = create_app(debug_settings, routers=[router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.post(
                "/target", json={"target_url": "https://internal.example/private-MARKER-4412"}
            )
        assert resp.status_code == 422
        assert "MARKER-4412" not in resp.text
        error = resp.json()["error"]
        assert error["message"] == "target_url: cannot reach [redacted]"

    async def test_multiple_errors_summarised(self, settings, hello_router) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.post("/order", json={})
        error = resp.json()["error"]
        assert error["message"] == "type: Field required (and 1 more)"
        assert [d["loc"] for d in error["details"]] == [
            ["body", "type"],
            ["body", "target_duration_seconds"],
        ]

    async def test_explicit_http_422_message_passes_through(self, settings, hello_router) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.post("/explicit-422")
        assert resp.status_code == 422
        assert resp.json()["error"]["message"] == "Order at least 18s"

    async def test_500_still_hides_internals_in_production(self, settings, hello_router) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.get("/boom")
        assert resp.status_code == 500
        assert "kaboom" not in resp.text
        assert "details" not in resp.json()["error"]

    @pytest.mark.parametrize(
        ("annotation", "submitted", "expected_msg"),
        [
            # union_tag_invalid: pydantic's msg quotes the caller's tag.
            (
                Annotated[_TagA | _TagB, Field(discriminator="kind")],
                {"kind": "Sk-Probe-Token-MARKER"},
                "Unknown 'kind'; expected one of 'a', 'b'",
            ),
            # literal_error: `expected` is schema-derived, the input is not echoed.
            (Literal["a_b", "c_d"], "Sk-Probe-Token-MARKER", "Input should be 'a_b' or 'c_d'"),
            # *_parsing: pydantic appends a parser error describing the input.
            (int, "not a number PROBEVALUE-MARKER", "Input should be a valid integer"),
            (uuid.UUID, "PROBEVALUE-MARKER", "Input should be a valid UUID"),
            (datetime.datetime, "PROBEVALUE-MARKER", "Input should be a valid datetime"),
            # too_short: pydantic's msg carries the INPUT's length; only min_length may go out.
            (
                Annotated[list[str], Field(min_length=3)],
                ["PROBEVALUE-MARKER"],
                "Should have at least 3 items",
            ),
            (Annotated[int, Field(gt=100)], 5, "Input should be greater than 100"),
        ],
    )
    async def test_builtin_error_msg_is_rebuilt_from_type_not_passed_through(
        self, settings, annotation, submitted, expected_msg
    ) -> None:
        """Auditor probe: pydantic's own ``msg`` interpolates the submitted value
        for built-in types, so the wire text must come from the fixed table."""
        probe_model = create_model("Probe", field=(annotation, ...))
        app = create_app(
            settings, routers=[_post_router("/probe", probe_model)], auth_enabled=False
        )
        async with await _client(app) as client:
            resp = await client.post("/probe", json={"field": submitted})
        assert resp.status_code == 422
        assert "MARKER" not in resp.text
        assert "PROBEVALUE" not in resp.text
        error = resp.json()["error"]
        assert error["details"][0]["msg"] == expected_msg
        assert error["message"].endswith(expected_msg)
        assert all(set(entry) == {"loc", "msg", "type"} for entry in error["details"])

    async def test_invalid_json_and_header_errors_do_not_echo_input(self, settings) -> None:
        router = _post_router("/json", Target)

        @router.get("/hdr")
        async def hdr_route(x_count: int = Header()) -> dict[str, str]:
            return {}

        app = create_app(settings, routers=[router], auth_enabled=False)
        async with await _client(app) as client:
            bad_json = await client.post(
                "/json",
                content=b'{"token": "PROBEVALUE-MARKER" ,,}',
                headers={"content-type": "application/json"},
            )
            bad_header = await client.get("/hdr", headers={"x-count": "Bearer-PROBEVALUE-MARKER"})
        assert bad_json.status_code == 422
        assert "MARKER" not in bad_json.text
        assert bad_json.json()["error"]["details"][0]["msg"] == "Invalid JSON"
        assert bad_header.status_code == 422
        assert "MARKER" not in bad_header.text
        assert bad_header.json()["error"]["message"] == ("x-count: Input should be a valid integer")

    async def test_unknown_error_type_fails_closed(self, settings) -> None:
        """A custom error type is not in the table: its msg must not pass through."""
        app = create_app(
            settings, routers=[_post_router("/custom", _CustomErrorPayload)], auth_enabled=False
        )
        async with await _client(app) as client:
            resp = await client.post("/custom", json={"field": "PROBEVALUE-MARKER"})
        assert "MARKER" not in resp.text
        assert resp.json()["error"]["details"] == [
            {"loc": ["body", "field"], "msg": "Invalid value", "type": "my_custom"}
        ]

    @pytest.mark.parametrize(
        ("submitted", "expected_msg"),
        [
            # A transformed or cross-field value is NOT caught by the verbatim
            # defense-in-depth scrub. These pin the documented contract that
            # value_error text is AUTHOR-controlled: the author must not
            # interpolate secrets. If one of these starts failing because the
            # value is now hidden, update the contract docs, not just the test.
            ({"token": "abcdefghijklmnop"}, "token prefix abcdefghij unknown"),
            ({"note": "Two\nLines"}, "bad note 'Two\\nLines'"),
            ({"email": "Bob@Example.com"}, "bob@example.com already registered"),
            ({"card": 4111111111111111}, "card 4111111111111111 declined"),
            # Verbatim echo IS scrubbed — including lowercase free text.
            ({"nickname": "correcthorsebattery"}, "nickname [redacted] is taken"),
        ],
    )
    async def test_value_error_text_is_author_controlled(
        self, settings, submitted, expected_msg
    ) -> None:
        app = create_app(
            settings, routers=[_post_router("/author", _AuthorPayload)], auth_enabled=False
        )
        async with await _client(app) as client:
            resp = await client.post("/author", json=submitted)
        assert resp.status_code == 422
        assert resp.json()["error"]["details"][0]["msg"] == expected_msg

    @pytest.mark.parametrize(
        ("nickname", "expected_msg"),
        [
            # Live prod defect: a short, ordinary submitted word must survive
            # verbatim inside author text — it collided with the validator's
            # OWN vocabulary, it wasn't a leak.
            ("title", "nickname title is taken"),
            ("name", "nickname name is taken"),
            ("url", "nickname url is taken"),
            # A 9-char ordinary lowercase word, not in the stoplist, is still
            # dictionary-shaped and must not be redacted either.
            ("dashboard", "nickname dashboard is taken"),
            # Secret-shaped values remain redacted regardless of length.
            ("sk-live-ABCDEF123456", "nickname [redacted] is taken"),
            ("correcthorsebattery", "nickname [redacted] is taken"),  # >=12 chars
            ("a@b.co", "nickname [redacted] is taken"),  # short but '@'-shaped
            ("Ab3!xy", "nickname [redacted] is taken"),  # short but mixed shape
        ],
    )
    async def test_scrub_spares_common_words_but_catches_secrets(
        self, settings, nickname, expected_msg
    ) -> None:
        app = create_app(
            settings, routers=[_post_router("/author", _AuthorPayload)], auth_enabled=False
        )
        async with await _client(app) as client:
            resp = await client.post("/author", json={"nickname": nickname})
        assert resp.status_code == 422
        assert resp.json()["error"]["details"][0]["msg"] == expected_msg

    async def test_prod_message_survives_common_word_collision(self, settings) -> None:
        """Live prod defect, reproduced verbatim: a ``website_review`` order's
        explicit minimum-order reason names ``title_card`` in its OWN fixed
        text, and the caller separately submitted a field whose value is the
        plain word ``"title"``. The old scrub matched ``"title"`` against the
        author text and mangled it into "[redacted]_card intro + [redacted]
        cards ..." — a confusing message about a value that was never a
        secret. The fix must leave the author's text untouched."""

        class WebsiteReviewOrder(BaseModel):
            type: str
            target_duration_seconds: float
            sections: list[str] = []

            @model_validator(mode="after")
            def _minimum_order(self) -> WebsiteReviewOrder:
                if self.type == "website_review" and self.target_duration_seconds < 7.2:
                    raise ValueError(
                        f"website_review ordered {self.target_duration_seconds:g}s, but the "
                        "shortest possible review is 7.2s (title_card intro + 2 cards, "
                        "3s each)"
                    )
                return self

        app = create_app(
            settings,
            routers=[_post_router("/website-review", WebsiteReviewOrder)],
            auth_enabled=False,
        )
        payload = {
            "type": "website_review",
            "target_duration_seconds": 3,
            # The submitted value that used to collide with the author text.
            "sections": ["title", "cards"],
        }
        async with await _client(app) as client:
            resp = await client.post("/website-review", json=payload)
        assert resp.status_code == 422
        expected = (
            "website_review ordered 3s, but the shortest possible review is 7.2s "
            "(title_card intro + 2 cards, 3s each)"
        )
        error = resp.json()["error"]
        assert error["message"] == expected
        assert error["details"] == [{"loc": ["body"], "msg": expected, "type": "value_error"}]
        assert "[redacted]" not in resp.text

    async def test_validation_projection_caps_error_count(
        self,
        settings,
        hello_router,
        caplog,
    ) -> None:
        """A hostile body with hundreds of bad fields must not flood the log."""
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        flood = {f"junk_{i}": i for i in range(_MAX_LOGGED_VALIDATION_ERRORS * 3)}
        with caplog.at_level(logging.WARNING, logger=_ENVELOPE_LOGGER):
            async with await _client(app) as client:
                resp = await client.post("/strict", json=flood)
        assert resp.status_code == 422
        warnings = _envelope_warnings(caplog)
        assert len(warnings) == 1
        # The projection is the last %s arg of the log call.
        args = warnings[0].args
        assert isinstance(args, tuple)
        projection = args[-1]
        assert isinstance(projection, list)
        assert len(projection) == _MAX_LOGGED_VALIDATION_ERRORS

    async def test_on_startup_hook_runs(self, settings, hello_router) -> None:
        marks: list[str] = []

        async def hook(app) -> None:
            marks.append("started")

        app = create_app(
            settings,
            routers=[hello_router],
            on_startup=[hook],
            auth_enabled=False,
        )
        # httpx ASGITransport does not drive the lifespan ASGI protocol;
        # invoke the lifespan context manager directly. This is the same
        # path FastAPI takes in production via uvicorn.
        async with app.router.lifespan_context(app), await _client(app) as client:
            resp = await client.get("/hello")
        assert resp.status_code == 200
        assert marks == ["started"]

    async def test_on_shutdown_hook_runs(self, settings, hello_router) -> None:
        marks: list[str] = []

        async def hook(app) -> None:
            marks.append("stopped")

        app = create_app(
            settings,
            routers=[hello_router],
            on_shutdown=[hook],
            auth_enabled=False,
        )
        async with app.router.lifespan_context(app), await _client(app) as client:
            resp = await client.get("/hello")
            assert resp.status_code == 200
        # __aexit__ of lifespan_context runs on_shutdown hooks.
        assert marks == ["stopped"]

    async def test_custom_error_handler_overrides_envelope(
        self,
        settings,
        hello_router,
    ) -> None:
        from fastapi.responses import JSONResponse

        async def custom_500(request, exc) -> JSONResponse:
            return JSONResponse({"shape": "custom"}, status_code=500)

        app = create_app(
            settings,
            routers=[hello_router],
            error_handlers={Exception: custom_500},
            auth_enabled=False,
        )
        async with await _client(app) as client:
            resp = await client.get("/boom")
        assert resp.status_code == 500
        assert resp.json() == {"shape": "custom"}

    async def test_title_defaults_from_settings_class_name(
        self,
        hello_router,
    ) -> None:
        class MyServiceSettings(APISettings):
            pass

        app = create_app(MyServiceSettings(port=8081), routers=[hello_router], auth_enabled=False)
        assert app.title == "MyService"

    async def test_accepts_composite_service_settings(self, hello_router) -> None:
        """A composite Settings (the fleet norm) composing APISettings as `.api`
        must work — CORS is read from `settings.api.cors_*`, not the top level."""
        from pydantic import Field
        from pydantic_settings import BaseSettings

        class Settings(BaseSettings):
            app_name: str = "Composite"
            auth: AuthSettings = Field(default_factory=AuthSettings)
            api: APISettings = Field(
                default_factory=lambda: APISettings(
                    port=8080, host="127.0.0.1", cors_origins=["https://svc.example"]
                )
            )

        app = create_app(Settings(), routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.options(
                "/hello",
                headers={
                    "Origin": "https://svc.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://svc.example"

    async def test_cors_preflight_answered(self, hello_router) -> None:
        settings = APISettings(port=8080, host="127.0.0.1", cors_origins=["https://app.example"])
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        async with await _client(app) as client:
            resp = await client.options(
                "/hello",
                headers={
                    "Origin": "https://app.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
        # CORS is outermost, so the preflight is answered before any auth.
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://app.example"

    async def test_cors_can_be_disabled(self, settings, hello_router) -> None:
        app = create_app(settings, routers=[hello_router], auth_enabled=False, cors_enabled=False)
        assert not any(
            m.cls is CORSMiddleware
            for m in app.user_middleware  # type: ignore[attr-defined]
        )

    async def test_default_operation_id_is_tag_name(self, hello_router) -> None:
        settings = APISettings(port=8080, host="127.0.0.1")
        app = create_app(settings, routers=[hello_router], auth_enabled=False)
        schema = app.openapi()
        # /hello has no explicit tag -> "default_hello"
        assert schema["paths"]["/hello"]["get"]["operationId"] == "default_hello"

    async def test_license_openapi_marks_protected_and_public_ops(self, hello_router) -> None:
        health = APIRouter()

        @health.get("/health", operation_id="health_check", tags=["health"])
        async def health_check() -> dict[str, str]:
            return {"status": "ok"}

        settings = APISettings(port=8080, host="127.0.0.1")
        app = create_app(settings, routers=[hello_router, health], auth_enabled=False)
        schema = app.openapi()

        # BearerAuth scheme + TenantIdHeader parameter are advertised.
        assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
        assert "TenantIdHeader" in schema["components"]["parameters"]

        # Non-public op requires BearerAuth; public /health does not.
        assert schema["paths"]["/hello"]["get"]["security"] == [{"BearerAuth": []}]
        assert "security" not in schema["paths"]["/health"]["get"]

    async def test_license_openapi_can_be_disabled(self, settings, hello_router) -> None:
        app = create_app(
            settings, routers=[hello_router], auth_enabled=False, license_openapi=False
        )
        schema = app.openapi()
        assert "BearerAuth" not in schema.get("components", {}).get("securitySchemes", {})

    async def test_auth_public_endpoints_bypass_enforcement(self, hello_router) -> None:
        """A path in ``auth_public_endpoints`` serves without a license token
        (the OAuth-callback hatch), while normal routes still 401. Replaces the
        per-service AuthMiddleware subclass."""
        from unittest.mock import patch

        callback = APIRouter()

        @callback.get("/api/auth/callback", operation_id="oauth_callback")
        async def cb() -> dict[str, bool]:
            return {"ok": True}

        settings = APISettings(port=8080, host="127.0.0.1")
        app = create_app(
            settings,
            routers=[hello_router, callback],
            auth_public_endpoints={"/api/auth/callback"},
        )
        # _is_trusted_network→False so the loopback test client cannot fail open.
        with patch("w2v_common.auth.middleware._is_trusted_network", return_value=False):
            async with await _client(app) as client:
                public = await client.get("/api/auth/callback")
                protected = await client.get("/hello")
        assert public.status_code == 200
        assert public.json() == {"ok": True}
        assert protected.status_code == 401

    async def test_auth_401_carries_request_id(self, hello_router) -> None:
        """Regression (RequestId reorder): a 401 short-circuited by AuthMiddleware
        must still carry X-Request-Id. Before RequestId was moved OUTER of Auth,
        the auth-rejection path never reached RequestIdMiddleware, so 401s had no
        request-id header and their logs were uncorrelated."""
        from unittest.mock import patch

        settings = APISettings(port=8080, host="127.0.0.1")
        app = create_app(settings, routers=[hello_router])
        with patch("w2v_common.auth.middleware._is_trusted_network", return_value=False):
            async with await _client(app) as client:
                # No inbound X-Request-Id → the middleware must GENERATE one and
                # still stamp it on the 401 that Auth returns.
                protected = await client.get("/hello")
        assert protected.status_code == 401
        assert "x-request-id" in {k.lower() for k in protected.headers}
        assert protected.headers["x-request-id"]

    async def test_auth_public_endpoints_marked_public_in_openapi(self, hello_router) -> None:
        """advertised == enforced: the extra public path also loses its
        ``security`` requirement in the generated schema."""
        callback = APIRouter()

        @callback.get("/api/auth/callback", operation_id="oauth_callback")
        async def cb() -> dict[str, bool]:
            return {"ok": True}

        settings = APISettings(port=8080, host="127.0.0.1")
        app = create_app(
            settings,
            routers=[hello_router, callback],
            auth_enabled=False,
            auth_public_endpoints={"/api/auth/callback"},
        )
        schema = app.openapi()
        assert "security" not in schema["paths"]["/api/auth/callback"]["get"]
        # A normal op is still protected.
        assert schema["paths"]["/hello"]["get"]["security"] == [{"BearerAuth": []}]


class TestMakeErrorPayload:
    def test_payload_shape(self) -> None:
        # Explicit timestamp so the dict compares deterministically.
        payload = make_error_payload(
            code="HTTP_404",
            message="missing",
            status=404,
            request_id="abc",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert payload == {
            "error": {
                "code": "HTTP_404",
                "message": "missing",
                "status": 404,
                "request_id": "abc",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        }

    def test_details_included_only_when_present(self) -> None:
        without = make_error_payload(code="HTTP_500", message="x", status=500, request_id="r")
        assert "details" not in without["error"]
        with_details = make_error_payload(
            code="VALIDATION_ERROR",
            message="bad input",
            status=422,
            request_id="abc",
            details="field email required",
        )
        assert with_details["error"]["details"] == "field email required"

    def test_timestamp_autogenerated_when_omitted(self) -> None:
        payload = make_error_payload(code="HTTP_500", message="x", status=500, request_id="r")
        # ISO-8601 UTC, e.g. 2026-05-25T12:34:56Z
        assert payload["error"]["timestamp"].endswith("Z")
