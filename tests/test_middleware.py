"""Tests for w2v_common.middleware (request_id and security)."""

import uuid

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from w2v_common.middleware.request_id import RequestIdMiddleware
from w2v_common.middleware.security import SecurityHeadersMiddleware


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with both middleware layers."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/ctxvar")
    async def ctxvar_endpoint():
        from w2v_common.observability.request_context import get_request_id

        # The handler reads the ContextVar the middleware should have bound.
        return {"request_id": get_request_id()}

    return app


class TestRequestIdMiddleware:
    """Tests for RequestIdMiddleware."""

    async def test_generates_request_id_when_missing(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test")
            assert resp.status_code == 200
            request_id = resp.headers.get("X-Request-Id")
            assert request_id is not None
            # Verify it's a valid UUID
            uuid.UUID(request_id)

    async def test_preserves_existing_request_id(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        custom_id = "my-custom-request-id-123"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test", headers={"x-request-id": custom_id})
            assert resp.status_code == 200
            assert resp.headers.get("X-Request-Id") == custom_id

    async def test_binds_request_id_contextvar(self):
        """The middleware must bind the request-id ContextVar (not just
        request.state) so the log formatter + trace_event_hook pick it up."""
        app = _make_app()
        transport = ASGITransport(app=app)
        custom_id = "ctxvar-trace-42"
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ctxvar", headers={"x-request-id": custom_id})
            assert resp.status_code == 200
            assert resp.json()["request_id"] == custom_id


class TestSecurityHeadersMiddleware:
    """Tests for SecurityHeadersMiddleware."""

    async def test_security_headers_added(self):
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test")
            assert resp.status_code == 200
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert "max-age=63072000" in resp.headers.get("Strict-Transport-Security", "")
            assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
            assert "camera=()" in resp.headers.get("Permissions-Policy", "")
            assert resp.headers.get("X-XSS-Protection") == "0"
