"""Tests for w2v_common.auth.models."""

import uuid

import pytest

from w2v_common.auth.models import User
from w2v_common.tenant import MissingTenantError


class TestUser:
    """Tests for the User model."""

    def test_user_creation_with_all_fields(self):
        user = User(
            sub="user-123",
            preferred_username="alice",
            # deliberate: a valid UUID *string* must still coerce (pydantic lane)
            tenant_id="0195f48b-0000-7000-8000-000000000101",  # ty: ignore[invalid-argument-type]
        )
        assert user.sub == "user-123"
        assert user.preferred_username == "alice"
        assert user.tenant_id == uuid.UUID("0195f48b-0000-7000-8000-000000000101")

    def test_user_creation_minimal(self):
        user = User(sub="user-456")
        assert user.sub == "user-456"
        assert user.preferred_username == "anonymous"
        assert user.tenant_id is None
        assert user.email is None
        assert user.roles == []

    def test_user_creation_with_jwt_fields(self):
        user = User(
            sub="kc-user",
            preferred_username="alice",
            tenant_id=uuid.UUID("0195f48b-0000-7000-8000-000000000101"),
            email="alice@example.test",
            roles=["admin", "user"],
        )
        assert user.email == "alice@example.test"
        assert user.roles == ["admin", "user"]

    def test_effective_tenant_id_with_tenant_id(self):
        user = User(sub="user-123", tenant_id=uuid.UUID("0195f48b-0000-7000-8000-000000000101"))
        assert user.effective_tenant_id == uuid.UUID("0195f48b-0000-7000-8000-000000000101")

    def test_effective_tenant_id_without_tenant_id_raises(self):
        """No self-naming. ``tenant_id or sub`` minted a tenant that never existed.

        ``sub`` is a Keycloak subject, not a tenant. Falling back to it produced
        an unregistered, non-UUID tenant that then got written to real rows.
        """
        user = User(sub="user-123")
        with pytest.raises(MissingTenantError):
            _ = user.effective_tenant_id

    def test_non_uuid_tenant_id_is_rejected_at_construction(self):
        """A slug tenant can no longer reach a ``User`` at all."""
        for bad in ("", "   ", "n8n-dev", "anonymous", "user-123"):
            with pytest.raises(ValueError):
                User(sub="user-123", tenant_id=bad)  # ty: ignore[invalid-argument-type]

    def test_user_model_dump(self):
        user = User(
            sub="user-123",
            preferred_username="bob",
            tenant_id=uuid.UUID("0195f48b-0000-7000-8000-000000000001"),
        )
        data = user.model_dump()
        assert data == {
            "sub": "user-123",
            "preferred_username": "bob",
            "tenant_id": uuid.UUID("0195f48b-0000-7000-8000-000000000001"),
            "email": None,
            "email_verified": False,
            "api_key_id": None,
            "roles": [],
            "realm": None,
        }

    def test_user_model_json_schema(self):
        schema = User.model_json_schema()
        assert "sub" in schema["properties"]
        assert "preferred_username" in schema["properties"]
        assert "tenant_id" in schema["properties"]
        assert "email" in schema["properties"]
        assert "roles" in schema["properties"]
