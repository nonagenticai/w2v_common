"""User model for authentication."""

import uuid

from pydantic import BaseModel, Field

from ..tenant import MissingTenantError


class User(BaseModel):
    """User model extracted from license token introspection or Keycloak JWT.

    ``tenant_id`` is a real ``uuid.UUID`` — never a slug, never a sentinel.
    Pydantic parses a UUID-shaped string into a ``UUID`` on construction and
    REJECTS anything else, so an invalid tenant fails at the boundary rather
    than surviving as a plausible-looking string.
    """

    sub: str = Field(..., description="User/Tenant ID")
    preferred_username: str = Field(default="anonymous", description="Username")
    tenant_id: uuid.UUID | None = None
    email: str | None = Field(default=None, description="User email (set for JWT-authed users)")
    email_verified: bool = Field(
        default=False,
        description=(
            "Keycloak ``email_verified`` claim. True ONLY when the verified JWT "
            "carried a literal boolean ``true`` — absent, non-boolean or any "
            "non-JWT lane is False, so an invite/email match can never be "
            "granted on an unproven address."
        ),
    )
    api_key_id: str | None = Field(
        default=None,
        description="Id of the tenant API key (w2v_sk_) that authenticated this caller",
    )
    roles: list[str] = Field(
        default_factory=list, description="Realm roles (set for JWT-authed users)"
    )
    realm: str | None = Field(
        default=None,
        description=(
            "Keycloak realm of the VERIFIED token issuer (JWT lane only; None for "
            "licence / API-key / trusted-network lanes). Realm-per-app P0: which "
            "app's user directory the person belongs to."
        ),
    )

    @property
    def effective_tenant_id(self) -> uuid.UUID:
        """The requestor's real tenant. Raises when there isn't one.

        This used to be ``self.tenant_id or self.sub`` — a SELF-NAMED tenant
        fallback that minted a tenant equal to the subject. That is a sentinel
        by another name: it produces a tenant that was never registered, and
        (because ``sub`` is a free-form string) it also produced non-UUID
        tenants. Both middleware lanes now guarantee a real tenant before a
        ``User`` is built, so the absence of one is a bug and says so.

        One exception, opt-in: with ``AUTH_JWT_TENANT_REQUIRED=false`` the JWT
        lane builds a tenant-LESS ``User`` and the app's membership middleware
        is responsible for resolving the tenant. Until it has, this still
        raises — that is the point.
        """
        if self.tenant_id is None:
            raise MissingTenantError(
                f"User(sub={self.sub!r}) has no tenant_id; refusing to fall back to sub"
            )
        return self.tenant_id
