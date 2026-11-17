"""Settings for downstream service authentication.

Peer-service URLs are resolved via :mod:`w2v_common.discovery` — do NOT add
per-peer URL fields here or in subclasses. This class only carries shared
auth material (license token) read from the ``DOWNSTREAM_`` env prefix.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DownstreamSettings(BaseSettings):
    """Shared auth configuration for outbound calls to peer services."""

    license_token: str = Field(
        default="",
        description="License token used when calling peer services (e.g. modality_gateway).",
    )

    model_config = SettingsConfigDict(
        env_prefix="DOWNSTREAM_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )
