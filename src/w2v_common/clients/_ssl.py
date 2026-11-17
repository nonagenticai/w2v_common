"""SSL context utilities for HTTP clients."""

from __future__ import annotations

import os
import ssl
from pathlib import Path


def get_ssl_context() -> ssl.SSLContext | bool:
    """Return an SSL context that trusts both system CAs and SSL_CERT_FILE.

    If SSL_CERT_FILE is set, returns an ssl.SSLContext (which httpx accepts
    via ``verify=``).  Otherwise returns ``True`` so httpx uses its default
    certifi bundle.
    """
    cert_file = os.environ.get("SSL_CERT_FILE")
    if cert_file and Path(cert_file).is_file():
        return ssl.create_default_context()
    return True
