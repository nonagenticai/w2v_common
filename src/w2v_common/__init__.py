"""w2v_common — shared library for w2v microservices."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("w2v-common")
except PackageNotFoundError:
    __version__ = "unknown"
