"""Tests for w2v_common.discovery."""

from __future__ import annotations

import pytest

from w2v_common.discovery import list_services, service_port, service_url


class TestServiceUrl:
    """Exercise service_url() resolution."""

    def test_in_cluster_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("W2V_LOCAL", raising=False)
        monkeypatch.delenv("W2V_ANALYSIS_URL", raising=False)
        assert service_url("analysis") == "http://analysis-service.w2v:6303"

    def test_html_render_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("W2V_LOCAL", raising=False)
        monkeypatch.delenv("W2V_HTML_RENDER_URL", raising=False)
        assert service_url("html_render") == "http://html-render-service.w2v:6319"

    def test_kag_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KAG is retired (W0b): its name no longer resolves, even with an override."""
        monkeypatch.delenv("W2V_LOCAL", raising=False)
        monkeypatch.delenv("W2V_KAG_URL", raising=False)
        with pytest.raises(KeyError, match="Unknown peer service"):
            service_url("kag")

    def test_local_mode_uses_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("W2V_LOCAL", "1")
        monkeypatch.delenv("W2V_ANALYSIS_URL", raising=False)
        assert service_url("analysis") == "http://localhost:6303"

    def test_explicit_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("W2V_LOCAL", "1")
        monkeypatch.setenv("W2V_ANALYSIS_URL", "http://staging.example.com:9999")
        assert service_url("analysis") == "http://staging.example.com:9999"

    def test_unknown_service_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown peer service"):
            service_url("nope")

    def test_platform_services_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """worker_management/artifacts/licenses live in their own namespaces."""
        monkeypatch.delenv("W2V_LOCAL", raising=False)
        monkeypatch.delenv("W2V_WORKER_MANAGEMENT_URL", raising=False)
        monkeypatch.delenv("W2V_ARTIFACTS_URL", raising=False)
        monkeypatch.delenv("W2V_LICENSES_URL", raising=False)
        assert service_url("worker_management") == "http://worker-management.workers:8887"
        assert service_url("artifacts") == "http://artifacts.artifacts:8556"
        assert service_url("licenses") == "http://licenses.licenses:8544"


class TestServicePort:
    def test_port_lookup(self) -> None:
        assert service_port("bdd_playwright") == 6313
        assert service_port("webscraper") == 9119

    def test_unknown_port_raises(self) -> None:
        with pytest.raises(KeyError):
            service_port("nope")


class TestListServices:
    def test_includes_all_w2v_services(self) -> None:
        names = list_services()
        # Spot-check a representative set — guards against accidental deletions
        for required in (
            "orchestrator",
            "analysis",
            "html_render",
            "bdd_playwright",
            "frontend_test_runner",
            "webscraper",
            "worker_management",
            "artifacts",
            "licenses",
        ):
            assert required in names, f"{required} missing from catalog"

    def test_sorted(self) -> None:
        names = list_services()
        assert names == sorted(names)


class TestDirectScrapeAndMediaPeers:
    """w2v calls webscraper and media-discovery DIRECTLY (KAG removal, 2026-10-17).

    KAG was only a gateway in front of them. discovery-engine and
    scientific-processor are still DENY'd and scaled to 0, so no peer may
    address them; the tripwire for those two stays.
    """

    FORBIDDEN_HOSTS = ("discovery-engine", "scientific-processor")
    FORBIDDEN_PORTS = frozenset({8807, 8808})

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("W2V_LOCAL", "W2V_WEBSCRAPER_URL", "W2V_MEDIA_DISCOVERY_URL"):
            monkeypatch.delenv(name, raising=False)

    def test_webscraper_resolves_to_its_own_namespace(self) -> None:
        assert service_url("webscraper") == "http://webscraper.webscraper:9119"
        assert service_port("webscraper") == 9119

    def test_media_discovery_resolves_to_its_own_namespace(self) -> None:
        assert service_url("media_discovery") == "http://media-discovery.media-discovery:8809"
        assert service_port("media_discovery") == 8809

    def test_webscraper_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("W2V_WEBSCRAPER_URL", "http://localhost:9119")
        assert service_url("webscraper") == "http://localhost:9119"

    def test_media_discovery_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("W2V_MEDIA_DISCOVERY_URL", "http://localhost:8809")
        assert service_url("media_discovery") == "http://localhost:8809"

    def test_w2v_local_uses_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("W2V_LOCAL", "1")
        assert service_url("webscraper") == "http://localhost:9119"
        assert service_url("media_discovery") == "http://localhost:8809"

    def test_no_peer_resolves_to_a_still_denied_service(self) -> None:
        for name in list_services():
            url = service_url(name)
            for forbidden in self.FORBIDDEN_HOSTS:
                assert forbidden not in url, f"Peer {name!r} addresses {forbidden!r}: {url!r}"

    def test_no_peer_uses_a_still_denied_port(self) -> None:
        for name in list_services():
            assert service_port(name) not in self.FORBIDDEN_PORTS, name

    def test_kag_gateway_is_gone(self) -> None:
        """W0b: no peer is named, hosted or ported like the retired KAG gateway."""
        assert "kag" not in list_services()
        for name in list_services():
            assert "kag" not in service_url(name), name
            assert service_port(name) != 8900, name
        with pytest.raises(KeyError):
            service_port("kag")


class TestCatalogInvariants:
    """Per-entry sanity: non-empty names, sensible ports."""

    def test_all_ports_in_valid_range(self) -> None:
        for name in list_services():
            port = service_port(name)
            assert 1 <= port <= 65535, f"{name} has invalid port {port}"

    def test_all_names_are_lowercase_underscore(self) -> None:
        for name in list_services():
            assert name == name.lower()
            assert " " not in name
            assert "-" not in name, f"Use underscore in logical names: {name!r} has a dash"
