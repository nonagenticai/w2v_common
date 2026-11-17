"""Tests for w2v_common.clients.url_graph.URLGraph."""

from __future__ import annotations

from w2v_common.clients.url_graph import URLGraph


class TestNormalizeStripsFragment:
    def test_strips_fragment(self) -> None:
        graph = URLGraph(domain="example.com")
        result = graph.normalize("https://example.com/page#section")
        assert result is not None
        assert "#" not in result
        assert result == "https://example.com/page"


class TestNormalizeRejectsExternal:
    def test_rejects_external_domain(self) -> None:
        graph = URLGraph(domain="example.com")
        result = graph.normalize("https://other.com/page")
        assert result is None

    def test_accepts_same_domain(self) -> None:
        graph = URLGraph(domain="example.com")
        result = graph.normalize("https://example.com/page")
        assert result is not None
        assert "example.com" in result


class TestAddPageReturnsNewUrls:
    def test_returns_new_urls(self) -> None:
        graph = URLGraph(domain="example.com")
        new = graph.add_page(
            "https://example.com",
            [
                "https://example.com/about",
                "https://example.com/contact",
            ],
        )
        assert len(new) == 2
        assert any("/about" in u for u in new)
        assert any("/contact" in u for u in new)


class TestAddPageDeduplicates:
    def test_deduplicates(self) -> None:
        graph = URLGraph(domain="example.com")
        # First call discovers /about
        new1 = graph.add_page(
            "https://example.com",
            ["https://example.com/about"],
        )
        assert len(new1) == 1

        # Second call with same link should return nothing new
        new2 = graph.add_page(
            "https://example.com/about",
            ["https://example.com/about"],
        )
        assert len(new2) == 0


class TestShouldExclude:
    def test_matches_exclusion_pattern(self) -> None:
        graph = URLGraph(domain="example.com")
        assert graph.should_exclude("https://example.com/admin/settings", [r"/admin/"]) is True

    def test_no_match_returns_false(self) -> None:
        graph = URLGraph(domain="example.com")
        assert graph.should_exclude("https://example.com/about", [r"/admin/"]) is False

    def test_empty_patterns(self) -> None:
        graph = URLGraph(domain="example.com")
        assert graph.should_exclude("https://example.com/anything", []) is False


class TestToDict:
    def test_to_dict_returns_sorted_links(self) -> None:
        graph = URLGraph(domain="example.com")
        graph.add_page(
            "https://example.com/",
            [
                "https://example.com/z-page",
                "https://example.com/a-page",
            ],
        )
        result = graph.to_dict()
        # normalize() converts "https://example.com/" path to "/" which stays as "/"
        normalized_key = "https://example.com/"
        assert normalized_key in result
        links = result[normalized_key]
        assert links == sorted(links)
        assert len(links) == 2

    def test_to_dict_empty_graph(self) -> None:
        graph = URLGraph(domain="example.com")
        assert graph.to_dict() == {}
