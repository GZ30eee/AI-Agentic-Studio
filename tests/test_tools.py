import pytest
from app import DuckDuckGoTool, WebScraperTool, NewsAPITool, RAGTool
from unittest.mock import patch

def test_duckduckgo_tool():
    tool = DuckDuckGoTool()
    with patch("app.DuckDuckGoSearchRun") as mock_search:
        mock_search.return_value.run.return_value = "mock result"
        result = tool._run("test")
        assert result == "mock result"

def test_webscraper_tool():
    tool = WebScraperTool()
    with patch("app.requests.get") as mock_get:
        mock_get.return_value.text = "<html><body>test</body></html>"
        result = tool._run("http://example.com")
        assert "test" in result
