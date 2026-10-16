import pytest

from app.data.provider import MarketDataProvider


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        MarketDataProvider()
