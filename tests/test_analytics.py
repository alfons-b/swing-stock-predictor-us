import pandas as pd

from app.features.relative_strength import add_relative_strength
from app.features.sector import sector_strength


def test_relative_strength():
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    stock = pd.DataFrame({"date": dates, "close": range(100, 130)})
    ihsg = pd.DataFrame({"date": dates, "close": range(100, 130)})
    out = add_relative_strength(stock, ihsg)
    assert "rs_vs_ihsg_20d" in out.columns
    assert len(out) == 30


def test_sector_strength():
    dates = pd.date_range("2025-01-01", periods=70, freq="B")
    frames = {
        "Technology": pd.DataFrame({
            "date": dates,
            "close": range(100, 170),
            "volume": [1000] * 70,
        }),
        "Finance": pd.DataFrame({
            "date": dates,
            "close": range(100, 170),
            "volume": [1000] * 70,
        }),
    }
    out = sector_strength(frames)
    assert len(out) == 2
    assert "strength" in out.columns
    assert out["strength"].between(0, 100).all()
