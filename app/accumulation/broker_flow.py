"""Antarmuka broker summary (opsional).

Broker summary (net beli/jual per kode broker) tidak dipublikasikan bebas di pasar AS (tidak ada padanan broker summary BEI); tersedia
melalui vendor berlisensi. Modul ini hanya mendefinisikan kontrak agar data vendor dapat dipasang tanpa mengubah
scorer. Tanpa provider → status BROKER_SUMMARY_UNAVAILABLE dan tidak memengaruhi skor (bukan nol).

Kontrak data (per emiten × tanggal × broker): ticker, date, broker_code, buy_shares, sell_shares, buy_value,
sell_value, is_foreign_broker (opsional).
"""
from __future__ import annotations

import pandas as pd

from app.config import get, resolve_path


class BrokerSummaryProvider:
    name = "broker_summary_file"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = resolve_path(cfg, str(get(cfg, "broker_summary.dir", "data/raw/broker_summary")))

    def availability(self) -> tuple[str, str]:
        if not get(self.cfg, "broker_summary.enabled", False):
            return "DISABLED", "broker_summary.enabled: false (butuh data berlisensi)"
        files = sorted(self.dir.glob("*.csv")) if self.dir.exists() else []
        return ("AVAILABLE", f"{len(files)} file") if files else ("NOT_CONFIGURED", f"tidak ada CSV di {self.dir}")

    def load(self, start=None, end=None) -> pd.DataFrame:
        if self.availability()[0] != "AVAILABLE":
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(p) for p in sorted(self.dir.glob("*.csv"))], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df


def broker_concentration(df: pd.DataFrame, window: int = 20, top_n: int = 3) -> pd.DataFrame:
    """Porsi net beli top-N broker terhadap volume (indikasi akumulasi terkonsentrasi). Butuh data vendor."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "date", "broker_top_net_ratio"])
    d = df.assign(net=df["buy_shares"] - df["sell_shares"], gross=df["buy_shares"] + df["sell_shares"])
    end = d["date"].max()
    w = d[d["date"] > end - pd.Timedelta(days=int(window * 1.5))]
    agg = w.groupby(["ticker", "broker_code"])["net"].sum().reset_index()
    top = agg.sort_values("net", ascending=False).groupby("ticker").head(top_n).groupby("ticker")["net"].sum()
    vol = w.groupby("ticker")["gross"].sum() / 2
    return pd.DataFrame({"ticker": top.index, "date": end, "broker_top_net_ratio": (top / vol.reindex(top.index)).values})
