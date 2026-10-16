"""Backtest engine portofolio (M) — event-driven harian, long-only.

Asumsi eksekusi (konservatif, semua tercatat di hasil):
1. Sinyal dari data close hari t. Order BUY LIMIT di `entry_high` berlaku hari t+1..t+entry_valid_days.
   Fill = min(open, entry_high) bila low <= entry_high. Bila open sudah <= stop → order dibatalkan.
2. Stop: bila low <= stop → keluar di min(open, stop) (gap down dihitung penuh).
3. TP1: jual `tp_partial_fraction` di max(open, TP1); stop sisa dinaikkan ke breakeven. TP2: jual sisa.
4. SL & TP di candle yang sama → SL lebih dulu (same_bar_rule=stop_first).
5. Time exit di close setelah `max_holding_days`. Hari suspensi: posisi tidak bisa ditransaksikan.
6. Fee beli/jual + slippage diterapkan di setiap fill. Harga fill dibulatkan ke tick $0,01.
7. Ukuran posisi dihitung ulang dari ekuitas aktual saat fill (risk-based, kelipatan lot).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.backtest.metrics import compute_metrics
from app.utils.market_rules import round_to_tick


@dataclass
class Position:
    ticker: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop: float
    tp1: float
    tp2: float
    initial_stop: float
    setup: str
    remaining: int = 0
    realized: float = 0.0
    fees: float = 0.0
    tp1_done: bool = False
    days_held: int = 0
    exits: list = field(default_factory=list)


class BacktestEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        b = cfg["backtest"]
        self.fee_buy, self.fee_sell = b["TRANSACTION_FEE_BUY"], b["TRANSACTION_FEE_SELL"]
        self.slip = b["SLIPPAGE_BPS"] / 1e4
        self.initial = float(b["initial_capital"])
        st = cfg["strategy"]
        self.max_hold = st["max_holding_days"]
        self.valid_days = st.get("entry_valid_days", 1)
        self.partial = st.get("tp_partial_fraction", 0.5)
        pf = cfg["portfolio"]
        self.max_pos = pf["max_open_positions"]
        self.risk_pt = pf["RISK_PER_TRADE"]
        self.max_pos_pct = pf.get("max_position_pct", 0.2)
        self.lot = pf.get("board_lot", 1)
        self.adv_pct = cfg["liquidity"].get("MAX_POSITION_PCT_OF_ADV", 0.05)
        self.rules = cfg.get("regime", {}).get("rules", {})

    def _buy_px(self, px):
        return round_to_tick(px * (1 + self.slip), "up")

    def _sell_px(self, px):
        return round_to_tick(px * (1 - self.slip), "down")

    def run(self, signals: pd.DataFrame, prices: pd.DataFrame, start=None, end=None) -> dict:
        px = prices[["ticker", "date", "open", "high", "low", "close", "is_suspended"]].copy()
        if start is not None:
            px = px[px["date"] >= pd.Timestamp(start)]
        if end is not None:
            px = px[px["date"] <= pd.Timestamp(end) + pd.Timedelta(days=30)]
        bars = {d: g.set_index("ticker") for d, g in px.groupby("date")}
        dates = sorted(bars)
        last_eval = pd.Timestamp(end) if end is not None else dates[-1]
        sig_by_date = {d: g for d, g in signals.groupby("date")}

        cash = self.initial
        positions: dict[str, Position] = {}
        pending: list[dict] = []
        trades, equity_rows = [], []
        last_close: dict[str, float] = {}

        for d in dates:
            bar = bars[d]
            # ---------- 1. eksekusi order pending (sinyal hari sebelumnya)
            still = []
            for o in pending:
                tk = o["ticker"]
                if tk in positions or tk not in bar.index or bar.at[tk, "is_suspended"]:
                    o["days"] += 1
                    if o["days"] < self.valid_days and tk not in positions:
                        still.append(o)
                    continue
                b = bar.loc[tk]
                if b["open"] <= o["stop_loss"] or b["low"] > o["entry_high"] or len(positions) >= self.max_pos:
                    o["days"] += 1
                    if o["days"] < self.valid_days and b["open"] > o["stop_loss"]:
                        still.append(o)
                    continue
                fill = self._buy_px(min(b["open"], o["entry_high"]))
                eq_now = cash + sum(p.remaining * last_close.get(p.ticker, p.entry_price) for p in positions.values())
                mult = self.rules.get(o["market_regime"], {}).get("risk_mult", 1.0)
                risk_ps = fill - o["stop_loss"]
                if risk_ps <= 0:
                    continue
                shares = min(eq_now * self.risk_pt * mult / risk_ps, eq_now * self.max_pos_pct / fill,
                             self.adv_pct * o["avg_value20"] / fill)
                shares = int(shares // self.lot) * self.lot
                cost = shares * fill * (1 + self.fee_buy)
                if cost > cash:
                    shares = int((cash / (fill * (1 + self.fee_buy))) // self.lot) * self.lot
                    cost = shares * fill * (1 + self.fee_buy)
                if shares < self.lot:
                    continue
                cash -= cost
                positions[tk] = Position(tk, o["date"], d, fill, shares, o["stop_loss"], o["take_profit_1"],
                                         o["take_profit_2"], o["stop_loss"], o["setup_type"], remaining=shares,
                                         fees=shares * fill * self.fee_buy)
            pending = still

            # ---------- 2. kelola posisi terbuka
            for tk in list(positions):
                p = positions[tk]
                if tk not in bar.index or bar.at[tk, "is_suspended"]:
                    continue
                b = bar.loc[tk]
                p.days_held += 1 if d > p.entry_date else 0

                def sell(qty, raw_px, reason):
                    nonlocal cash
                    sp = self._sell_px(raw_px)
                    gross = qty * sp
                    fee = gross * self.fee_sell
                    cash += gross - fee
                    p.realized += gross - fee
                    p.fees += fee
                    p.remaining -= qty
                    p.exits.append((d, qty, sp, reason))

                # hari entry: hanya cek stop bila low setelah fill bisa diketahui → konservatif: cek stop
                stop_hit = b["low"] <= p.stop
                tp1_hit = (not p.tp1_done) and b["high"] >= p.tp1
                tp2_hit = b["high"] >= p.tp2
                if stop_hit:  # stop_first
                    sell(p.remaining, min(b["open"], p.stop), "STOP" if not p.tp1_done else "BREAKEVEN_STOP")
                else:
                    if tp1_hit and d > p.entry_date:
                        q = int((p.shares * self.partial) // self.lot) * self.lot
                        q = q if 0 < q < p.remaining else p.remaining
                        sell(q, max(b["open"], p.tp1), "TP1")
                        p.tp1_done = True
                        p.stop = max(p.stop, p.entry_price)
                    if p.remaining > 0 and tp2_hit and d > p.entry_date:
                        sell(p.remaining, max(b["open"], p.tp2), "TP2")
                    if p.remaining > 0 and p.days_held >= self.max_hold:
                        sell(p.remaining, b["close"], "TIME_EXIT")
                if p.remaining <= 0:
                    cost = p.shares * p.entry_price * (1 + self.fee_buy)
                    net = p.realized - cost
                    trades.append({"ticker": tk, "setup": p.setup, "signal_date": p.signal_date, "entry_date": p.entry_date,
                                   "exit_date": d, "entry_price": p.entry_price, "shares": p.shares,
                                   "entry_value": p.shares * p.entry_price, "initial_stop": p.initial_stop,
                                   "tp1": p.tp1, "tp2": p.tp2, "avg_exit_price": sum(q * x for _, q, x, _ in p.exits) / p.shares,
                                   "exit_reason": "+".join(r for *_, r in p.exits), "holding_days": max(1, p.days_held),
                                   "fees": p.fees, "net_pnl": net, "net_return": net / cost,
                                   "r_multiple": net / (p.shares * (p.entry_price - p.initial_stop))})
                    del positions[tk]

            for tk in bar.index:
                last_close[tk] = bar.at[tk, "close"]
            # ---------- 3. sinyal baru (hanya sampai akhir periode evaluasi)
            if d <= last_eval and d in sig_by_date:
                for _, s in sig_by_date[d].iterrows():
                    if s["ticker"] not in positions and all(o["ticker"] != s["ticker"] for o in pending):
                        pending.append({**s.to_dict(), "days": 0})
            mv = sum(p.remaining * last_close.get(p.ticker, p.entry_price) for p in positions.values())
            equity_rows.append({"date": d, "cash": cash, "market_value": mv, "equity": cash + mv,
                                "n_positions": len(positions), "exposure": mv / (cash + mv) if cash + mv > 0 else 0})
            if d > last_eval and not positions:
                break

        equity = pd.DataFrame(equity_rows)
        tr = pd.DataFrame(trades)
        metrics = compute_metrics(equity, tr, self.initial, self.cfg["backtest"].get("risk_free_rate", 0.0))
        metrics["assumptions"] = {"fee_buy": self.fee_buy, "fee_sell": self.fee_sell, "slippage_bps": self.slip * 1e4,
                                  "same_bar_rule": "stop_first", "entry": "limit @ entry_high, t+1",
                                  "max_open_positions": self.max_pos, "risk_per_trade": self.risk_pt}
        return {"equity": equity, "trades": tr, "metrics": metrics}


def benchmark_buy_hold(index: pd.DataFrame, start, end, initial: float) -> dict:
    ix = index[(index["date"] >= pd.Timestamp(start)) & (index["date"] <= pd.Timestamp(end))].sort_values("date")
    if len(ix) < 2:
        return {}
    eq = initial * ix["close"] / ix["close"].iloc[0]
    e = pd.DataFrame({"date": ix["date"], "equity": eq.to_numpy(), "exposure": 1.0})
    m = compute_metrics(e, pd.DataFrame(), initial)
    return {k: m[k] for k in ("total_return", "cagr", "max_drawdown", "sharpe")}
