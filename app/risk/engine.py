from dataclasses import dataclass
import math


@dataclass
class RiskPlan:
    decision: str
    entry_low: float | None
    entry_ideal: float | None
    entry_high: float | None
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    risk_reward: float | None
    risk_amount: float
    position_size: int
    number_of_lots: int
    estimated_loss: float


def calculate_risk(
    price: float,
    atr: float,
    support: float | None,
    resistance: float | None,
    probability: float,
    expected_return: float,
    capital: float,
    risk_per_trade: float = 0.01,
    minimum_rr: float = 1.5,
    board_lot: int = 1,
) -> RiskPlan:
    if not price or not atr or capital <= 0:
        return RiskPlan("NO TRADE", None,None,None,None,None,None,None,0,0,0,0)

    entry_low = max(0, price - 0.5 * atr)
    entry_ideal = price
    entry_high = price + 0.5 * atr

    support_level = support if support and support < price else price - 1.5 * atr
    stop = min(support_level, price - 1.0 * atr)
    if stop <= 0 or stop >= price:
        return RiskPlan("NO TRADE", entry_low,entry_ideal,entry_high,None,None,None,None,0,0,0,0)

    risk_per_share = price - stop
    tp1 = resistance if resistance and resistance > price else price + 1.5 * risk_per_share
    tp2 = max(tp1, price + 2.5 * risk_per_share)
    rr = (tp1 - price) / risk_per_share if risk_per_share else 0

    risk_amount = capital * risk_per_trade
    shares = math.floor(risk_amount / risk_per_share)
    lots = shares // board_lot
    shares = lots * board_lot
    estimated_loss = shares * risk_per_share

    decision = "BUY"
    if probability < 0.65 or expected_return <= 0 or rr < minimum_rr:
        decision = "WATCHLIST"
    if probability < 0.55 or rr < 1.0:
        decision = "NO TRADE"

    return RiskPlan(
        decision, entry_low, entry_ideal, entry_high, stop, tp1, tp2, rr,
        risk_amount, shares, lots, estimated_loss
    )
