from app.risk.engine import calculate_risk


def test_board_lot_position_sizing():
    plan = calculate_risk(
        price=1000,
        atr=30,
        support=950,
        resistance=1100,
        probability=0.8,
        expected_return=0.05,
        capital=10_000_000,
        risk_per_trade=0.01,
        minimum_rr=1.5,
        board_lot=100,
    )
    assert plan.number_of_lots >= 0
    assert plan.position_size % 100 == 0


def test_bad_probability_rejects_trade():
    plan = calculate_risk(
        price=1000,
        atr=30,
        support=950,
        resistance=1100,
        probability=0.40,
        expected_return=0.05,
        capital=10_000_000,
    )
    assert plan.decision == "NO TRADE"
