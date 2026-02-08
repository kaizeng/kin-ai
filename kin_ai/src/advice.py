from __future__ import annotations

from typing import Dict, List, Tuple


def _classify_assets(tickers: List[str]) -> Tuple[List[str], List[str], List[str]]:
    bonds = {"AGG", "BND", "IEF", "TLT", "SHY", "LQD"}
    gold = {"GLD", "IAU"}
    equities = [t for t in tickers if t not in bonds and t not in gold]
    bond_list = [t for t in tickers if t in bonds]
    gold_list = [t for t in tickers if t in gold]
    return equities, bond_list, gold_list


def generate_advice(
    trend_regime: str,
    cycle: str,
    base_weights: Dict[str, float],
    max_tilt: float = 0.10,
) -> Dict[str, object]:
    """Generate allocation advice based on regime and cycle."""
    tickers = list(base_weights.keys())
    equities, bonds, gold = _classify_assets(tickers)

    weights = base_weights.copy()

    if trend_regime == "bull":
        tilt_up = equities
        tilt_down = bonds + gold
        action = "Risk-on tilt"
    else:
        tilt_up = bonds + gold
        tilt_down = equities
        action = "Risk-off tilt"

    if tilt_up and tilt_down:
        per_up = max_tilt / len(tilt_up)
        per_down = max_tilt / len(tilt_down)
        for t in tilt_up:
            weights[t] = weights.get(t, 0.0) + per_up
        for t in tilt_down:
            weights[t] = max(0.0, weights.get(t, 0.0) - per_down)

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return {
        "trend_regime": trend_regime,
        "cycle": cycle,
        "action": action,
        "suggested_weights": weights,
    }
