# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v6.0.0

    High-frequency market microstructure scalping system.
    Uses cutting-edge microstructure signals tuned for 1-minute bars on Kraken.

    Score: 0.0 – 1.0 across three equal signals (~0.33 each).
      >= 0.40 → entry (2/3 signals firing)
      >= 0.67 → high-conviction entry (all 3 signals) → maximum position size
    """

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Primary entry: returns (score, components_dict)
    # ------------------------------------------------------------------
    def calculate_scalp_score(self, crypto):
        """
        Calculate the aggregate scalp score using microstructure signals.

        Signals
        -------
        1. Order Book Imbalance (OBI): (bid_size - ask_size) / (bid_size + ask_size) > 0.5
        2. Volume Ignition: current 1-min volume > 3× the 20-bar moving average volume
        3. Micro-Trend: 1-min Close > 5-period 1-min EMA

        Returns
        -------
        (score, components) where score ∈ [0, 1] and components maps each
        signal name to its individual contribution.
        """
        components = {
            'obi':            0.0,
            'vol_ignition':   0.0,
            'micro_trend':    0.0,
        }

        try:
            # ----------------------------------------------------------
            # Signal 1: Order Book Imbalance (OBI)
            # Uses real bid/ask sizes captured from QuoteBars.
            # OBI = (bid_size - ask_size) / (bid_size + ask_size)
            # Strong buy pressure when OBI > 0.5 (bid wall dominates).
            # ----------------------------------------------------------
            bid_size = crypto.get('bid_size', 0.0)
            ask_size = crypto.get('ask_size', 0.0)
            total_size = bid_size + ask_size
            if total_size > 0:
                obi = (bid_size - ask_size) / total_size
                if obi > 0.5:
                    components['obi'] = 0.33
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"OBI Signal: obi={obi:.3f} bid={bid_size:.2f} ask={ask_size:.2f}")
                elif obi > 0.2:
                    # Partial credit for mild buy-side imbalance
                    components['obi'] = 0.15

            # ----------------------------------------------------------
            # Signal 2: Volume Ignition
            # Current 1-minute volume > 3× the 20-bar moving-average volume.
            # Signals sudden surge of market participation (ignition bar).
            # ----------------------------------------------------------
            if len(crypto['volume']) >= 20:
                volumes = list(crypto['volume'])
                vol_ma_20 = np.mean(volumes[-20:])
                current_vol = volumes[-1]
                if vol_ma_20 > 0:
                    ratio = current_vol / vol_ma_20
                    if ratio >= 3.0:
                        components['vol_ignition'] = 0.33
                        if not self.algo.IsWarmingUp:
                            self.algo.Debug(
                                f"Volume Ignition: vol={current_vol:.2f} ma20={vol_ma_20:.2f} ratio={ratio:.1f}x")
                    elif ratio >= 2.0:
                        # Partial credit for a meaningful (2×) volume spike
                        components['vol_ignition'] = 0.17

            # ----------------------------------------------------------
            # Signal 3: Micro-Trend
            # 1-minute Close > 5-period 1-minute EMA → short-term uptrend.
            # ----------------------------------------------------------
            if crypto['ema_5'].IsReady and len(crypto['prices']) >= 1:
                price = crypto['prices'][-1]
                ema5 = crypto['ema_5'].Current.Value
                if price > ema5:
                    components['micro_trend'] = 0.33

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_scalp_score error: {e}")

        score = sum(components.values())
        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Aggressive position sizing for compound growth on a $20 account.

        Returns 90–99% of available capital depending on conviction.
        Bear regime: base reduced by 20%.
        """
        if score >= 0.67:
            # All three signals firing – maximum conviction
            size = 0.99
        elif score >= self.algo.high_conviction_threshold:
            # 2+ signals: high conviction
            size = 0.95
        elif score >= threshold:
            # Entry threshold met: still aggressive
            size = 0.90
        else:
            size = 0.70

        if self.algo.market_regime == "bear":
            size *= 0.80

        kelly = self.algo._kelly_fraction()
        return size * kelly

