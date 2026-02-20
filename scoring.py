# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v7.0.0

    High-frequency market microstructure scalping system.
    Uses five signals tuned for 1-minute bars on Kraken.

    Score: 0.0 – 1.0 across five equal signals (0.20 each).
      >= 0.55 → entry (3/5 signals firing)
      >= 0.80 → high-conviction entry (4+/5 signals) → maximum position size
    """

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Primary entry: returns (score, components_dict)
    # ------------------------------------------------------------------
    def calculate_scalp_score(self, crypto):
        """
        Calculate the aggregate scalp score using microstructure signals.

        Signals (0.20 each, max 1.0)
        -------
        1. Order Book Imbalance (OBI): (bid_size - ask_size) / (bid_size + ask_size) > 0.5
        2. Volume Ignition: current 1-min volume > 3× the 20-bar moving average volume
        3. Micro-Trend: 1-min Close > 5-period 1-min EMA
        4. VWAP Filter: Close > rolling 20-bar VWAP (trend direction gate)
        5. EMA Cross: EMA-5 > EMA-12 (short-term trend structure)

        Returns
        -------
        (score, components) where score ∈ [0, 1] and components maps each
        signal name to its individual contribution.
        """
        components = {
            'obi':            0.0,
            'vol_ignition':   0.0,
            'micro_trend':    0.0,
            'vwap':           0.0,
            'ema_cross':      0.0,
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
                    components['obi'] = 0.20
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"OBI Signal: obi={obi:.3f} bid={bid_size:.2f} ask={ask_size:.2f}")
                elif obi > 0.2:
                    # Partial credit for mild buy-side imbalance
                    components['obi'] = 0.10

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
                        components['vol_ignition'] = 0.20
                        if not self.algo.IsWarmingUp:
                            self.algo.Debug(
                                f"Volume Ignition: vol={current_vol:.2f} ma20={vol_ma_20:.2f} ratio={ratio:.1f}x")
                    elif ratio >= 2.0:
                        # Partial credit for a meaningful (2×) volume spike
                        components['vol_ignition'] = 0.10

            # ----------------------------------------------------------
            # Signal 3: Micro-Trend
            # 1-minute Close > 5-period 1-minute EMA → short-term uptrend.
            # ----------------------------------------------------------
            if crypto['ema_5'].IsReady and len(crypto['prices']) >= 1:
                price = crypto['prices'][-1]
                ema5 = crypto['ema_5'].Current.Value
                if price > ema5:
                    components['micro_trend'] = 0.20

            # ----------------------------------------------------------
            # Signal 4: VWAP Filter
            # Close > rolling 20-bar Volume-Weighted Average Price.
            # Ensures we only buy when price is above its anchored fair value,
            # which filters out mean-reversion traps.
            # ----------------------------------------------------------
            if len(crypto['dollar_volume']) >= 20 and len(crypto['volume']) >= 20:
                dv_arr = list(crypto['dollar_volume'])[-20:]
                vol_arr = list(crypto['volume'])[-20:]
                total_vol = sum(vol_arr)
                if total_vol > 0 and len(crypto['prices']) >= 1:
                    vwap = sum(dv_arr) / total_vol
                    price = crypto['prices'][-1]
                    if price > vwap:
                        components['vwap'] = 0.20

            # ----------------------------------------------------------
            # Signal 5: EMA Cross (trend structure confirmation)
            # EMA-5 > EMA-12 confirms the short-term trend is above the
            # medium-term trend, filtering out counter-trend scalps.
            # ----------------------------------------------------------
            if crypto['ema_5'].IsReady and crypto['ema_medium'].IsReady:
                if crypto['ema_5'].Current.Value > crypto['ema_medium'].Current.Value:
                    components['ema_cross'] = 0.20

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_scalp_score error: {e}")

        score = sum(components.values())
        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Conservative fixed-fractional position sizing to protect a $20 account.

        Returns 50–70% of available capital depending on conviction.
        Bear regime: base reduced by 20%.
        """
        if score >= 0.80:
            # 4+ signals firing – maximum conviction
            size = 0.70
        elif score >= self.algo.high_conviction_threshold:
            # 3+ signals: high conviction
            size = 0.60
        elif score >= threshold:
            # Entry threshold met: base allocation
            size = 0.50
        else:
            size = 0.30

        if self.algo.market_regime == "bear":
            size *= 0.80

        kelly = self.algo._kelly_fraction()
        return size * kelly

