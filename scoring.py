# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v5.0.0

    Replaces the multi-factor OpusScoringEngine with a focused 5-signal
    scalp engine optimised for fast, high-frequency entries and exits.

    Score: 0.0 – 1.0, each of the 5 signals contributes exactly 0.20.
      >= 0.40 → entry (2+ signals firing)
      >= 0.60 → high-conviction entry (3+ signals) → full position size
    """

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Primary entry: returns (score, components_dict)
    # ------------------------------------------------------------------
    def calculate_scalp_score(self, crypto):
        """
        Calculate the aggregate scalp score for a crypto asset.

        Returns
        -------
        (score, components) where score ∈ [0, 1] and components is a dict
        mapping each signal name to its individual contribution (0 or 0.20).
        """
        components = {
            'rsi_divergence':       0.0,
            'volume_bb_compression': 0.0,
            'ema_crossover':        0.0,
            'vwap_reclaim':         0.0,
            'orderbook_proxy':      0.0,
        }

        try:
            # ----------------------------------------------------------
            # Signal 1: RSI Divergence Scalp
            # RSI(7) recovering from oversold (<35) while price makes a
            # higher low over the last few bars (bullish divergence).
            # ----------------------------------------------------------
            if (crypto['rsi'].IsReady
                    and len(crypto['lows']) >= 6
                    and len(crypto['returns']) >= 1):
                rsi = crypto['rsi'].Current.Value
                lows = list(crypto['lows'])
                if rsi < 35 and lows[-1] > lows[-3]:
                    # Classic bullish divergence: oversold + higher low
                    components['rsi_divergence'] = 0.20
                elif rsi < 40 and crypto['returns'][-1] > 0 and lows[-1] > lows[-2]:
                    # Partial: recovering from near-oversold with momentum
                    components['rsi_divergence'] = 0.10

            # ----------------------------------------------------------
            # Signal 2: Volume Spike + Price Compression
            # Volume > 2× median AND BB width in bottom 20th percentile
            # → breakout imminent
            # ----------------------------------------------------------
            if len(crypto['volume']) >= 12 and len(crypto['bb_width']) >= 10:
                volumes = list(crypto['volume'])
                median_vol = np.median(volumes[-12:])
                current_vol = volumes[-1]

                bb_widths = list(crypto['bb_width'])
                pct20 = np.percentile(bb_widths, 20)
                pct35 = np.percentile(bb_widths, 35)
                current_bw = bb_widths[-1]

                if (median_vol > 0
                        and current_vol > 2.0 * median_vol
                        and current_bw <= pct20):
                    components['volume_bb_compression'] = 0.20
                elif (median_vol > 0
                      and current_vol > 1.5 * median_vol
                      and current_bw <= pct35):
                    components['volume_bb_compression'] = 0.10

            # ----------------------------------------------------------
            # Signal 3: EMA Crossover Momentum
            # EMA(3) crosses above EMA(6) with volume > 1.5× average
            # ----------------------------------------------------------
            if (crypto['ema_ultra_short'].IsReady
                    and crypto['ema_short'].IsReady
                    and len(crypto['volume']) >= 6):
                ema3 = crypto['ema_ultra_short'].Current.Value
                ema6 = crypto['ema_short'].Current.Value
                volumes = list(crypto['volume'])
                avg_vol = np.mean(volumes[-6:])
                current_vol = volumes[-1]

                if ema3 > ema6:
                    if avg_vol > 0 and current_vol > 1.5 * avg_vol:
                        components['ema_crossover'] = 0.20
                    elif ema3 > ema6 * 1.001:
                        # Meaningful crossover even without volume spike
                        components['ema_crossover'] = 0.10

            # ----------------------------------------------------------
            # Signal 4: VWAP Reclaim
            # Price crosses above VWAP from below with increasing volume.
            # VWAP approximated as volume-weighted mean of last 12 bars.
            # ----------------------------------------------------------
            if len(crypto['prices']) >= 4 and len(crypto['volume']) >= 4:
                prices = list(crypto['prices'])
                volumes = list(crypto['volume'])
                lookback = min(12, len(prices))
                total_vol = sum(volumes[-lookback:])
                vwap = (
                    sum(p * v for p, v in
                        zip(prices[-lookback:], volumes[-lookback:]))
                    / max(total_vol, 1e-10)
                )
                price = prices[-1]
                prev_price = prices[-2]
                vol_increasing = (len(volumes) >= 2
                                  and volumes[-1] > volumes[-2])

                if prev_price <= vwap < price and vol_increasing:
                    components['vwap_reclaim'] = 0.20
                elif price > vwap and vol_increasing:
                    # Already above VWAP with volume support → partial credit
                    components['vwap_reclaim'] = 0.10

            # ----------------------------------------------------------
            # Signal 5: Orderbook Imbalance Proxy
            # Spread narrowing + volume spike → buy-pressure confirmation
            # ----------------------------------------------------------
            if len(crypto['spreads']) >= 4 and len(crypto['volume']) >= 6:
                spreads = list(crypto['spreads'])
                volumes = list(crypto['volume'])
                avg_spread = np.mean(spreads[-4:])
                current_spread = spreads[-1]
                avg_vol = np.mean(volumes[-6:])
                current_vol = volumes[-1]

                spread_narrowing = (avg_spread > 0
                                    and current_spread < avg_spread * 0.85)
                vol_spike = avg_vol > 0 and current_vol > 1.5 * avg_vol

                if spread_narrowing and vol_spike:
                    components['orderbook_proxy'] = 0.20
                elif vol_spike:
                    components['orderbook_proxy'] = 0.10

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_scalp_score error: {e}")

        score = sum(components.values())
        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Kelly-adjusted aggressive position sizing.

        Base: 70 % of available capital.
        Conviction multiplier: 1.0 (normal) → 1.3 (high-conviction).
        Bear regime: base reduced to 50 %.
        """
        base = self.algo.position_size_pct  # 0.70

        # Conviction scales linearly from 0.40 → 0.60 threshold range
        if score >= 0.60:
            conviction = 1.30
        elif score >= threshold:
            conviction = 1.0 + (score - threshold) / 0.20 * 0.30
        else:
            conviction = 0.70

        if self.algo.market_regime == "bear":
            base *= 0.50

        kelly = self.algo._kelly_fraction()
        return base * conviction * kelly
