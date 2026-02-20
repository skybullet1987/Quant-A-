# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v7.0.0

    High-frequency market microstructure scalping system.
    Uses cutting-edge microstructure signals tuned for 1-minute bars on Kraken.

    Score: 0.0 – 1.0 across five equal signals (0.20 each).
      >= 0.60 → entry (3/5 signals firing)
      >= 0.80 → high-conviction entry (4+ signals) → maximum position size

    Signals
    -------
    1. Order Book Imbalance (OBI): bid/ask pressure (tightened threshold)
    2. Volume Ignition: 4× volume surge (tightened from 3×)
    3. MTF Trend Alignment: EMA5 > EMA20 (short-term trend aligned with medium)
    4. ADX Regime Filter: ADX > 20 (trending market, avoids choppy environments)
    5. VWAP Reclaim: price above rolling 20-bar VWAP (institutional reference level)
    """

    # Tunable signal thresholds (easy to adjust for backtesting)
    OBI_STRONG_THRESHOLD    = 0.60   # strong bid-side imbalance
    OBI_PARTIAL_THRESHOLD   = 0.30   # partial bid-side imbalance
    VOL_SURGE_STRONG        = 4.0    # 4× average volume = strong ignition
    VOL_SURGE_PARTIAL       = 2.5    # 2.5× volume = moderate spike
    ADX_STRONG_THRESHOLD    = 25     # strong directional trend
    ADX_MODERATE_THRESHOLD  = 20     # moderate directional trend
    VWAP_BUFFER             = 1.001  # 0.1% above VWAP for confirmed reclaim

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Primary entry: returns (score, components_dict)
    # ------------------------------------------------------------------
    def calculate_scalp_score(self, crypto):
        """
        Calculate the aggregate scalp score using five microstructure signals.

        Returns
        -------
        (score, components) where score ∈ [0, 1] and components maps each
        signal name to its individual contribution (0.20 max each).
        """
        components = {
            'obi':            0.0,
            'vol_ignition':   0.0,
            'micro_trend':    0.0,
            'adx_filter':     0.0,
            'vwap_signal':    0.0,
        }

        try:
            # ----------------------------------------------------------
            # Signal 1: Order Book Imbalance (OBI)
            # OBI = (bid_size - ask_size) / (bid_size + ask_size)
            # Strong buy pressure when OBI > 0.6 (bid wall dominates).
            # Tightened from 0.5 → 0.6 to reduce false signals.
            # ----------------------------------------------------------
            bid_size = crypto.get('bid_size', 0.0)
            ask_size = crypto.get('ask_size', 0.0)
            total_size = bid_size + ask_size
            if total_size > 0:
                obi = (bid_size - ask_size) / total_size
                if obi > self.OBI_STRONG_THRESHOLD:
                    components['obi'] = 0.20
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"OBI Signal: obi={obi:.3f} bid={bid_size:.2f} ask={ask_size:.2f}")
                elif obi > self.OBI_PARTIAL_THRESHOLD:
                    # Partial credit for meaningful buy-side imbalance
                    components['obi'] = 0.10

            # ----------------------------------------------------------
            # Signal 2: Volume Ignition
            # Current 1-minute volume > 4× the 20-bar moving-average volume.
            # Tightened from 3× → 4× to target only strong conviction bars.
            # ----------------------------------------------------------
            if len(crypto['volume']) >= 20:
                volumes = list(crypto['volume'])
                vol_ma_20 = np.mean(volumes[-20:])
                current_vol = volumes[-1]
                if vol_ma_20 > 0:
                    ratio = current_vol / vol_ma_20
                    if ratio >= self.VOL_SURGE_STRONG:
                        components['vol_ignition'] = 0.20
                        if not self.algo.IsWarmingUp:
                            self.algo.Debug(
                                f"Volume Ignition: vol={current_vol:.2f} ma20={vol_ma_20:.2f} ratio={ratio:.1f}x")
                    elif ratio >= self.VOL_SURGE_PARTIAL:
                        # Partial credit for a meaningful volume spike
                        components['vol_ignition'] = 0.10

            # ----------------------------------------------------------
            # Signal 3: MTF Trend Alignment
            # Price > EMA5 AND EMA5 > EMA20 → short-term and medium-term
            # trends are aligned (simulates 5m/20m multi-timeframe check).
            # ----------------------------------------------------------
            if (crypto['ema_5'].IsReady and crypto.get('ema_medium') is not None
                    and crypto['ema_medium'].IsReady and len(crypto['prices']) >= 1):
                price = crypto['prices'][-1]
                ema5 = crypto['ema_5'].Current.Value
                ema20 = crypto['ema_medium'].Current.Value
                if price > ema5 and ema5 > ema20:
                    # Full credit: short-term and medium-term trends aligned
                    components['micro_trend'] = 0.20
                elif price > ema5:
                    # Partial credit: price above immediate EMA only
                    components['micro_trend'] = 0.10

            # ----------------------------------------------------------
            # Signal 4: ADX Regime Filter
            # ADX > 20 → market is trending (not choppy/ranging).
            # DI+ > DI- → bullish directional bias.
            # Prevents entries in low-probability sideways conditions.
            # ----------------------------------------------------------
            adx_indicator = crypto.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val = adx_indicator.Current.Value
                di_plus = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                if adx_val > self.ADX_STRONG_THRESHOLD and di_plus > di_minus:
                    # Strong trend with bullish bias
                    components['adx_filter'] = 0.20
                elif adx_val > self.ADX_MODERATE_THRESHOLD and di_plus > di_minus:
                    # Moderate trend with bullish bias
                    components['adx_filter'] = 0.10

            # ----------------------------------------------------------
            # Signal 5: VWAP Reclaim
            # Price > rolling 20-bar VWAP → price is above the volume-
            # weighted average, indicating institutional buying support.
            # ----------------------------------------------------------
            vwap = crypto.get('vwap', 0.0)
            if vwap > 0 and len(crypto['prices']) >= 1:
                price = crypto['prices'][-1]
                if price > vwap * self.VWAP_BUFFER:
                    # Price clearly above VWAP (0.1% buffer)
                    components['vwap_signal'] = 0.20
                elif price > vwap:
                    # Price marginally above VWAP
                    components['vwap_signal'] = 0.10

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_scalp_score error: {e}")

        score = sum(components.values())
        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Conservative position sizing prioritising capital preservation.

        Returns 70–90% of available capital depending on conviction.
        Bear regime: base reduced by 30%.
        """
        if score >= 0.80:
            # 4+ signals firing – high conviction
            size = 0.90
        elif score >= self.algo.high_conviction_threshold:
            # 3+ signals: good conviction
            size = 0.80
        elif score >= threshold:
            # Entry threshold met: moderate sizing
            size = 0.70
        else:
            size = 0.50

        if self.algo.market_regime == "bear":
            size *= 0.70

        kelly = self.algo._kelly_fraction()
        return size * kelly

