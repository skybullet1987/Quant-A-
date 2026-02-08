# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class OpusScoringEngine:
    """
    Scoring Engine for Opus Crypto Strategy
    
    Handles all 8-factor scoring, composite scoring, fee adjustments,
    and position sizing calculations. This module is separated from
    the main algorithm to keep both files under QuantConnect's 64K limit.
    """
    
    def __init__(self, algorithm):
        """
        Initialize scoring engine with reference to main algorithm.
        
        Args:
            algorithm: Reference to OpusCryptoStrategy instance
        """
        self.algo = algorithm
    
    def _normalize(self, v, mn, mx):
        """Normalize value to [0, 1] range."""
        return max(0, min(1, (v - mn) / (mx - mn)))
    
    def calculate_factor_scores(self, symbol, crypto):
        """
        Calculate all 8 factor scores for a given crypto.
        
        Args:
            symbol: Symbol object
            crypto: Crypto data dictionary with indicators
            
        Returns:
            Dictionary of factor scores or None if calculation fails
        """
        try:
            scores = {}

            # 1. RELATIVE STRENGTH vs BTC
            if len(crypto['rs_vs_btc']) >= 3:
                scores['relative_strength'] = self._normalize(np.mean(list(crypto['rs_vs_btc'])[-3:]), -0.05, 0.05)
            else:
                scores['relative_strength'] = 0.5

            # 2. VOLUME MOMENTUM (enhanced)
            if len(crypto['volume_ma']) >= 2 and len(crypto['returns']) >= 3:
                vol_ma_prev = crypto['volume_ma'][-2]
                vol_trend = (crypto['volume_ma'][-1] / (vol_ma_prev + 1e-8)) - 1
                price_trend = np.mean(list(crypto['returns'])[-3:])
                # Volume spike detection
                if len(crypto['volume']) >= 12:
                    median_vol = np.median(list(crypto['volume'])[-12:])
                    current_vol = crypto['volume'][-1]
                    vol_spike = current_vol / (median_vol + 1e-8)
                else:
                    vol_spike = 1.0
                if vol_trend > 0 and price_trend > 0:
                    base = min(0.5 + vol_trend * 5 + price_trend * 25, 1.0)
                    if vol_spike > 2.0:
                        base = min(base + 0.15, 1.0)  # Volume spike bonus
                    scores['volume_momentum'] = base
                elif price_trend > 0:
                    scores['volume_momentum'] = 0.55
                else:
                    scores['volume_momentum'] = 0.3
            else:
                scores['volume_momentum'] = 0.5

            # 3. TREND STRENGTH (enhanced with more EMAs)
            if crypto['ema_short'].IsReady and crypto['ema_trend'].IsReady:
                us = crypto['ema_ultra_short'].Current.Value
                s = crypto['ema_short'].Current.Value
                m = crypto['ema_medium'].Current.Value
                l = crypto['ema_long'].Current.Value
                t = crypto['ema_trend'].Current.Value
                aligned = sum([us > s, s > m, m > l, l > t])
                if aligned >= 4:
                    scores['trend_strength'] = min(0.7 + ((s - t) / t) * 8, 1.0)
                elif aligned >= 3:
                    scores['trend_strength'] = 0.7
                elif aligned >= 2:
                    scores['trend_strength'] = 0.55
                elif us < s < m < l:
                    scores['trend_strength'] = 0.15
                else:
                    scores['trend_strength'] = 0.35
            elif crypto['ema_short'].IsReady:
                s = crypto['ema_short'].Current.Value
                m = crypto['ema_medium'].Current.Value
                l = crypto['ema_long'].Current.Value
                if s > m > l:
                    scores['trend_strength'] = min(0.6 + ((s - l) / l) * 10, 1.0)
                elif s > m:
                    scores['trend_strength'] = 0.6
                else:
                    scores['trend_strength'] = 0.3
            else:
                scores['trend_strength'] = 0.5

            # 4. MEAN REVERSION (enhanced)
            if len(crypto['zscore']) >= 1:
                z = crypto['zscore'][-1]
                rsi = crypto['rsi'].Current.Value
                if z < -2.0 and rsi < 25:
                    scores['mean_reversion'] = 1.0     # Extreme oversold
                elif z < -1.5 and rsi < 35:
                    scores['mean_reversion'] = 0.9
                elif z < -1.0 and rsi < 40:
                    scores['mean_reversion'] = 0.75
                elif z > 2.5 or rsi > 80:
                    scores['mean_reversion'] = 0.05    # Extremely overbought — avoid
                elif z > 2.0 or rsi > 75:
                    scores['mean_reversion'] = 0.1
                else:
                    scores['mean_reversion'] = 0.5
            else:
                scores['mean_reversion'] = 0.5

            # 5. LIQUIDITY
            if len(crypto['dollar_volume']) >= 12:
                avg = np.mean(list(crypto['dollar_volume'])[-12:])
                if avg > 10000:
                    scores['liquidity'] = 0.9
                elif avg > 5000:
                    scores['liquidity'] = 0.7
                elif avg > 1000:
                    scores['liquidity'] = 0.5
                else:
                    scores['liquidity'] = 0.25
            else:
                scores['liquidity'] = 0.5

            # 6. RISK-ADJUSTED MOMENTUM (Sharpe-like)
            if len(crypto['returns']) >= self.algo.medium_period:
                rets = list(crypto['returns'])[-self.algo.medium_period:]
                std = np.std(rets)
                if std > 1e-10:
                    sharpe = np.mean(rets) / std
                    scores['risk_adjusted_momentum'] = self._normalize(sharpe, -1, 1)
                else:
                    scores['risk_adjusted_momentum'] = 0.5
            else:
                scores['risk_adjusted_momentum'] = 0.5

            # 7. BREAKOUT SCORE (NEW)
            scores['breakout_score'] = self.calculate_breakout_score(crypto)

            # 8. MULTI-TIMEFRAME ALIGNMENT (NEW)
            scores['multi_timeframe'] = self.calculate_multi_tf_score(crypto)

            return scores
        except:
            return None

    def calculate_breakout_score(self, crypto):
        """
        Detect breakouts using price range, volume, and Bollinger Bands.
        
        Args:
            crypto: Crypto data dictionary with price and indicator history
            
        Returns:
            Breakout score between 0 and 1
        """
        score = 0.5
        price = crypto['last_price']
        if price <= 0:
            return score

        # Price breakout from recent range
        if len(crypto['highs']) >= self.algo.long_period:
            recent_high = max(list(crypto['highs'])[-self.algo.long_period:])
            recent_low = min(list(crypto['lows'])[-self.algo.long_period:])
            range_pct = (recent_high - recent_low) / recent_low if recent_low > 0 else 0
            position_in_range = (price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5

            if position_in_range > 0.95:  # Breaking out above range
                score += 0.2
            elif position_in_range > 0.85:
                score += 0.1

        # Bollinger Band squeeze (low width → upcoming breakout)
        if len(crypto['bb_width']) >= self.algo.medium_period:
            current_width = crypto['bb_width'][-1]
            avg_width = np.mean(list(crypto['bb_width']))
            if avg_width > 0:
                squeeze_ratio = current_width / avg_width
                if squeeze_ratio < 0.6:  # Squeeze detected
                    # Check if breaking upward
                    if len(crypto['bb_upper']) >= 1 and price > crypto['bb_upper'][-1]:
                        score += 0.25  # Squeeze breakout!
                    elif len(crypto['returns']) >= 2 and np.mean(list(crypto['returns'])[-2:]) > 0:
                        score += 0.1   # Squeeze with upward bias

        # Volume confirmation
        if len(crypto['volume']) >= 12:
            median_vol = np.median(list(crypto['volume'])[-12:])
            if median_vol > 0 and crypto['volume'][-1] > 2.0 * median_vol:
                score += 0.1  # Volume spike confirms breakout

        return min(score, 1.0)

    def calculate_multi_tf_score(self, crypto):
        """
        Score based on multi-timeframe momentum alignment.
        
        Checks 3h/6h/12h/24h momentum and rewards strong alignment.
        
        Args:
            crypto: Crypto data dictionary with returns history
            
        Returns:
            Multi-timeframe alignment score between 0 and 1
        """
        if len(crypto['returns']) < self.algo.long_period:
            return 0.5

        returns = list(crypto['returns'])
        agreements = 0
        total_checks = 0

        # 3h momentum
        if len(returns) >= 3:
            total_checks += 1
            if np.mean(returns[-3:]) > 0:
                agreements += 1

        # 6h momentum
        if len(returns) >= 6:
            total_checks += 1
            if np.mean(returns[-6:]) > 0:
                agreements += 1

        # 12h momentum
        if len(returns) >= 12:
            total_checks += 1
            if np.mean(returns[-12:]) > 0:
                agreements += 1

        # 24h momentum
        if len(returns) >= 24:
            total_checks += 1
            if np.mean(returns[-24:]) > 0:
                agreements += 1

        if total_checks == 0:
            return 0.5

        alignment = agreements / total_checks
        if alignment >= 0.75:
            return 0.85
        elif alignment >= 0.5:
            return 0.65
        elif alignment <= 0.25:
            return 0.15
        return 0.4

    def calculate_composite_score(self, factors):
        """
        Calculate weighted composite score with regime adjustments.
        
        Args:
            factors: Dictionary of individual factor scores
            
        Returns:
            Composite score adjusted for market regime and breadth
        """
        score = sum(factors.get(f, 0.5) * w for f, w in self.algo.weights.items())
        # Regime adjustments
        if self.algo.market_regime == "bear":
            score *= 0.78
        if self.algo.volatility_regime == "high":
            score *= 0.85
        if self.algo.market_breadth > 0.7:
            score *= 1.08
        elif self.algo.market_breadth < 0.3:
            score *= 0.80
        # Bonus for very strong breakout + multi-TF alignment
        if factors.get('breakout_score', 0) > 0.8 and factors.get('multi_timeframe', 0) > 0.7:
            score *= 1.10
        return min(score, 1.0)

    def apply_fee_adjustment(self, score):
        """
        Apply fee and slippage buffer deduction to score.
        
        Args:
            score: Raw composite score
            
        Returns:
            Fee-adjusted score
        """
        return score - (self.algo.expected_round_trip_fees * 1.1 + self.algo.fee_slippage_buffer)

    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Calculate position size using conviction, volatility, and Kelly fraction.
        
        Args:
            score: Net score after fee adjustment
            threshold: Entry threshold
            asset_vol_ann: Annualized volatility of the asset
            
        Returns:
            Position size as percentage of portfolio
        """
        conviction_mult = max(0.8, min(1.5, 0.8 + (score - threshold) * 3.5))
        vol_floor = max(asset_vol_ann if asset_vol_ann else 0.05, 0.05)
        risk_mult = max(0.7, min(1.3, self.algo.target_position_ann_vol / vol_floor))
        kelly_mult = self.algo._kelly_fraction()
        return self.algo.position_size_pct * conviction_mult * risk_mult * kelly_mult
