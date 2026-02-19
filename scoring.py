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
        if mx == mn:
            return 0.5
        return max(0, min(1, (v - mn) / (mx - mn)))
    
    def _hurst_exponent(self, prices, max_lag=20):
        """Estimate Hurst exponent to detect trending vs mean-reverting behavior."""
        if len(prices) < max_lag + 5:
            return 0.5  # Not enough data, assume random walk
        try:
            price_arr = np.array(list(prices))
            lags = range(2, min(max_lag, len(price_arr) // 2))
            if len(list(lags)) < 3:
                return 0.5
            tau = [np.std(price_arr[lag:] - price_arr[:-lag]) for lag in lags]
            if any(t <= 0 for t in tau):
                return 0.5
            log_lags = np.log(list(lags))
            log_tau = np.log(tau)
            poly = np.polyfit(log_lags, log_tau, 1)
            return max(0.0, min(1.0, poly[0]))
        except Exception as e:
            self.algo.Debug(f"Error in _hurst_exponent: {e}")
            return 0.5
    
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

            # 2. VOLUME MOMENTUM (smoothed)
            if len(crypto['volume_ma']) >= 2 and len(crypto['returns']) >= 3:
                vol_ma_prev = crypto['volume_ma'][-2]
                vol_trend = (crypto['volume_ma'][-1] / (vol_ma_prev + 1e-8)) - 1
                price_trend = np.mean(list(crypto['returns'])[-3:])
                
                # Continuous base score from price trend
                base = 0.5 + price_trend * 20  # scale price trend to [0,1] range
                base = max(0.1, min(0.9, base))
                
                # Volume trend bonus (continuous)
                if vol_trend > 0:
                    vol_bonus = min(0.15, vol_trend * 3)
                    base += vol_bonus
                
                # Volume spike bonus (continuous, not binary)
                if len(crypto['volume']) >= 12:
                    median_vol = np.median(list(crypto['volume'])[-12:])
                    current_vol = crypto['volume'][-1]
                    if median_vol > 0:
                        vol_ratio = current_vol / median_vol
                        spike_bonus = min(0.1, max(0, (vol_ratio - 1.5) * 0.1))
                        base += spike_bonus
                
                scores['volume_momentum'] = max(0.05, min(0.95, base))
            else:
                scores['volume_momentum'] = 0.5

            # 3. TREND STRENGTH (smoothed + incorporates multi-timeframe)
            if crypto['ema_short'].IsReady and crypto['ema_trend'].IsReady:
                us = crypto['ema_ultra_short'].Current.Value
                s = crypto['ema_short'].Current.Value
                m = crypto['ema_medium'].Current.Value
                l = crypto['ema_long'].Current.Value
                t = crypto['ema_trend'].Current.Value
                
                # Continuous alignment score: each pair contributes proportionally
                pairs = [(us, s), (s, m), (m, l), (l, t)]
                alignment_score = 0.0
                for fast, slow in pairs:
                    if slow > 0:
                        diff_pct = (fast - slow) / slow
                        # Sigmoid-like: maps diff_pct to [0, 1] smoothly
                        pair_score = 1.0 / (1.0 + np.exp(-diff_pct * 200))
                        alignment_score += pair_score * 0.25  # each pair = 25%
                
                # Blend with multi-timeframe momentum for richer signal
                if len(crypto['returns']) >= 24:
                    returns = list(crypto['returns'])
                    tf_scores = []
                    for period in [3, 6, 12, 24]:
                        if len(returns) >= period:
                            tf_ret = np.mean(returns[-period:])
                            tf_scores.append(1.0 / (1.0 + np.exp(-tf_ret * 300)))
                    if tf_scores:
                        multi_tf = np.mean(tf_scores)
                        # 70% EMA alignment + 30% multi-TF momentum
                        scores['trend_strength'] = max(0.05, min(0.95, alignment_score * 0.7 + multi_tf * 0.3))
                    else:
                        scores['trend_strength'] = max(0.05, min(0.95, alignment_score))
                else:
                    scores['trend_strength'] = max(0.05, min(0.95, alignment_score))
            elif crypto['ema_short'].IsReady:
                s = crypto['ema_short'].Current.Value
                m = crypto['ema_medium'].Current.Value
                l = crypto['ema_long'].Current.Value
                if l > 0:
                    diff = (s - l) / l
                    scores['trend_strength'] = max(0.1, min(0.9, 0.5 + diff * 10))
                else:
                    scores['trend_strength'] = 0.5
            else:
                scores['trend_strength'] = 0.5

            # 4. MEAN REVERSION (smoothed continuous)
            if len(crypto['zscore']) >= 1:
                z = crypto['zscore'][-1]
                rsi = crypto['rsi'].Current.Value
                
                # Sigmoid mapping: very negative z + low RSI = high score (oversold bounce opportunity)
                # very positive z + high RSI = low score (overbought, avoid)
                z_component = 1.0 / (1.0 + np.exp(z * 1.5))  # z=-2 → 0.95, z=0 → 0.5, z=2 → 0.05
                rsi_component = 1.0 / (1.0 + np.exp((rsi - 50) * 0.08))  # rsi=20 → 0.92, rsi=50 → 0.5, rsi=80 → 0.08
                
                # Weighted blend: z-score is more reliable than RSI for mean reversion
                scores['mean_reversion'] = max(0.05, min(0.95, z_component * 0.6 + rsi_component * 0.4))
            else:
                scores['mean_reversion'] = 0.5

            # 5. LIQUIDITY (smoothed)
            if len(crypto['dollar_volume']) >= 12:
                avg = np.mean(list(crypto['dollar_volume'])[-12:])
                # Log-scale continuous scoring
                if avg > 0:
                    scores['liquidity'] = max(0.1, min(0.95, 0.2 + 0.15 * np.log10(max(avg, 1))))
                else:
                    scores['liquidity'] = 0.1
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
        except Exception as e:
            self.algo.Debug(f"Error in calculate_factor_scores for {symbol.Value}: {e}")
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
        Now returns neutral since multi-TF is merged into trend_strength.
        
        Args:
            crypto: Crypto data dictionary with returns history
            
        Returns:
            Neutral score of 0.5
        """
        return 0.5

    def calculate_composite_score(self, factors, crypto=None):
        """
        Calculate weighted composite score with regime adjustments.
        
        Args:
            factors: Dictionary of individual factor scores
            crypto: Optional crypto data dictionary for overextension penalty
            
        Returns:
            Composite score adjusted for market regime and breadth
        """
        # Dynamic weight adjustment based on Hurst exponent
        weights = dict(self.algo.weights)  # copy base weights

        hurst_mult = 1.0
        if crypto and len(crypto.get('prices', [])) >= 30:
            hurst = self._hurst_exponent(crypto['prices'])
            if hurst > 0.6:
                hurst_mult = 1.05  # trending: boost score
            elif hurst < 0.4:
                hurst_mult = 0.95  # mean-reverting: reduce score
        
        # Normalize weights to sum to 1.0
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        
        score = sum(factors.get(f, 0.5) * w for f, w in weights.items())
        score *= hurst_mult  # apply Hurst-based multiplier (item 10)
        
        # Regime adjustments (mild penalties to avoid stacking with main.py)
        if self.algo.market_regime == "bear":
            score *= 0.92  # Mild penalty
        if self.algo.volatility_regime == "high":
            score *= 0.93
        if self.algo.market_breadth > 0.7:
            score *= 1.08
        elif self.algo.market_breadth < 0.3:
            score *= 0.95  # Mild penalty
        
        # Bonus for strong breakout
        if factors.get('breakout_score', 0) > 0.8:
            score *= 1.05
        
        # NEW: Overextension penalty
        # Penalize entries where price is far above EMA-medium (buying the top)
        if crypto and crypto.get('ema_medium') and crypto['ema_medium'].IsReady:
            price = crypto.get('last_price', 0)
            ema_m = crypto['ema_medium'].Current.Value
            if ema_m > 0 and price > 0:
                extension_pct = (price - ema_m) / ema_m
                if extension_pct > 0.08:  # >8% above EMA-medium
                    # Continuous penalty: 8% over = small penalty, 15%+ = heavy penalty
                    penalty = min(0.25, (extension_pct - 0.08) * 2.5)
                    score *= (1.0 - penalty)
        
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

    def calculate_explosion_score(self, crypto):
        """
        Detect the IGNITION moment of a potential altcoin spike.
        Combines volume explosion, BB squeeze breakout, KAMA divergence,
        multi-bar acceleration, and period-high breakout.

        Returns 0.0-1.0, where >0.7 = extremely high conviction spike signal.
        """
        price = crypto['last_price']
        if price <= 0:
            return 0.0

        score = 0.0
        signals_firing = 0

        # === SIGNAL 1: Volume Explosion (strongest predictor) ===
        lookback = self.algo.explosion_volume_lookback
        if len(crypto['volume']) >= lookback:
            volumes = list(crypto['volume'])
            median_vol = np.median(volumes[-lookback:])
            if median_vol > 0:
                current_ratio = volumes[-1] / median_vol
                recent_avg = np.mean(volumes[-3:]) / median_vol if len(volumes) >= 3 else current_ratio

                if current_ratio > 5.0:
                    score += 0.30
                    signals_firing += 1
                elif current_ratio > 3.0:
                    score += 0.20
                    signals_firing += 1
                elif recent_avg > 2.0:
                    score += 0.10
                    signals_firing += 1

        # === SIGNAL 2: BB Squeeze → Breakout ===
        bb_lookback = self.algo.explosion_bb_lookback
        if len(crypto['bb_width']) >= bb_lookback and len(crypto['bb_upper']) >= 1:
            avg_width = np.mean(list(crypto['bb_width'])[-bb_lookback:])
            current_width = crypto['bb_width'][-1]
            if avg_width > 0:
                squeeze = current_width / avg_width
                if squeeze < 0.5 and price > crypto['bb_upper'][-1]:
                    score += 0.25
                    signals_firing += 1
                elif squeeze < 0.7 and price > crypto['bb_upper'][-1]:
                    score += 0.15
                    signals_firing += 1

        # === SIGNAL 3: KAMA Divergence (catches acceleration) ===
        if crypto['kama'].IsReady and crypto['ema_short'].IsReady:
            kama_val = crypto['kama'].Current.Value
            ema_val = crypto['ema_short'].Current.Value
            if ema_val > 0:
                kama_lead = (kama_val - ema_val) / ema_val
                if kama_lead > 0.01:
                    score += 0.15
                    signals_firing += 1

        # === SIGNAL 4: Multi-bar Acceleration ===
        if len(crypto['returns']) >= 6:
            returns = list(crypto['returns'])
            r_recent = np.mean(returns[-3:])
            r_prior = np.mean(returns[-6:-3])
            if r_recent > 0 and r_prior > 0 and r_recent > r_prior * 2:
                score += 0.15
                signals_firing += 1
            elif r_recent > 0.02:
                score += 0.10
                signals_firing += 1

        # === SIGNAL 5: Breaking Period High ===
        if len(crypto['highs']) >= 48:
            period_high = max(list(crypto['highs'])[-48:])
            if period_high > 0 and price > period_high:
                score += 0.15
                signals_firing += 1

        # === CONFLUENCE BONUS ===
        if signals_firing >= 4:
            score *= 1.3
        elif signals_firing >= 3:
            score *= 1.15

        # === SPIKE CONFIRMATION (item 5) ===
        if len(crypto['returns']) >= 1:
            returns_list = list(crypto['returns'])
            curr_ret = returns_list[-1]
            if curr_ret < -0.02:
                score *= 0.3  # Current bar dropped >2%: likely dump, not pump
            elif len(returns_list) >= 2 and returns_list[-2] <= 0:
                score *= 0.7  # Previous bar not positive: weaker confirmation

        return min(score, 1.0)

    def calculate_accumulation_score(self, crypto):
        """
        Detect Wyckoff-style accumulation: smart money loading before a pump.
        This catches the SETUP phase, not the explosion.
        Returns 0.0-1.0.
        """
        score = 0.0
        signals = 0
        price = crypto['last_price']
        if price <= 0:
            return 0.0

        # SIGNAL 1: Volume Dry-Up then Surge (Accumulation Signature)
        if len(crypto['volume']) >= 24:
            volumes = list(crypto['volume'])
            recent_vol = np.mean(volumes[-3:])
            prior_vol = np.mean(volumes[-12:-3])
            baseline_vol = np.median(volumes[-24:])

            if baseline_vol > 0:
                prior_ratio = prior_vol / baseline_vol
                recent_ratio = recent_vol / baseline_vol

                if prior_ratio < 0.6 and recent_ratio > 1.2:
                    score += 0.25
                    signals += 1
                elif prior_ratio < 0.8 and recent_ratio > 1.5:
                    score += 0.20
                    signals += 1

        # SIGNAL 2: Narrowing Range with Rising Floor (Compression before expansion)
        if len(crypto['highs']) >= 24 and len(crypto['lows']) >= 24:
            highs = list(crypto['highs'])
            lows = list(crypto['lows'])

            recent_range = max(highs[-6:]) - min(lows[-6:])
            prior_range = max(highs[-18:-6]) - min(lows[-18:-6]) if len(highs) >= 18 else recent_range * 2

            if len(lows) >= 12:
                recent_low = min(lows[-6:])
                prior_low = min(lows[-12:-6])
                lows_rising = recent_low > prior_low
            else:
                lows_rising = False

            if prior_range > 0:
                range_compression = recent_range / prior_range
                if range_compression < 0.5 and lows_rising:
                    score += 0.25
                    signals += 1
                elif range_compression < 0.7 and lows_rising:
                    score += 0.15
                    signals += 1

        # SIGNAL 3: OBV Divergence (cumulative buying pressure rising while price flat)
        if len(crypto['returns']) >= 18 and len(crypto['volume']) >= 18:
            returns = list(crypto['returns'])
            volumes = list(crypto['volume'])

            obv_changes = []
            for i in range(-18, 0):
                if returns[i] > 0:
                    obv_changes.append(volumes[i])
                elif returns[i] < 0:
                    obv_changes.append(-volumes[i])
                else:
                    obv_changes.append(0)

            obv = np.cumsum(obv_changes)

            price_change_18 = sum(returns[-18:])
            obv_trend = obv[-1] - obv[0]

            if abs(price_change_18) < 0.03 and obv_trend > 0:
                total_vol = sum(abs(v) for v in obv_changes)
                if total_vol > 0:
                    obv_strength = obv_trend / total_vol
                    if obv_strength > 0.3:
                        score += 0.25
                        signals += 1
                    elif obv_strength > 0.15:
                        score += 0.15
                        signals += 1

        # SIGNAL 4: RSI Recovery from Oversold (early momentum shift)
        if crypto['rsi'].IsReady:
            rsi = crypto['rsi'].Current.Value
            if len(crypto['zscore']) >= 6:
                recent_z = list(crypto['zscore'])
                was_oversold = any(z < -1.5 for z in recent_z[-6:])
                recovering = rsi > 40 and rsi < 60

                if was_oversold and recovering:
                    score += 0.20
                    signals += 1

        # CONFLUENCE BONUS
        if signals >= 3:
            score *= 1.30
        elif signals >= 2:
            score *= 1.15

        return min(score, 1.0)

    def calculate_relative_outperformance_score(self, crypto):
        """
        Find coins outperforming the market — the ones pumping while everything else is flat.
        Returns 0.0-1.0.
        """
        score = 0.0

        if len(crypto['rs_vs_btc']) >= 6:
            rs_list = list(crypto['rs_vs_btc'])
            rs_3 = np.mean(rs_list[-3:])
            rs_6 = np.mean(rs_list[-6:])

            if rs_3 > 0.02:
                score += 0.25
            elif rs_3 > 0.01:
                score += 0.15

            if rs_6 > 0.015:
                score += 0.15

            if rs_3 > rs_6 and rs_3 > 0.01:
                score += 0.10

        if len(crypto['returns']) >= 24:
            returns = list(crypto['returns'])
            recent_ret = np.mean(returns[-3:])
            historical_std = np.std(returns[-24:])

            if historical_std > 0:
                z_move = recent_ret / historical_std
                if z_move > 2.0:
                    score += 0.25
                elif z_move > 1.0:
                    score += 0.15

        if len(crypto['volume']) >= 12 and len(crypto['returns']) >= 6:
            volumes = list(crypto['volume'])
            returns = list(crypto['returns'])

            recent_vol = np.mean(volumes[-3:])
            baseline_vol = np.median(volumes[-12:])
            recent_ret = sum(returns[-6:])

            if baseline_vol > 0 and recent_vol > baseline_vol * 1.3 and recent_ret > 0.02:
                score += 0.15

        return min(score, 1.0)

    def calculate_smart_money_flow(self, crypto):
        """
        Detect institutional accumulation from OHLCV candle structure.
        Returns 0.0-1.0.
        """
        score = 0.0

        if len(crypto['prices']) < 12 or len(crypto['highs']) < 12 or len(crypto['lows']) < 12 or len(crypto['volume']) < 12:
            return 0.0

        prices = list(crypto['prices'])
        highs = list(crypto['highs'])
        lows = list(crypto['lows'])
        volumes = list(crypto['volume'])

        # SIGNAL 1: Close Location Value (CLV) trend
        clv_values = []
        for i in range(-12, 0):
            bar_range = highs[i] - lows[i]
            if bar_range > 0:
                clv = (prices[i] - lows[i]) / bar_range
                clv_values.append(clv)

        if len(clv_values) >= 6:
            avg_clv = np.mean(clv_values[-6:])
            recent_clv = np.mean(clv_values[-3:])

            if recent_clv > 0.65:
                score += 0.20
            if avg_clv > 0.55 and recent_clv > avg_clv:
                score += 0.10

        # SIGNAL 2: Money Flow = CLV × Volume (Chaikin-style)
        if len(clv_values) >= 12:
            mf_recent = sum((clv_values[i] * 2 - 1) * volumes[-12 + i] for i in range(6, 12))
            mf_prior = sum((clv_values[i] * 2 - 1) * volumes[-12 + i] for i in range(0, 6))

            price_change = (prices[-1] - prices[-7]) / prices[-7] if prices[-7] > 0 else 0

            if mf_recent > mf_prior and abs(price_change) < 0.03:
                score += 0.25
            elif mf_recent > mf_prior * 1.5:
                score += 0.15

        # SIGNAL 3: Bar Absorption (high volume + small range = large player absorbing)
        absorption_count = 0
        for i in range(-6, 0):
            bar_range = highs[i] - lows[i]
            median_range = np.median([highs[j] - lows[j] for j in range(-12, 0)])
            median_vol = np.median(volumes[-12:])

            if median_range > 0 and median_vol > 0:
                range_ratio = bar_range / median_range
                vol_ratio = volumes[i] / median_vol

                if range_ratio < 0.5 and vol_ratio > 1.5:
                    absorption_count += 1

        if absorption_count >= 2:
            score += 0.20
        elif absorption_count >= 1:
            score += 0.10

        # SIGNAL 4: Tail Ratio (lower wicks = buyers defending)
        tail_strength = 0
        for i in range(-6, 0):
            bar_range = highs[i] - lows[i]
            if bar_range > 0 and i > -6:
                close_vs_prev = prices[i] - prices[i - 1]
                if close_vs_prev >= 0:
                    lower_tail = (min(prices[i], prices[i - 1]) - lows[i]) / bar_range
                    if lower_tail > 0.3:
                        tail_strength += 1

        if tail_strength >= 3:
            score += 0.15

        return min(score, 1.0)

    def calculate_snipe_score(self, crypto):
        """
        Master snipe score combining all 4 detection engines.
        Returns (snipe_score, is_snipe, components_dict).
        """
        try:
            explosion = self.calculate_explosion_score(crypto)
            accumulation = self.calculate_accumulation_score(crypto)
            relative_outperf = self.calculate_relative_outperformance_score(crypto)
            smart_money = self.calculate_smart_money_flow(crypto)

            snipe_score = (
                explosion * 0.20 +
                accumulation * 0.30 +
                relative_outperf * 0.25 +
                smart_money * 0.25
            )

            engines_firing = sum(1 for s in [explosion, accumulation, relative_outperf, smart_money] if s > 0.4)
            if engines_firing >= 3:
                snipe_score *= 1.25
            elif engines_firing >= 2:
                snipe_score *= 1.10

            engines_above_half = sum(1 for s in [explosion, accumulation, relative_outperf, smart_money] if s > 0.5)
            is_snipe = snipe_score > 0.55 and engines_above_half >= 2

            components = {
                'explosion': explosion,
                'accumulation': accumulation,
                'relative_outperf': relative_outperf,
                'smart_money': smart_money,
            }

            return min(snipe_score, 1.0), is_snipe, components
        except Exception as e:
            self.algo.Debug(f"Error in calculate_snipe_score: {e}")
            return 0.0, False, {}

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
