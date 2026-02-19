# region imports
from AlgorithmImports import *
from execution import *
from scoring import OpusScoringEngine
from collections import deque
import numpy as np
# endregion


class SimplifiedCryptoStrategy(QCAlgorithm):
    """
    Simplified 5-Factor Strategy - v4.0.0
    Exchange-status gate, stale handling, liquidity/impact guards, spread cap 3%, min notional $5,
    cash reserve 10%, open-order cap 3, session blacklist daily reset, size cap 25% in high-vol/sideways,
    time stop 18h if PnL<+1%, zombie auto-blacklist, insights suppressed, ObjectStore writes disabled.
    Aggressive tuning + 10 structural optimizations applied.
    """

    def Initialize(self):
        self.SetStartDate(2025, 1, 1)
        self.SetCash(20)
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        self.threshold_bull = self._get_param("threshold_bull", 0.42)
        self.threshold_bear = self._get_param("threshold_bear", 0.52)
        self.threshold_sideways = self._get_param("threshold_sideways", 0.47)
        self.threshold_high_vol = self._get_param("threshold_high_vol", 0.50)
        self.trailing_activation = self._get_param("trailing_activation", 0.04)
        self.trailing_stop_pct = self._get_param("trailing_stop_pct", 0.025)
        self.base_stop_loss = self._get_param("base_stop_loss", 0.06)
        self.base_take_profit = self._get_param("base_take_profit", 0.12)
        self.atr_sl_mult = self._get_param("atr_sl_mult", 2.0)
        self.atr_tp_mult = self._get_param("atr_tp_mult", 3.5)

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.35)
        self.portfolio_vol_cap = self._get_param("portfolio_vol_cap", 0.45)
        self.signal_decay_buffer = self._get_param("signal_decay_buffer", 0.03)
        self.min_signal_age_hours = self._get_param("signal_decay_min_hours", 12)
        self.cash_reserve_pct = 0.10

        self.ultra_short_period = 3
        self.short_period = 6
        self.medium_period = 24
        self.trend_period = 48
        self.long_period = 72
        self.lookback = 72
        self.sqrt_annualization = np.sqrt(24 * 365)
        self.min_asset_vol_floor = 0.05

        self.base_max_positions = 3
        self.max_positions = self.base_max_positions
        self.position_size_pct = 0.40
        self.min_notional = 5.0  # lowered per request
        self.min_price_usd = 0.005

        self.expected_round_trip_fees = 0.0020
        self.fee_slippage_buffer = 0.002

        self.max_spread_pct = 0.03
        self.spread_median_window = 12
        self.spread_widen_mult = 2.5
        self.skip_hours_utc = [3, 4]
        self.max_daily_trades = 10
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.stale_order_timeout_seconds = 300
        self.live_stale_order_timeout_seconds = 900
        self.max_concurrent_open_orders = 3
        self.open_orders_cash_threshold = 0.5  # Exit early if ≥50% cash reserved for pending orders
        
        # Order fill verification settings
        self.order_fill_check_threshold_seconds = 120
        self.order_timeout_seconds = 300
        self.resync_log_interval_seconds = 1800
        self.portfolio_mismatch_threshold = 0.10
        self.portfolio_mismatch_min_dollars = 1.00
        self.portfolio_mismatch_cooldown_seconds = 3600
        self.retry_pending_cooldown_seconds = 120
        self.rate_limit_cooldown_minutes = 10

        self._positions_synced = False
        self._session_blacklist = set()
        self._max_session_blacklist_size = 100
        self._first_post_warmup = True
        self._submitted_orders = {}  # {symbol: {'order_id': OrderId, 'time': datetime, 'quantity': float}}
        self._symbol_slippage_history = {}  # {ticker_string: deque(maxlen=10) of absolute slippage pcts}
        self._order_retries = {}  # {order_id: retry_count} - track order retry attempts
        self._retry_pending = {}  # {symbol: cancel_time} - track symbols awaiting retry after cancel
        self._rate_limit_until = None  # Rate limit hard block timestamp
        self._last_mismatch_warning = None  # Last portfolio mismatch warning time

        self.weights = {
            'relative_strength': 0.25,
            'volume_momentum': 0.20,
            'trend_strength': 0.20,
            'mean_reversion': 0.10,
            'liquidity': 0.10,
            'risk_adjusted_momentum': 0.15,
        }

        self.peak_value = None
        self.max_drawdown_limit = 0.30
        self.drawdown_cooldown = 0
        self.cooldown_hours = 24
        self.consecutive_losses = 0
        self.max_consecutive_losses = 8  # reduced to trigger cooldown sooner

        self.crypto_data = {}
        self.entry_prices = {}
        self.highest_prices = {}
        self.entry_times = {}
        self.trade_count = 0
        
        self._pending_orders = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns = {}
        self._symbol_loss_cooldowns = {}
        self.exit_cooldown_hours = 6
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 2

        self._cash_mode_until = None
        self._recent_trade_outcomes = deque(maxlen=20)

        self.trailing_grace_hours = 8
        self.atr_trail_mult = 2.5

        self._slip_abs = deque(maxlen=50)
        self._slippage_alert_until = None
        self.slip_alert_threshold = 0.0015
        self.slip_outlier_threshold = 0.004
        self.slip_alert_duration_hours = 2
        self._bad_symbol_counts = {}

        self._recent_tickets = deque(maxlen=25)

        self.btc_symbol = None
        self.btc_returns = deque(maxlen=self.long_period)
        self.btc_prices = deque(maxlen=self.long_period)
        self.btc_volatility = deque(maxlen=self.long_period)
        self._rolling_wins = deque(maxlen=50)
        self._rolling_win_sizes = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._last_live_trade_time = None

        self.market_regime = "unknown"
        self.volatility_regime = "normal"
        self.market_breadth = 0.5
        self._regime_hold_count = 0

        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.trade_log = []
        self.log_budget = 0
        self.last_log_time = None

        self.min_volume_usd = 500
        self.max_universe_size = 300

        # Kraken status gate:
        # - "online": trade
        # - "maintenance" / "cancel_only": block
        # - "unknown": allowed after warmup fallback to "online"
        self.kraken_status = "unknown"

        self._last_skip_reason = None

        # === Spike Sniper v5.0 ===
        self.fear_greed = 50  # neutral default (0=extreme fear, 100=extreme greed)
        self.whale_net_flow = 0  # net exchange flow (positive = sell pressure)
        self.fng_symbol = None
        self.whale_symbol = None
        self._spike_entries = {}  # {symbol: True} — track which entries were spike-triggered

        # Spike detection settings
        self.explosion_volume_lookback = 24
        self.explosion_bb_lookback = 12
        self.explosion_high_conviction_threshold = 0.70  # bypass normal scoring
        self.explosion_boost_threshold = 0.40  # boost composite score
        self.spike_position_size_pct = 0.50  # was 0.85; now capped at 50%
        self.spike_stop_loss = 0.08  # wider stop for spike trades
        self.spike_take_profit = 0.25  # higher TP target
        self.spike_trailing_pct = 0.03  # tighter trail after profit
        self.spike_time_exit_hours = 48  # longer time stop

        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(CryptoUniverse.Kraken(self.UniverseFilter))

        try:
            btc = self.AddCrypto("BTCUSD", Resolution.Hour, Market.Kraken)
            self.btc_symbol = btc.Symbol
        except Exception as e:
            self.Debug(f"Warning: Could not add BTC - {e}")

        # Fear & Greed Index (daily, free)
        try:
            from alt_data import FearGreedData
            self.fng_symbol = self.AddData(FearGreedData, "FNG", Resolution.Daily).Symbol
            self.Debug("Added Fear & Greed data feed")
        except Exception as e:
            self.Debug(f"Could not add Fear/Greed data: {e}")
            self.fng_symbol = None

        # Whale Alert (hourly, free tier)
        try:
            from alt_data import WhaleAlertData
            try:
                whale_key = self.GetParameter("whale_alert_api_key") or ""
            except Exception:
                whale_key = ""
            if whale_key:
                self.whale_symbol = self.AddData(WhaleAlertData, "WHALE", Resolution.Hour).Symbol
                self.Debug("Added Whale Alert data feed")
            else:
                self.Debug("Whale Alert API key not set, skipping")
                self.whale_symbol = None
        except Exception as e:
            self.Debug(f"Could not add Whale Alert data: {e}")
            self.whale_symbol = None

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.Rebalance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.CheckExits)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 1), self.DailyReport)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 0), self.ResetDailyCounters)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=6)), self.ReviewPerformance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(12, 0), self.HealthCheck)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=5)), self.ResyncHoldings)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=2)), self.VerifyOrderFills)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=15)), self.PortfolioSanityCheck)

        self.SetWarmUp(timedelta(days=5))
        self.SetSecurityInitializer(lambda security: security.SetSlippageModel(RealisticCryptoSlippage()))
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.Settings.InsightScore = False

        self._scoring_engine = OpusScoringEngine(self)

        if self.LiveMode:
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING (SAFE) v4.0.0 (aggressive + structural) ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max pos: {self.max_positions} | Size: {self.position_size_pct:.0%}")

    def _get_param(self, name, default):
        try:
            param = self.GetParameter(name)
            if param is not None and param != "":
                return float(param)
            return default
        except Exception as e:
            self.Debug(f"Error getting parameter {name}: {e}")
            return default

    def _normalize_order_time(self, order_time):
        """Helper to normalize order time by removing timezone info if present."""
        return normalize_order_time(order_time)
    
    def _record_exit_pnl(self, symbol, entry_price, exit_price):
        """Helper to record PnL from an exit trade. Returns None if prices are invalid."""
        return record_exit_pnl(self, symbol, entry_price, exit_price)

    def EmitInsights(self, *insights):
        return []

    def EmitInsight(self, insight):
        return []

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date = self.Time.date()
        for crypto in self.crypto_data.values():
            crypto['trade_count_today'] = 0
        if len(self._session_blacklist) > 0:
            self.Debug(f"Clearing session blacklist ({len(self._session_blacklist)} items)")
            self._session_blacklist.clear()
        persist_state(self)

    def HealthCheck(self):
        if self.IsWarmingUp: return
        health_check(self)

    def ResyncHoldings(self):
        if self.IsWarmingUp: return
        if not self.LiveMode: return
        resync_holdings_full(self)

    def VerifyOrderFills(self):
        if self.IsWarmingUp: return
        verify_order_fills(self)

    def PortfolioSanityCheck(self):
        if self.IsWarmingUp: return
        portfolio_sanity_check(self)

    def ReviewPerformance(self):
        if self.IsWarmingUp or len(self.trade_log) < 10: return
        review_performance(self)

    def _cancel_stale_orders(self):
        try:
            open_orders = self.Transactions.GetOpenOrders()
            if len(open_orders) > 0:
                self.Debug(f"Found {len(open_orders)} open orders - canceling all...")
                for order in open_orders:
                    self.Transactions.CancelOrder(order.Id)
        except Exception as e:
            self.Debug(f"Error canceling stale orders: {e}")

    def UniverseFilter(self, universe):
        selected = []
        for crypto in universe:
            ticker = crypto.Symbol.Value
            if ticker in SYMBOL_BLACKLIST or ticker in self._session_blacklist:
                continue
            if not ticker.endswith("USD"):
                continue
            if crypto.VolumeInUsd is None or crypto.VolumeInUsd == 0:
                continue
            if crypto.VolumeInUsd > self.min_volume_usd:
                selected.append(crypto)
        selected.sort(key=lambda x: x.VolumeInUsd, reverse=True)
        return [c.Symbol for c in selected[:self.max_universe_size]]

    def _initialize_symbol(self, symbol):
        self.crypto_data[symbol] = {
            'prices': deque(maxlen=self.lookback),
            'returns': deque(maxlen=self.lookback),
            'volume': deque(maxlen=self.lookback),
            'volume_ma': deque(maxlen=self.medium_period),
            'dollar_volume': deque(maxlen=self.lookback),
            'ema_ultra_short': ExponentialMovingAverage(self.ultra_short_period),
            'ema_short': ExponentialMovingAverage(self.short_period),
            'ema_medium': ExponentialMovingAverage(self.medium_period),
            'ema_long': ExponentialMovingAverage(self.long_period),
            'ema_trend': ExponentialMovingAverage(self.trend_period),
            'kama': KaufmanAdaptiveMovingAverage(10, 2, 30),
            'atr': AverageTrueRange(14),
            'volatility': deque(maxlen=self.medium_period),
            'rsi': RelativeStrengthIndex(14),
            'rs_vs_btc': deque(maxlen=self.medium_period),
            'zscore': deque(maxlen=self.short_period),
            'last_price': 0,
            'recent_net_scores': deque(maxlen=3),
            'spreads': deque(maxlen=self.spread_median_window),
            'trail_stop': None,
            'highs': deque(maxlen=self.long_period),
            'lows': deque(maxlen=self.long_period),
            'bb_upper': deque(maxlen=self.short_period),
            'bb_lower': deque(maxlen=self.short_period),
            'bb_width': deque(maxlen=self.medium_period),
            'trade_count_today': 0,
            'last_loss_time': None,
        }

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.crypto_data:
                self._initialize_symbol(symbol)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if not self.IsWarmingUp and is_invested_not_dust(self, symbol):
                smart_liquidate(self, symbol, "Removed from universe")
                # Don't cleanup here — let OnOrderEvent handle it on fill
                self.Debug(f"FORCED EXIT: {symbol.Value} - removed from universe")
            # Only delete crypto_data if not invested (otherwise OnOrderEvent needs it)
            if symbol in self.crypto_data and not is_invested_not_dust(self, symbol):
                del self.crypto_data[symbol]

    def OnData(self, data):
        if self.btc_symbol and data.Bars.ContainsKey(self.btc_symbol):
            btc_bar = data.Bars[self.btc_symbol]
            btc_price = float(btc_bar.Close)
            if len(self.btc_prices) > 0:
                btc_return = (btc_price - self.btc_prices[-1]) / self.btc_prices[-1]
                self.btc_returns.append(btc_return)
            self.btc_prices.append(btc_price)
            if len(self.btc_returns) >= 10:
                self.btc_volatility.append(np.std(list(self.btc_returns)[-10:]))
        # === Consume alternative data feeds ===
        try:
            from alt_data import FearGreedData, WhaleAlertData
            if self.fng_symbol:
                fng_data = data.Get(FearGreedData)
                if fng_data and self.fng_symbol in fng_data:
                    self.fear_greed = fng_data[self.fng_symbol].Value
            if self.whale_symbol:
                whale_data = data.Get(WhaleAlertData)
                if whale_data and self.whale_symbol in whale_data:
                    self.whale_net_flow = whale_data[self.whale_symbol].net_exchange_flow
        except Exception:
            pass  # Degrade gracefully
        for symbol in list(self.crypto_data.keys()):
            if not data.Bars.ContainsKey(symbol):
                continue
            try:
                self._update_symbol_data(symbol, data.Bars[symbol])
            except Exception as e:
                self.Debug(f"Error updating symbol data for {symbol.Value}: {e}")
                pass
        if self.IsWarmingUp:
            return
        if not self._positions_synced:
            if not self._first_post_warmup:
                self._cancel_stale_orders()
            sync_existing_positions(self)
            self._positions_synced = True
            self._first_post_warmup = False
            # Fallback: if status never set, assume online after warmup
            if self.kraken_status == "unknown":
                self.kraken_status = "online"
                self.Debug("Fallback: kraken_status set to online after warmup")
            ready_count = sum(1 for c in self.crypto_data.values() if self._is_ready(c))
            self.Debug(f"Post-warmup: {ready_count} symbols ready")
        self._update_market_context()

    def _update_symbol_data(self, symbol, bar):
        crypto = self.crypto_data[symbol]
        price = float(bar.Close)
        high = float(bar.High)
        low = float(bar.Low)
        volume = float(bar.Volume)
        crypto['prices'].append(price)
        crypto['highs'].append(high)
        crypto['lows'].append(low)
        if crypto['last_price'] > 0:
            ret = (price - crypto['last_price']) / crypto['last_price']
            crypto['returns'].append(ret)
        crypto['last_price'] = price
        crypto['volume'].append(volume)
        crypto['dollar_volume'].append(price * volume)
        if len(crypto['volume']) >= self.short_period:
            crypto['volume_ma'].append(np.mean(list(crypto['volume'])[-self.short_period:]))
        crypto['ema_ultra_short'].Update(bar.EndTime, price)
        crypto['ema_short'].Update(bar.EndTime, price)
        crypto['ema_medium'].Update(bar.EndTime, price)
        crypto['ema_long'].Update(bar.EndTime, price)
        crypto['ema_trend'].Update(bar.EndTime, price)
        crypto['kama'].Update(bar.EndTime, price)
        crypto['atr'].Update(bar)
        if len(crypto['returns']) >= 10:
            crypto['volatility'].append(np.std(list(crypto['returns'])[-10:]))
        crypto['rsi'].Update(bar.EndTime, price)
        if len(crypto['returns']) >= self.short_period and len(self.btc_returns) >= self.short_period:
            coin_ret = np.sum(list(crypto['returns'])[-self.short_period:])
            btc_ret = np.sum(list(self.btc_returns)[-self.short_period:])
            crypto['rs_vs_btc'].append(coin_ret - btc_ret)
        if len(crypto['prices']) >= self.medium_period:
            prices_arr = np.array(list(crypto['prices'])[-self.medium_period:])
            std = np.std(prices_arr)
            mean = np.mean(prices_arr)
            if std > 0:
                crypto['zscore'].append((price - mean) / std)
                crypto['bb_upper'].append(mean + 2 * std)
                crypto['bb_lower'].append(mean - 2 * std)
                crypto['bb_width'].append(4 * std / mean if mean > 0 else 0)
        sp = get_spread_pct(self, symbol)
        if sp is not None:
            crypto['spreads'].append(sp)

    def _update_market_context(self):
        if len(self.btc_prices) >= 72:
            btc_arr = np.array(list(self.btc_prices))
            btc_sma = np.mean(btc_arr[-72:])
            current_btc = btc_arr[-1]
            if current_btc > btc_sma * 1.05:
                new_regime = "bull"
            elif current_btc < btc_sma * 0.95:
                new_regime = "bear"
            else:
                new_regime = "sideways"
            # Momentum confirmation: positive BTC returns over last 12-24 bars bias toward bull
            if new_regime == "sideways" and len(self.btc_returns) >= 24:
                btc_mom_12 = np.mean(list(self.btc_returns)[-12:])
                btc_mom_24 = np.mean(list(self.btc_returns)[-24:])
                if btc_mom_12 > 0 and btc_mom_24 > 0:
                    new_regime = "bull"
                elif btc_mom_12 < 0 and btc_mom_24 < 0:
                    new_regime = "bear"
            # Hysteresis: only change if held for 6+ bars
            if new_regime != self.market_regime:
                self._regime_hold_count += 1
                if self._regime_hold_count >= 6:
                    self.market_regime = new_regime
                    self._regime_hold_count = 0
            else:
                self._regime_hold_count = 0
        if len(self.btc_volatility) >= 5:
            current_vol = self.btc_volatility[-1]
            avg_vol = np.mean(list(self.btc_volatility))
            if current_vol > avg_vol * 1.5:
                self.volatility_regime = "high"
            elif current_vol < avg_vol * 0.5:
                self.volatility_regime = "low"
            else:
                self.volatility_regime = "normal"
        uptrend_count = 0
        total_ready = 0
        for crypto in self.crypto_data.values():
            if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
                total_ready += 1
                if crypto['ema_short'].Current.Value > crypto['ema_medium'].Current.Value:
                    uptrend_count += 1
        if total_ready > 10:
            self.market_breadth = uptrend_count / total_ready

    def _annualized_vol(self, crypto):
        if crypto is None:
            return None
        if len(crypto.get('volatility', [])) == 0:
            return None
        return float(crypto['volatility'][-1]) * self.sqrt_annualization

    def _compute_portfolio_risk_estimate(self):
        total_value = self.Portfolio.TotalPortfolioValue
        if total_value <= 0:
            return 0.0
        risk = 0.0
        for kvp in self.Portfolio:
            symbol, holding = kvp.Key, kvp.Value
            if not is_invested_not_dust(self, symbol):
                continue
            crypto = self.crypto_data.get(symbol)
            asset_vol_ann = self._annualized_vol(crypto)
            if asset_vol_ann is None:
                asset_vol_ann = self.min_asset_vol_floor
            weight = abs(holding.HoldingsValue) / total_value
            risk += weight * asset_vol_ann
        return risk

    def _normalize(self, v, mn, mx):
        if mx - mn <= 0:
            return 0.5
        return max(0, min(1, (v - mn) / (mx - mn)))

    def _calculate_factor_scores(self, symbol, crypto):
        return self._scoring_engine.calculate_factor_scores(symbol, crypto)

    def _get_regime_weights(self):
        """Return signal weights adapted to the current market regime."""
        if self.market_regime == "bear":
            return {
                'mean_reversion': 0.30, 'relative_strength': 0.10, 'trend_strength': 0.10,
                'risk_adjusted_momentum': 0.20, 'volume_momentum': 0.15, 'liquidity': 0.15,
            }
        elif self.market_regime == "sideways":
            return {
                'mean_reversion': 0.25, 'relative_strength': 0.15, 'trend_strength': 0.10,
                'risk_adjusted_momentum': 0.20, 'volume_momentum': 0.15, 'liquidity': 0.15,
            }
        return dict(self.weights)  # bull: use static weights

    def _calculate_composite_score(self, factors, crypto=None):
        return self._scoring_engine.calculate_composite_score(factors, crypto)

    def _apply_fee_adjustment(self, score):
        return score - (self.expected_round_trip_fees * 1.1 + self.fee_slippage_buffer)

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        if self.market_regime == "bull":
            base_size = 0.40
        elif self.market_regime == "sideways":
            base_size = 0.28
        else:
            base_size = 0.22
        conviction = max(0.7, min(1.3, 0.7 + (score - threshold) * 3.0))
        vol_floor = max(asset_vol_ann or 0.05, 0.05)
        vol_scale = max(0.6, min(1.2, self.target_position_ann_vol / vol_floor))
        kelly = self._kelly_fraction()
        return base_size * conviction * vol_scale * kelly

    def _kelly_fraction(self):
        return kelly_fraction(self)

    def _get_max_daily_trades(self):
        """Return max daily trades adapted to market regime."""
        if self.market_regime == "bear":
            return 2
        elif self.market_regime == "sideways":
            return 3
        return 8  # bull

    def _check_correlation(self, new_symbol):
        """Reject candidate if it is too correlated with any existing position (item 8)."""
        if not self.entry_prices:
            return True
        new_crypto = self.crypto_data.get(new_symbol)
        if not new_crypto or len(new_crypto['returns']) < 24:
            return True
        new_rets = np.array(list(new_crypto['returns'])[-24:])
        if np.std(new_rets) < 1e-10:
            return True
        for sym in list(self.entry_prices.keys()):
            if sym == new_symbol:
                continue
            existing = self.crypto_data.get(sym)
            if not existing or len(existing['returns']) < 24:
                continue
            exist_rets = np.array(list(existing['returns'])[-24:])
            if np.std(exist_rets) < 1e-10:
                continue
            try:
                corr = np.corrcoef(new_rets, exist_rets)[0, 1]
                if corr > 0.85:
                    return False
            except Exception:
                continue
        return True

    def _log_skip(self, reason):
        if self.LiveMode:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason
        elif reason != self._last_skip_reason:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        
        # Cash mode — pause trading when recent performance is poor
        if self._cash_mode_until is not None and self.Time < self._cash_mode_until:
            self._log_skip("cash mode - poor recent performance")
            return
        
        # Reset log budget at each rebalance call for consistent logging
        self.log_budget = 20
        
        # Check rate limit hard block
        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            self._log_skip("rate limited")
            return
        
        # Live safety checks
        if self.LiveMode and not live_safety_checks(self):
            return
        # Only block on explicit bad states; unknown is allowed (and will have fallback after warmup)
        if self.LiveMode and self.kraken_status in ("maintenance", "cancel_only"):
            self._log_skip("kraken not online")
            return
        cancel_stale_new_orders(self)
        if self.LiveMode and self.Time.hour in self.skip_hours_utc:
            self._log_skip("skip hour")
            return
        if self.daily_trade_count >= self._get_max_daily_trades():
            self._log_skip("max daily trades")
            return
        val = self.Portfolio.TotalPortfolioValue
        if self.peak_value is None or self.peak_value < 1:
            self.peak_value = val
        if self.drawdown_cooldown > 0:
            self.drawdown_cooldown -= 2
            if self.drawdown_cooldown <= 0:
                self.peak_value = val
                self.consecutive_losses = 0
            else:
                self._log_skip(f"drawdown cooldown {self.drawdown_cooldown}h")
                return
        self.peak_value = max(self.peak_value, val)
        dd = (self.peak_value - val) / self.peak_value if self.peak_value > 0 else 0
        if dd > self.max_drawdown_limit:
            self.drawdown_cooldown = self.cooldown_hours
            self._log_skip(f"drawdown {dd:.1%} > limit")
            return
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.drawdown_cooldown = 12
            self.consecutive_losses = 0
            self._log_skip("consecutive loss cooldown")
            return
        # Fear & Greed hard gate (item 9)
        if self.fear_greed > 85:
            dynamic_max_pos = 1  # extreme greed: de-risk
        elif self.fear_greed < 15:
            dynamic_max_pos = self.base_max_positions + 1  # extreme fear: contrarian
        else:
            dynamic_max_pos = self.base_max_positions
        pos_count = get_actual_position_count(self)
        if pos_count >= dynamic_max_pos:
            self._log_skip("at max positions")
            return
        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            self._log_skip("too many open orders")
            return
        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap:
            self._log_skip("risk cap")
            return
        
        # Diagnostic counters for filter funnel
        total_symbols = len(self.crypto_data)
        count_not_blacklisted = 0
        count_no_open_orders = 0
        count_spread_ok = 0
        count_ready = 0
        count_scored = 0
        count_above_thresh = 0
        
        scores = []
        threshold_now = self._get_threshold()
        for symbol in list(self.crypto_data.keys()):
            if symbol.Value in SYMBOL_BLACKLIST or symbol.Value in self._session_blacklist:
                continue
            count_not_blacklisted += 1
            
            if has_open_orders(self, symbol):
                continue
            count_no_open_orders += 1
            
            if not spread_ok(self, symbol):
                continue
            count_spread_ok += 1
            
            crypto = self.crypto_data[symbol]
            if not self._is_ready(crypto):
                continue
            count_ready += 1
            
            factor_scores = self._calculate_factor_scores(symbol, crypto)
            if not factor_scores:
                continue
            count_scored += 1
            
            composite_score = self._calculate_composite_score(factor_scores, crypto)
            net_score = self._apply_fee_adjustment(composite_score)

            # Calculate snipe score (replaces old explosion-only detection)
            try:
                snipe_score, is_snipe, snipe_components = self._scoring_engine.calculate_snipe_score(crypto)
            except Exception:
                snipe_score, is_snipe, snipe_components = 0.0, False, {}

            is_spike = False
            explosion = snipe_components.get('explosion', 0.0)

            # HIGH CONVICTION SNIPE: bypass normal scoring
            if snipe_score > 0.65:
                net_score = max(net_score, 0.80)
                is_spike = True
            # MEDIUM CONVICTION: boost composite score proportionally
            elif snipe_score > 0.40:
                boost = 1.0 + snipe_score * 0.6
                composite_score *= boost
                net_score = self._apply_fee_adjustment(composite_score)
                is_spike = snipe_score > 0.50
            
            # Populate recent_net_scores for persistence filter
            crypto['recent_net_scores'].append(net_score)
            
            min_conviction_gap = 0.05 if self.market_regime != "bull" else 0.02
            if net_score > threshold_now + min_conviction_gap:
                count_above_thresh += 1
                scores.append({
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'net_score': net_score,
                    'factors': factor_scores,
                    'volatility': crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                    'dollar_volume': list(crypto['dollar_volume'])[-6:] if len(crypto['dollar_volume']) >= 6 else [],
                    'is_spike': is_spike,
                    'explosion': explosion,
                    'snipe_score': snipe_score,
                    'snipe_components': snipe_components,
                })
        
        # Log diagnostic summary
        try:
            cash = self.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            cash = self.Portfolio.Cash
        
        debug_limited(self, f"REBALANCE: {count_above_thresh}/{count_scored} above thresh={threshold_now:.3f} | cash=${cash:.2f}")
        
        if len(scores) == 0:
            self._log_skip("no candidates passed filters")
            return
        scores.sort(key=lambda x: x['net_score'], reverse=True)
        self._last_skip_reason = None
        self._execute_trades(scores, threshold_now, dynamic_max_pos)

    def _get_open_buy_orders_value(self):
        """Calculate total value reserved by open buy orders."""
        return get_open_buy_orders_value(self)

    def _execute_trades(self, candidates, threshold_now, dynamic_max_pos):
        if not self._positions_synced:
            return
        if self.LiveMode and self.kraken_status in ("maintenance", "cancel_only"):
            return
        cancel_stale_new_orders(self)
        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            return
        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap:
            return
        
        try:
            available_cash = self.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_cash = self.Portfolio.Cash
        
        open_buy_orders_value = self._get_open_buy_orders_value()
        
        if available_cash <= 0:
            debug_limited(self, f"SKIP TRADES: No cash available (${available_cash:.2f})")
            return
        if open_buy_orders_value > available_cash * self.open_orders_cash_threshold:
            debug_limited(self, f"SKIP TRADES: ${open_buy_orders_value:.2f} reserved (>{self.open_orders_cash_threshold:.0%} of ${available_cash:.2f})")
            return
        
        reject_pending_orders = 0
        reject_open_orders = 0
        reject_already_invested = 0
        reject_spread = 0
        reject_exit_cooldown = 0
        reject_loss_cooldown = 0
        reject_price_invalid = 0
        reject_price_too_low = 0
        reject_cash_reserve = 0
        reject_min_qty_too_large = 0
        reject_ema_gate = 0
        reject_recent_net_scores = 0
        reject_dollar_volume = 0
        reject_impact_ratio = 0
        reject_notional = 0
        success_count = 0
        
        for cand in candidates:
            if self.daily_trade_count >= self._get_max_daily_trades():
                break
            if get_actual_position_count(self) >= dynamic_max_pos:
                break
            sym = cand['symbol']
            comp_score = cand.get('composite_score', 0.5)
            net_score = cand.get('net_score', 0.5)
            is_spike = cand.get('is_spike', False)
            explosion = cand.get('explosion', 0.0)
            if sym in self._pending_orders and self._pending_orders[sym] > 0:
                reject_pending_orders += 1
                continue
            if has_open_orders(self, sym):
                reject_open_orders += 1
                continue
            if is_invested_not_dust(self, sym):
                reject_already_invested += 1
                continue
            if not spread_ok(self, sym):
                reject_spread += 1
                continue
            if sym in self._exit_cooldowns and self.Time < self._exit_cooldowns[sym]:
                reject_exit_cooldown += 1
                continue
            if sym in self._symbol_loss_cooldowns and self.Time < self._symbol_loss_cooldowns[sym]:
                reject_loss_cooldown += 1
                continue
            sec = self.Securities[sym]
            price = sec.Price
            if price is None or price <= 0:
                reject_price_invalid += 1
                continue
            if price < self.min_price_usd:
                reject_price_too_low += 1
                continue

            effective_size_cap = self.position_size_pct
            if self.LiveMode and (self.volatility_regime == "high" or self.market_regime == "sideways"):
                effective_size_cap = min(effective_size_cap, 0.25)
            # Cap position size for small portfolios to limit concentration risk
            total_value = self.Portfolio.TotalPortfolioValue
            if total_value < 100:
                effective_size_cap = min(effective_size_cap, 0.20)
            # Sentiment-driven position size caps
            if self.fear_greed < 25:
                effective_size_cap = max(effective_size_cap, 0.70)  # contrarian: allow larger size
            elif self.fear_greed > 75:
                effective_size_cap = min(effective_size_cap, 0.35)  # cautious during greed
            # Whale sell pressure gate: skip unless high-conviction spike
            if self.whale_net_flow > 5 and not (is_spike and explosion > 0.6):
                continue

            try:
                available_cash = self.Portfolio.CashBook["USD"].Amount
            except (KeyError, AttributeError):
                available_cash = self.Portfolio.Cash
            
            if open_buy_orders_value > available_cash:
                self.Debug(f"⚠️ CASH INCONSISTENCY: ${open_buy_orders_value:.2f} reserved but only ${available_cash:.2f} available")
            available_cash = max(0, available_cash - open_buy_orders_value)
            
            portfolio_reserve = total_value * self.cash_reserve_pct
            fee_reserve = total_value * 0.02
            effective_reserve = max(portfolio_reserve, fee_reserve)
            reserved_cash = available_cash - effective_reserve
            if reserved_cash <= 0:
                reject_cash_reserve += 1
                continue
            min_qty = get_min_quantity(self, sym)
            min_notional_usd = get_min_notional_usd(self, sym)
            if min_qty * price > reserved_cash * 0.85:
                reject_min_qty_too_large += 1
                continue
            crypto = self.crypto_data.get(sym)
            if not crypto:
                continue

            # Per-symbol daily trade limit: max entries per symbol per day
            if crypto.get('trade_count_today', 0) >= self.max_symbol_trades_per_day:
                continue

            # Snipe bypass: high-conviction snipes skip EMA and momentum gates
            is_snipe_bypass = cand.get('snipe_score', 0) > 0.60

            if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
                ema_short = crypto['ema_short'].Current.Value
                ema_medium = crypto['ema_medium'].Current.Value

                is_mean_reversion = False
                if len(crypto['zscore']) >= 1 and crypto['rsi'].IsReady:
                    z = crypto['zscore'][-1]
                    rsi = crypto['rsi'].Current.Value
                    if z < -1.5 and rsi < 35:
                        is_mean_reversion = True

                if not is_mean_reversion and not is_snipe_bypass:
                    if ema_short <= ema_medium:
                        # Soft gate in ALL regimes: apply 0.80 penalty instead of hard reject
                        penalized_score = net_score * 0.80
                        if penalized_score <= threshold_now:
                            reject_ema_gate += 1
                            continue
                        net_score = penalized_score

            if len(crypto['recent_net_scores']) >= 3:
                above_threshold_count = sum(1 for score in list(crypto['recent_net_scores'])[-3:] if score > threshold_now)
                if above_threshold_count == 0:
                    reject_recent_net_scores += 1
                    continue
            
            vol = self._annualized_vol(crypto)
            size = self._calculate_position_size(net_score, threshold_now, vol)
            if is_spike:
                size = min(size * 1.3, 0.50)  # boost by 30%, cap at 50%
            size = min(size, effective_size_cap)
            if self.volatility_regime == "high":
                size *= 0.7
            
            slippage_penalty = get_slippage_penalty(self, sym)
            size *= slippage_penalty
            if slippage_penalty <= 0.3:
                self.Debug(f"⚠️ HIGH SLIPPAGE PENALTY: {sym.Value} | size reduced to {slippage_penalty:.0%}")

            if self.LiveMode and len(crypto['dollar_volume']) >= 6:
                recent_dollar_vol6 = np.mean(list(crypto['dollar_volume'])[-6:])
                if recent_dollar_vol6 < 5000:
                    reject_dollar_volume += 1
                    continue
            recent_dollar_vol3 = np.mean(list(crypto['dollar_volume'])[-3:]) if len(crypto['dollar_volume']) >= 3 else 0

            if len(crypto['dollar_volume']) >= 3:
                order_value_estimate = reserved_cash * size
                if recent_dollar_vol3 > 0:
                    impact_ratio = order_value_estimate / recent_dollar_vol3
                    portfolio_value = self.Portfolio.TotalPortfolioValue
                    impact_hard_cap = 0.05 if portfolio_value < 500 else (0.03 if portfolio_value < 5000 else 0.02)
                    impact_soft_cap = impact_hard_cap * 0.6
                    if impact_ratio > impact_hard_cap:
                        reject_impact_ratio += 1
                        continue
                    if impact_ratio > impact_soft_cap:
                        size *= max(0.3, 1.0 - impact_ratio)
                    max_child = 0.15 * recent_dollar_vol3
                    if order_value_estimate > max_child:
                        size *= max_child / order_value_estimate

            val = reserved_cash * size
            qty = round_quantity(self, sym, val / price)
            if qty < min_qty:
                qty = round_quantity(self, sym, min_qty)
                val = qty * price
            total_cost_with_fee = val * 1.006
            if total_cost_with_fee > available_cash - fee_reserve:
                reject_cash_reserve += 1
                continue
            if val < min_notional_usd or val < self.min_notional or val > reserved_cash:
                reject_notional += 1
                continue
            # Correlation gate: reject if too correlated with existing positions (item 8)
            if not self._check_correlation(sym):
                continue
            try:
                # Use limit orders only in live mode
                if self.LiveMode:
                    ticket = place_limit_or_market(self, sym, qty, timeout_seconds=60, tag="Entry")
                else:
                    ticket = self.MarketOrder(sym, qty, tag="Entry")
                if ticket is not None:
                    self._recent_tickets.append(ticket)
                    self.Debug(f"ORDER: {sym.Value} | ${val:.2f} | id={ticket.OrderId}")
                    success_count += 1
                    self.trade_count += 1
                    crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1
                    if is_spike:
                        self._spike_entries[sym] = True
                        components = cand.get('snipe_components', {})
                        self.Debug(f"🎯 SNIPE ENTRY: {sym.Value} | snipe={cand.get('snipe_score', 0):.2f} | "
                                   f"acc={components.get('accumulation', 0):.2f} "
                                   f"rel={components.get('relative_outperf', 0):.2f} "
                                   f"smf={components.get('smart_money', 0):.2f} "
                                   f"exp={components.get('explosion', 0):.2f}")
                    if self.LiveMode:
                        self._last_live_trade_time = self.Time
            except Exception as e:
                self.Debug(f"ORDER FAILED: {sym.Value} - {e}")
                self._session_blacklist.add(sym.Value)
                continue
            if self.LiveMode and success_count >= 2:
                break
        
        if success_count > 0 or reject_ema_gate + reject_recent_net_scores + reject_impact_ratio + reject_loss_cooldown > 5:
            debug_limited(self, f"EXECUTE: {success_count}/{len(candidates)} | rejects: ema={reject_ema_gate} scores={reject_recent_net_scores} impact={reject_impact_ratio} loss_cd={reject_loss_cooldown}")

    def _get_threshold(self):
        if self.market_regime == "bull" and self.market_breadth > 0.6:
            base = self.threshold_bull
        elif self.market_regime == "bear":
            base = self.threshold_bear
        elif self.volatility_regime == "high":
            base = self.threshold_high_vol
        else:
            base = self.threshold_sideways

        # Sentiment overlay from Fear & Greed
        if self.fear_greed < 20:
            base *= 0.93  # Lower bar during extreme fear (contrarian buy)
        elif self.fear_greed > 80:
            base *= 1.07  # Higher bar during extreme greed (cautious)

        # Whale flow overlay - heavy exchange inflows = sell pressure
        if self.whale_net_flow > 5:
            base *= 1.05  # Raise bar when sell pressure detected
        elif self.whale_net_flow < -3:
            base *= 0.97  # Lower bar when accumulation detected

        return base

    def _is_ready(self, c):
        return len(c['prices']) >= 14 and c['rsi'].IsReady

    def CheckExits(self):
        if self.IsWarmingUp:
            return
        # Check rate limit hard block
        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            return
        for kvp in self.Portfolio:
            if not is_invested_not_dust(self, kvp.Key):
                continue
            self._check_exit(kvp.Key, self.Securities[kvp.Key].Price, kvp.Value)

    def _check_exit(self, symbol, price, holding):
        if len(self.Transactions.GetOpenOrders(symbol)) > 0:
            return
        if symbol in self._cancel_cooldowns and self.Time < self._cancel_cooldowns[symbol]:
            return
        if symbol not in self.entry_prices:
            self.entry_prices[symbol] = holding.AveragePrice
            self.highest_prices[symbol] = holding.AveragePrice
            self.entry_times[symbol] = self.Time
        entry = self.entry_prices[symbol]
        highest = self.highest_prices.get(symbol, entry)
        if price > highest:
            self.highest_prices[symbol] = price
        pnl = (price - entry) / entry if entry > 0 else 0
        
        exit_slip_estimate = 0.0
        crypto = self.crypto_data.get(symbol)
        if crypto and len(crypto.get('dollar_volume', [])) >= 6:
            dv_list = list(crypto['dollar_volume'])[-6:]
            avg_dv = np.mean(dv_list)
            exit_value = abs(holding.Quantity) * price
            if avg_dv > 0 and exit_value / avg_dv > 0.02:
                exit_slip_estimate = min(0.02, exit_value / avg_dv * 0.1)
                pnl -= exit_slip_estimate
        
        dd = (highest - price) / highest if highest > 0 else 0
        hours = (self.Time - self.entry_times.get(symbol, self.Time)).total_seconds() / 3600
        # ATR-based stops as primary (item 3): ATR is primary, absolute values are floors
        atr = crypto['atr'].Current.Value if crypto and crypto['atr'].IsReady else None
        if atr and entry > 0:
            sl = (atr * self.atr_sl_mult) / entry
            tp = (atr * self.atr_tp_mult) / entry
        else:
            sl, tp = self.base_stop_loss, self.base_take_profit
        sl = max(sl, 0.015)  # 1.5% floor
        tp = max(tp, 0.03)   # 3% floor
        # Enforce minimum 1.5:1 R:R
        if tp < sl * 1.5:
            tp = sl * 1.5
        if self.volatility_regime == "high":
            sl *= 1.2; tp *= 1.3
        elif self.market_regime == "bear":
            sl *= 0.8; tp *= 0.7

        trailing_activation = self.trailing_activation
        trailing_stop_pct = self.trailing_stop_pct
        if self.volatility_regime == "high" or self.market_regime == "sideways":
            trailing_activation = 0.03
            trailing_stop_pct = 0.02

        # Spike-specific exit parameters
        is_spike_entry = symbol in self._spike_entries
        if is_spike_entry:
            sl = max(sl, self.spike_stop_loss)
            tp = max(tp, self.spike_take_profit)
            trailing_stop_pct = self.spike_trailing_pct
            time_exit_hours = self.spike_time_exit_hours
            time_exit_pnl_thresh = 0.01
            # Item 6: tighten stops once spike has achieved 5%+ profit
            if pnl > 0.05:
                trailing_stop_pct = 0.02
                trailing_activation = min(trailing_activation, 0.02)
        else:
            # Regime-adaptive time stop (item 4)
            if self.market_regime == "bull":
                time_exit_hours = 36
                time_exit_pnl_thresh = 0.005
            elif self.market_regime == "sideways":
                time_exit_hours = 72
                time_exit_pnl_thresh = 0.003
            else:  # bear or unknown
                time_exit_hours = 24
                time_exit_pnl_thresh = 0.003

        # Fear & Greed: tighten trailing stops during extreme greed
        if self.fear_greed > 75:
            trailing_stop_pct *= 0.80

        tag = ""
        min_notional_usd = get_min_notional_usd(self, symbol)
        trailing_allowed = hours >= self.trailing_grace_hours
        if pnl <= -sl:
            tag = "Stop Loss"
        elif pnl >= tp:
            tag = "Take Profit"
        elif trailing_allowed and pnl > trailing_activation and dd >= trailing_stop_pct:
            tag = "Trailing Stop"
        elif self.market_regime == "bear" and pnl > 0.05:
            tag = "Bear Exit"
        elif self.whale_net_flow > 8 and hours > 12 and pnl < 0.02:
            tag = "Whale Exit"
        elif is_spike_entry and hours > 12 and pnl < 0.02:
            tag = "Spike Fizzle"
        elif hours > time_exit_hours and pnl < time_exit_pnl_thresh:
            tag = "Time Exit"
        if not tag and trailing_allowed and atr and entry > 0 and holding.Quantity != 0:
            trail_offset = atr * self.atr_trail_mult
            trail_level = price - trail_offset
            current_trail = crypto.get('trail_stop') if crypto else None
            if current_trail is None or trail_level > current_trail:
                crypto['trail_stop'] = trail_level
            if crypto and crypto['trail_stop'] is not None:
                if holding.Quantity > 0 and price <= crypto['trail_stop']:
                    tag = "ATR Trail"
                elif holding.Quantity < 0 and price >= crypto['trail_stop']:
                    tag = "ATR Trail"
        if not tag and hours >= self.min_signal_age_hours:
            try:
                if crypto and self._is_ready(crypto):
                    # Require 3 consecutive below-threshold readings before triggering Signal Decay
                    threshold = self._get_threshold()
                    decay_threshold = threshold - self.signal_decay_buffer
                    recent_scores = list(crypto.get('recent_net_scores', []))
                    if len(recent_scores) >= 3 and all(s < decay_threshold for s in recent_scores[-3:]):
                        tag = "Signal Decay"
            except Exception as e:
                pass
        if tag:
            if price * abs(holding.Quantity) < min_notional_usd * 0.9:
                return
            if pnl < 0:
                self._symbol_loss_cooldowns[symbol] = self.Time + timedelta(hours=12)
            smart_liquidate(self, symbol, tag)
            self._exit_cooldowns[symbol] = self.Time + timedelta(hours=self.exit_cooldown_hours)
            self.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.0f}h")

    def OnOrderEvent(self, event):
        try:
            symbol = event.Symbol
            self.Debug(f"ORDER: {symbol.Value} {event.Status} {event.Direction} qty={event.FillQuantity or event.Quantity} price={event.FillPrice} id={event.OrderId}")
            if event.Status == OrderStatus.Submitted:
                if symbol not in self._pending_orders:
                    self._pending_orders[symbol] = 0
                intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
                self._pending_orders[symbol] += intended_qty
                if symbol not in self._submitted_orders:
                    has_position = symbol in self.Portfolio and self.Portfolio[symbol].Invested
                    if event.Direction == OrderDirection.Sell and has_position:
                        inferred_intent = 'exit'
                    elif event.Direction == OrderDirection.Buy and not has_position:
                        inferred_intent = 'entry'
                    else:
                        inferred_intent = 'entry' if event.Direction == OrderDirection.Buy else 'exit'
                    
                    self._submitted_orders[symbol] = {
                        'order_id': event.OrderId,
                        'time': self.Time,
                        'quantity': event.Quantity,
                        'intent': inferred_intent
                    }
                else:
                    self._submitted_orders[symbol]['order_id'] = event.OrderId
            elif event.Status == OrderStatus.PartiallyFilled:
                if symbol in self._pending_orders:
                    self._pending_orders[symbol] -= abs(event.FillQuantity)
                    if self._pending_orders[symbol] <= 0:
                        self._pending_orders.pop(symbol, None)
                if event.Direction == OrderDirection.Buy:
                    if symbol not in self.entry_prices:
                        self.entry_prices[symbol] = event.FillPrice
                        self.highest_prices[symbol] = event.FillPrice
                        self.entry_times[symbol] = self.Time
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Filled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)  # Remove from verification tracking
                self._order_retries.pop(event.OrderId, None)  # Clean up retry tracking
                if event.Direction == OrderDirection.Buy:
                    self.entry_prices[symbol] = event.FillPrice
                    self.highest_prices[symbol] = event.FillPrice
                    self.entry_times[symbol] = self.Time
                    self.daily_trade_count += 1
                else:
                    entry = self.entry_prices.get(symbol, None)
                    if entry is None:
                        entry = event.FillPrice
                        self.Debug(f"⚠️ WARNING: Missing entry price for {symbol.Value} sell, using fill price")
                    pnl = (event.FillPrice - entry) / entry if entry > 0 else 0
                    self._rolling_wins.append(1 if pnl > 0 else 0)
                    self._recent_trade_outcomes.append(1 if pnl > 0 else 0)
                    if pnl > 0:
                        self._rolling_win_sizes.append(pnl)
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                    else:
                        self._rolling_loss_sizes.append(abs(pnl))
                        self.losing_trades += 1
                        self.consecutive_losses += 1
                    self.total_pnl += pnl
                    self.trade_log.append({
                        'time': self.Time,
                        'symbol': symbol.Value,
                        'pnl_pct': pnl,
                        'exit_reason': 'filled_sell',
                    })
                    # Activate cash mode if recent win rate is very low
                    if len(self._recent_trade_outcomes) >= 8:
                        recent_wr = sum(self._recent_trade_outcomes) / len(self._recent_trade_outcomes)
                        if recent_wr < 0.25:
                            self._cash_mode_until = self.Time + timedelta(hours=24)
                            self.Debug(f"⚠️ CASH MODE: WR={recent_wr:.0%} over {len(self._recent_trade_outcomes)} trades. Pausing 24h.")
                    cleanup_position(self, symbol)
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Canceled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                if event.Direction == OrderDirection.Sell and symbol not in self.entry_prices:
                    if is_invested_not_dust(self, symbol):
                        holding = self.Portfolio[symbol]
                        self.entry_prices[symbol] = holding.AveragePrice
                        self.highest_prices[symbol] = holding.AveragePrice
                        self.entry_times[symbol] = self.Time
                        self.Debug(f"RE-TRACKED after cancel: {symbol.Value}")
            elif event.Status == OrderStatus.Invalid:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                if event.Direction == OrderDirection.Sell and symbol not in self.entry_prices:
                    if is_invested_not_dust(self, symbol):
                        holding = self.Portfolio[symbol]
                        self.entry_prices[symbol] = holding.AveragePrice
                        self.highest_prices[symbol] = holding.AveragePrice
                        self.entry_times[symbol] = self.Time
                        self.Debug(f"RE-TRACKED after invalid: {symbol.Value}")
                self._session_blacklist.add(symbol.Value)
        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")
        if self.LiveMode:
            persist_state(self)
        
    def OnBrokerageMessage(self, message):
        try:
            txt = message.Message.lower()
            if "system status:" in txt:
                if "online" in txt:
                    self.kraken_status = "online"
                elif "maintenance" in txt:
                    self.kraken_status = "maintenance"
                elif "cancel_only" in txt:
                    self.kraken_status = "cancel_only"
                elif "post_only" in txt:
                    self.kraken_status = "post_only"
                else:
                    self.kraken_status = "unknown"
                self.Debug(f"Kraken status update: {self.kraken_status}")
            
            if "rate limit" in txt or "too many" in txt:
                self.Debug(f"⚠️ RATE LIMIT - pausing {self.rate_limit_cooldown_minutes}min")
                self._rate_limit_until = self.Time + timedelta(minutes=self.rate_limit_cooldown_minutes)
                self._last_live_trade_time = self.Time
        except Exception as e:
            self.Debug(f"BrokerageMessage parse error: {e}")

    def OnEndOfAlgorithm(self):
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        self.Debug("=== FINAL ===")
        self.Debug(f"Trades: {self.trade_count} | WR: {wr:.1%}")
        self.Debug(f"Final: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug(f"PnL: {self.total_pnl:+.2%}")
        persist_state(self)

    def DailyReport(self):
        if self.IsWarmingUp: return
        daily_report(self)
