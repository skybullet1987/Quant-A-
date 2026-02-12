# region imports
from AlgorithmImports import *
from execution import *
from collections import deque
import statistics
import numpy as np
# endregion

class SimplifiedCryptoStrategy(QCAlgorithm):
    """
    v3.0.0 (qa-logic + opus-execution)
    Combines main_qa signal/scoring with opus execution enhancements.
    Single inheritance only - uses composition pattern for execution functions.
    """

    def Initialize(self):
        self.SetStartDate(2025, 1, 1)
        self.SetCash(20)
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        self.threshold_bull = self._get_param("threshold_bull", 0.50)
        self.threshold_bear = self._get_param("threshold_bear", 0.62)
        self.threshold_sideways = self._get_param("threshold_sideways", 0.55)
        self.threshold_high_vol = self._get_param("threshold_high_vol", 0.58)
        self.trailing_activation = self._get_param("trailing_activation", 0.05)
        self.trailing_stop_pct = self._get_param("trailing_stop_pct", 0.03)
        self.base_stop_loss = self._get_param("base_stop_loss", 0.05)
        self.base_take_profit = self._get_param("base_take_profit", 0.12)
        self.atr_sl_mult = self._get_param("atr_sl_mult", 1.6)
        self.atr_tp_mult = self._get_param("atr_tp_mult", 3.0)

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.25)
        self.portfolio_vol_cap = self._get_param("portfolio_vol_cap", 0.35)
        self.signal_decay_buffer = self._get_param("signal_decay_buffer", 0.05)
        self.min_signal_age_hours = self._get_param("signal_decay_min_hours", 6)
        self.cash_reserve_pct = 0.15

        self.short_period = 6
        self.medium_period = 24
        self.long_period = 72
        self.lookback = 72
        self.sqrt_annualization = np.sqrt(24 * 365)
        self.min_asset_vol_floor = 0.05

        self.base_max_positions = 2
        self.max_positions = self.base_max_positions
        self.position_size_pct = 0.45
        self.min_notional = 5.0
        self.min_price_usd = 0.005

        self.expected_round_trip_fees = 0.0026
        self.fee_slippage_buffer = 0.003

        self.max_spread_pct = 0.02
        self.spread_median_window = 12
        self.spread_widen_mult = 1.5
        self.skip_hours_utc = [2, 3, 4, 5]
        self.max_daily_trades = 6
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.stale_order_timeout_seconds = 300
        self.live_stale_order_timeout_seconds = 900
        self.max_concurrent_open_orders = 2

        self._positions_synced = False
        self._session_blacklist = set()
        self._max_session_blacklist_size = 100
        self._first_post_warmup = True

        self.weights = {
            'relative_strength': 0.25,
            'volume_momentum': 0.20,
            'trend_strength': 0.20,
            'mean_reversion': 0.10,
            'liquidity': 0.10,
            'risk_adjusted_momentum': 0.15,
        }

        self.peak_value = None
        self.max_drawdown_limit = 0.25
        self.drawdown_cooldown = 0
        self.cooldown_hours = 24
        self.consecutive_losses = 0
        self.max_consecutive_losses = 10

        self.crypto_data = {}
        self.entry_prices = {}
        self.highest_prices = {}
        self.entry_times = {}
        self.trade_count = 0
        
        self._pending_orders = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns = {}
        self.exit_cooldown_hours = 2
        self.cancel_cooldown_minutes = 1

        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.last_hour_rebalance = None
        self.last_exit_check = None
        self.last_universe_size = 0
        self.min_volume_usd = 1000
        self.max_universe_size = 1000
        self.market_regime = "sideways"
        self.volatility_regime = "high"
        self.kraken_status = "online"
        self.btc_price = None
        self.btc_ema_short = None
        self.btc_ema_long = None
        self.market_breadth = 0.5
        self._last_skip_reason = None
        self.log_budget = 5 if self.LiveMode else 0

        # Opus enhancements: Kelly tracking deques
        self._rolling_wins = deque(maxlen=50)
        self._rolling_win_sizes = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._last_live_trade_time = None
        self._slip_abs = deque(maxlen=100)
        self.slip_outlier_threshold = 0.03

        # Load persisted state for live trading
        load_persisted_state(self)
        
        # Clean up old object store keys
        cleanup_object_store(self)

        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(self._coarse_filter)
        self.SetWarmUp(timedelta(hours=self.lookback+1))
        
        # Cancel any stale orders from previous session
        cancel_stale_orders(self)
        
        # Schedules
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=2)), self.Rebalance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.CheckExits)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 5), self.DailyReset)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=15)), self.ResyncHoldings)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=30)), self.ReviewPerformance)

    def _get_param(self, key, default):
        return self.GetParameter(key, default) if hasattr(self, 'GetParameter') else default

    def _coarse_filter(self, coarse):
        selected = [
            c for c in coarse
            if c.Market == Market.Kraken
            and c.Value > self.min_price_usd
            and c.Volume * c.Value >= self.min_volume_usd
            and c.Symbol.Value not in SYMBOL_BLACKLIST
        ]
        return [c.Symbol for c in sorted(selected, key=lambda x: x.Volume * x.Value, reverse=True)[:self.max_universe_size]]

    def OnData(self, data):
        for symbol in data.Keys:
            if symbol not in self.crypto_data:
                self.crypto_data[symbol] = {
                    'prices': deque(maxlen=self.lookback),
                    'returns': deque(maxlen=self.lookback),
                    'dollar_volume': deque(maxlen=self.lookback),
                    'volume_ma': deque(maxlen=self.lookback),
                    'rs_vs_btc': deque(maxlen=self.lookback),
                    'zscore': deque(maxlen=self.lookback),
                    'spreads': deque(maxlen=self.spread_median_window),
                    'ema_short': ExponentialMovingAverage(self.short_period),
                    'ema_medium': ExponentialMovingAverage(self.medium_period),
                    'ema_long': ExponentialMovingAverage(self.long_period),
                    'rsi': RelativeStrengthIndex(14),
                    'atr': AverageTrueRange(14),
                    'trail_stop': None,
                }
            crypto = self.crypto_data[symbol]
            if data.ContainsKey(symbol):
                bar = data[symbol]
                if bar is None:
                    continue
                crypto['prices'].append(bar.Close)
                if len(crypto['prices']) >= 2:
                    ret = (bar.Close - crypto['prices'][-2]) / crypto['prices'][-2]
                    crypto['returns'].append(ret)
                dv = bar.Volume * bar.Close
                crypto['dollar_volume'].append(dv)
                if len(crypto['dollar_volume']) >= 12:
                    crypto['volume_ma'].append(np.mean(list(crypto['dollar_volume'])[-12:]))
                if self.btc_price and self.btc_price > 0 and bar.Close > 0:
                    rs = (bar.Close / self.btc_price) - 1
                    crypto['rs_vs_btc'].append(rs)
                if len(crypto['prices']) >= self.medium_period:
                    mn = np.mean(list(crypto['prices'])[-self.medium_period:])
                    sd = np.std(list(crypto['prices'])[-self.medium_period:])
                    z = (bar.Close - mn) / sd if sd > 1e-10 else 0
                    crypto['zscore'].append(z)
                crypto['ema_short'].Update(bar.EndTime, bar.Close)
                crypto['ema_medium'].Update(bar.EndTime, bar.Close)
                crypto['ema_long'].Update(bar.EndTime, bar.Close)
                crypto['rsi'].Update(bar.EndTime, bar.Close)
                crypto['atr'].Update(bar)
                sp = get_spread_pct(self, symbol)
                if sp is not None:
                    crypto['spreads'].append(sp)

            if symbol.Value == "BTCUSD":
                bar = data.get(symbol)
                if bar:
                    self.btc_price = bar.Close
                    if self.btc_ema_short is None:
                        self.btc_ema_short = ExponentialMovingAverage(self.medium_period)
                        self.btc_ema_long = ExponentialMovingAverage(self.long_period * 2)
                    self.btc_ema_short.Update(bar.EndTime, bar.Close)
                    self.btc_ema_long.Update(bar.EndTime, bar.Close)

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        if not live_safety_checks(self):
            return
        if not self._positions_synced:
            sync_existing_positions(self)
            self._positions_synced = True
        if self._first_post_warmup:
            self._first_post_warmup = False
            if self.kraken_status == "unknown":
                self.kraken_status = "online"
        if self.Time.hour in self.skip_hours_utc:
            return
        if self.kraken_status not in ["online", "post_only"]:
            self._log_skip(f"Kraken {self.kraken_status}")
            return
        if self.Time.date() != self.last_trade_date:
            self.daily_trade_count = 0
            self.last_trade_date = self.Time.date()
        if self.daily_trade_count >= self.max_daily_trades:
            self._log_skip("Daily limit")
            return
        if self.last_hour_rebalance and (self.Time - self.last_hour_rebalance).total_seconds() < 7000:
            return
        self.last_hour_rebalance = self.Time
        open_order_count = len(self.Transactions.GetOpenOrders())
        if open_order_count >= self.max_concurrent_open_orders:
            self._log_skip(f"Open orders: {open_order_count}")
            return
        cancel_stale_new_orders(self)
        self._update_market_context()
        if self.drawdown_cooldown > 0:
            self.drawdown_cooldown -= 1
            self._log_skip(f"Cooldown {self.drawdown_cooldown}")
            return
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._log_skip("Max losses")
            return
        threshold = self._get_regime_threshold()
        candidates = []
        for symbol, crypto in self.crypto_data.items():
            if symbol.Value in self._session_blacklist:
                continue
            if not self._is_ready(crypto):
                continue
            if symbol.Value in SYMBOL_BLACKLIST:
                continue
            factors = self._calculate_factor_scores(symbol, crypto)
            if not factors:
                continue
            composite = self._calculate_composite_score(factors)
            adjusted = self._apply_fee_adjustment(composite)
            if adjusted >= threshold:
                candidates.append((symbol, adjusted, factors))
        if not candidates:
            return
        candidates.sort(key=lambda x: x[1], reverse=True)
        self._execute_trades(candidates, threshold, self.max_positions)

    def _update_market_context(self):
        if self.btc_ema_short and self.btc_ema_short.IsReady and self.btc_ema_long and self.btc_ema_long.IsReady:
            s, l = self.btc_ema_short.Current.Value, self.btc_ema_long.Current.Value
            if s > l * 1.03:
                self.market_regime = "bull"
            elif s < l * 0.97:
                self.market_regime = "bear"
            else:
                self.market_regime = "sideways"
        vols = []
        for symbol, crypto in self.crypto_data.items():
            if len(crypto['returns']) >= self.medium_period:
                vols.append(np.std(list(crypto['returns'])[-self.medium_period:]))
        if len(vols) > 5:
            med = np.median(vols)
            self.volatility_regime = "high" if med > 0.025 else "normal"
        up_count = 0
        total_count = 0
        for symbol, crypto in self.crypto_data.items():
            if self._is_ready(crypto):
                total_count += 1
                if len(crypto['returns']) >= 3:
                    if np.mean(list(crypto['returns'])[-3:]) > 0:
                        up_count += 1
        if total_count > 0:
            self.market_breadth = up_count / total_count
        else:
            self.market_breadth = 0.5

    def _get_regime_threshold(self):
        if self.market_regime == "bear":
            return self.threshold_bear
        elif self.volatility_regime == "high":
            return self.threshold_high_vol
        elif self.market_regime == "sideways":
            return self.threshold_sideways
        else:
            return self.threshold_bull

    def _is_ready(self, c):
        return len(c['prices']) >= self.medium_period and c['rsi'].IsReady

    def _calculate_factor_scores(self, symbol, crypto):
        try:
            scores = {}
            if len(crypto['rs_vs_btc']) >= 3:
                scores['relative_strength'] = self._normalize(np.mean(list(crypto['rs_vs_btc'])[-3:]), -0.05, 0.05)
            else:
                scores['relative_strength'] = 0.5
            if len(crypto['volume_ma']) >= 2 and len(crypto['returns']) >= 3:
                vol_ma_prev = crypto['volume_ma'][-2] if len(crypto['volume_ma']) >= 2 else 1
                vol_trend = (crypto['volume_ma'][-1] / (vol_ma_prev + 1e-8)) - 1
                price_trend = np.mean(list(crypto['returns'])[-3:])
                if vol_trend > 0 and price_trend > 0:
                    scores['volume_momentum'] = min(0.5 + vol_trend * 5 + price_trend * 25, 1.0)
                elif price_trend > 0:
                    scores['volume_momentum'] = 0.55
                else:
                    scores['volume_momentum'] = 0.3
            else:
                scores['volume_momentum'] = 0.5
            if crypto['ema_short'].IsReady:
                s, m, l = crypto['ema_short'].Current.Value, crypto['ema_medium'].Current.Value, crypto['ema_long'].Current.Value
                if s > m > l:
                    scores['trend_strength'] = min(0.6 + ((s - l) / l) * 10, 1.0)
                elif s > m:
                    scores['trend_strength'] = 0.6
                elif s < m < l:
                    scores['trend_strength'] = 0.2
                else:
                    scores['trend_strength'] = 0.4
            else:
                scores['trend_strength'] = 0.5
            if len(crypto['zscore']) >= 1:
                z = crypto['zscore'][-1]
                rsi = crypto['rsi'].Current.Value
                if z < -1.5 and rsi < 35:
                    scores['mean_reversion'] = 0.9
                elif z < -1.0 and rsi < 40:
                    scores['mean_reversion'] = 0.75
                elif z > 2.0 or rsi > 75:
                    scores['mean_reversion'] = 0.1
                else:
                    scores['mean_reversion'] = 0.5
            else:
                scores['mean_reversion'] = 0.5
            if len(crypto['dollar_volume']) >= 12:
                avg = np.mean(list(crypto['dollar_volume'])[-12:])
                if avg > 10000:
                    scores['liquidity'] = 0.9
                elif avg > 5000:
                    scores['liquidity'] = 0.7
                elif avg > 1000:
                    scores['liquidity'] = 0.5
                else:
                    scores['liquidity'] = 0.2
            else:
                scores['liquidity'] = 0.5
            if len(crypto['returns']) >= self.medium_period:
                std = np.std(list(crypto['returns'])[-self.medium_period:])
                if std > 1e-10:
                    sharpe = np.mean(list(crypto['returns'])[-self.medium_period:]) / std
                    scores['risk_adjusted_momentum'] = self._normalize(sharpe, -1, 1)
                else:
                    scores['risk_adjusted_momentum'] = 0.5
            else:
                scores['risk_adjusted_momentum'] = 0.5
            return scores
        except:
            return None

    def _normalize(self, v, mn, mx):
        return max(0, min(1, (v - mn) / (mx - mn)))

    def _calculate_composite_score(self, factors):
        score = sum(factors.get(f, 0.5) * w for f, w in self.weights.items())
        if self.market_regime == "bear":
            score *= 0.75
        if self.volatility_regime == "high":
            score *= 0.85
        if self.market_breadth > 0.7:
            score *= 1.05
        elif self.market_breadth < 0.3:
            score *= 0.85
        return min(score, 1.0)

    def _apply_fee_adjustment(self, score):
        return score - (self.expected_round_trip_fees * 1.1 + self.fee_slippage_buffer)

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        conviction_mult = max(0.8, min(1.2, 0.8 + (score - threshold) * 2))
        vol_floor = max(asset_vol_ann if asset_vol_ann else 0.05, 0.05)
        risk_mult = max(0.8, min(1.2, self.target_position_ann_vol / vol_floor))
        return self.position_size_pct * conviction_mult * risk_mult

    def CheckExits(self):
        if self.IsWarmingUp:
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
        
        # Exit slippage penalty for micro-caps
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
        sl, tp = self.base_stop_loss, self.base_take_profit
        if self.volatility_regime == "high":
            sl *= 1.2; tp *= 1.3
        elif self.market_regime == "bear":
            sl *= 0.8; tp *= 0.7
        atr = crypto['atr'].Current.Value if crypto and crypto['atr'].IsReady else None
        if atr and entry > 0:
            atr_sl = (atr * self.atr_sl_mult) / entry
            atr_tp = (atr * self.atr_tp_mult) / entry
            sl = max(sl, atr_sl * 0.8)
            tp = max(tp, atr_tp * 0.7)

        trailing_activation = self.trailing_activation
        trailing_stop_pct = self.trailing_stop_pct
        if self.volatility_regime == "high" or self.market_regime == "sideways":
            trailing_activation = 0.04
            trailing_stop_pct = 0.025

        tag = ""
        if pnl >= tp:
            tag = "TP"
        elif pnl <= -sl:
            tag = "SL"
        elif pnl >= trailing_activation:
            if crypto and crypto['trail_stop'] is None:
                crypto['trail_stop'] = highest
            if crypto and crypto['trail_stop']:
                if dd >= trailing_stop_pct:
                    tag = "Trail"
        if not tag and hours >= 24 and pnl < 0.01:
            tag = "TimeStop"
        if tag:
            smart_liquidate(self, symbol, tag)

    def _execute_trades(self, candidates, threshold, max_pos):
        current_count = get_actual_position_count(self)
        if current_count >= max_pos:
            return
        budget = self.Portfolio.TotalPortfolioValue * (1 - self.cash_reserve_pct)
        if budget <= 0:
            return
        slots_available = max_pos - current_count
        top = candidates[:min(slots_available, len(candidates))]
        for symbol, score, factors in top:
            if symbol in self.entry_prices:
                continue
            if symbol in self._exit_cooldowns and self.Time < self._exit_cooldowns[symbol]:
                continue
            if symbol.Value in self._session_blacklist:
                continue
            if has_open_orders(self, symbol):
                continue
            if not spread_ok(self, symbol):
                continue
            crypto = self.crypto_data.get(symbol)
            if not crypto or not self._is_ready(crypto):
                continue
            if len(crypto.get('dollar_volume', [])) >= 6:
                dv_list = list(crypto['dollar_volume'])[-6:]
                avg_dv = np.mean(dv_list)
                if self.LiveMode and avg_dv < 5000:
                    continue
            # EMA gate: require short > medium unless deep mean-reversion
            s_ema = crypto['ema_short'].Current.Value
            m_ema = crypto['ema_medium'].Current.Value
            if s_ema < m_ema:
                # Allow if strong mean-reversion
                if factors.get('mean_reversion', 0.5) < 0.75:
                    continue
            # 3-bar return check
            if len(crypto['returns']) >= 3:
                recent_ret = np.mean(list(crypto['returns'])[-3:])
                if recent_ret <= 0 and factors.get('mean_reversion', 0.5) < 0.75:
                    continue
            # Volume impact check
            price = self.Securities[symbol].Price
            target_pct = self._calculate_position_size(score, threshold, None)
            target_usd = budget * target_pct
            if price > 0:
                qty = target_usd / price
                order_value = qty * price
                if len(crypto.get('dollar_volume', [])) >= 3:
                    dv_3bar = np.mean(list(crypto['dollar_volume'])[-3:])
                    if dv_3bar > 0:
                        impact_pct = order_value / dv_3bar
                        cap = 0.05
                        if impact_pct > cap:
                            qty *= (cap / impact_pct)
                        elif impact_pct > cap * 0.6:
                            scale = 1 - (impact_pct - cap * 0.6) / (cap * 0.4) * 0.3
                            qty *= scale
                min_qty = get_min_quantity(self, symbol)
                min_notional = get_min_notional_usd(self, symbol)
                if qty < min_qty:
                    continue
                if qty * price < min_notional:
                    continue
                # High-vol/sideways cap
                if self.LiveMode and (self.volatility_regime == "high" or self.market_regime == "sideways"):
                    max_pct = 0.25
                    if qty * price / self.Portfolio.TotalPortfolioValue > max_pct:
                        qty = (self.Portfolio.TotalPortfolioValue * max_pct) / price
                qty = round_quantity(self, symbol, qty)
                if qty >= min_qty and qty * price >= min_notional:
                    try:
                        self.MarketOrder(symbol, qty, tag="Entry")
                        self._pending_orders[symbol] = self.Time
                        self.daily_trade_count += 1
                        if self.LiveMode:
                            self._last_live_trade_time = self.Time
                        break
                    except:
                        pass

    def DailyReset(self):
        if len(self._session_blacklist) > self._max_session_blacklist_size:
            self._session_blacklist = set(list(self._session_blacklist)[-self._max_session_blacklist_size:])

    def ResyncHoldings(self):
        resync_holdings(self)

    def ReviewPerformance(self):
        if self.IsWarmingUp:
            return
        current = self.Portfolio.TotalPortfolioValue
        if self.peak_value is None:
            self.peak_value = current
        elif current > self.peak_value:
            self.peak_value = current
        if self.peak_value and self.peak_value > 0:
            dd = (self.peak_value - current) / self.peak_value
            if dd >= self.max_drawdown_limit:
                self.drawdown_cooldown = self.cooldown_hours
                for kvp in self.Portfolio:
                    if is_invested_not_dust(self, kvp.Key):
                        smart_liquidate(self, kvp.Key, "DD")

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status != OrderStatus.Filled:
            return
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        if order is None:
            return
        symbol = orderEvent.Symbol
        # Log slippage with live alert
        slip_log(self, symbol, order.Direction, orderEvent.FillPrice)
        # Track entry
        if order.Direction == OrderDirection.Buy and orderEvent.FillQuantity > 0:
            self.entry_prices[symbol] = orderEvent.FillPrice
            self.highest_prices[symbol] = orderEvent.FillPrice
            self.entry_times[symbol] = self.Time
            self._pending_orders.pop(symbol, None)
        # Track exit and update Kelly stats
        if (order.Direction == OrderDirection.Sell and orderEvent.FillQuantity < 0) or (order.Direction == OrderDirection.Buy and orderEvent.FillQuantity < 0):
            if symbol in self.entry_prices:
                entry = self.entry_prices[symbol]
                pnl = (orderEvent.FillPrice - entry) / entry if entry > 0 else 0
                abs_pnl = abs(pnl)
                if pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    self.total_pnl += pnl
                    self._rolling_wins.append(1)
                    self._rolling_win_sizes.append(abs_pnl)
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
                    self.total_pnl += pnl
                    self._rolling_wins.append(0)
                    self._rolling_loss_sizes.append(abs_pnl)
                cleanup_position(self, symbol)
                self._exit_cooldowns[symbol] = self.Time + timedelta(hours=self.exit_cooldown_hours)
        # Persist state after each trade in live
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
                self.Debug("⚠️ RATE LIMIT HIT - pausing trades for 5 minutes")
                self._last_live_trade_time = self.Time
        except Exception as e:
            self.Debug(f"BrokerageMessage parse error: {e}")

    def OnEndOfAlgorithm(self):
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        self.Debug(f"Wins: {self.winning_trades} | Losses: {self.losing_trades} | WR: {wr:.2%} | Total PnL: {self.total_pnl:.2%}")
        persist_state(self)

    def _log_skip(self, reason):
        if reason != self._last_skip_reason:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason
