# region imports
from AlgorithmImports import *
from execution import *
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
        self.trailing_activation = self._get_param("trailing_activation", 0.035)
        self.trailing_stop_pct = self._get_param("trailing_stop_pct", 0.02)
        self.base_stop_loss = self._get_param("base_stop_loss", 0.04)
        self.base_take_profit = self._get_param("base_take_profit", 0.08)
        self.atr_sl_mult = self._get_param("atr_sl_mult", 1.3)
        self.atr_tp_mult = self._get_param("atr_tp_mult", 2.2)

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.35)
        self.portfolio_vol_cap = self._get_param("portfolio_vol_cap", 0.45)
        self.signal_decay_buffer = self._get_param("signal_decay_buffer", 0.03)
        self.min_signal_age_hours = self._get_param("signal_decay_min_hours", 4)
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
        self.position_size_pct = 0.55
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
        self.cooldown_hours = 12
        self.consecutive_losses = 0
        self.max_consecutive_losses = 15  # relaxed to avoid unintended lockouts

        self.crypto_data = {}
        self.entry_prices = {}
        self.highest_prices = {}
        self.entry_times = {}
        self.trade_count = 0
        
        self._pending_orders = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns = {}
        self.exit_cooldown_hours = 1
        self.cancel_cooldown_minutes = 1

        self.trailing_grace_hours = 2
        self.atr_trail_mult = 1.2

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

        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(CryptoUniverse.Kraken(self.UniverseFilter))

        try:
            btc = self.AddCrypto("BTCUSD", Resolution.Hour, Market.Kraken)
            self.btc_symbol = btc.Symbol
        except Exception as e:
            self.Debug(f"Warning: Could not add BTC - {e}")

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
        return order_time.replace(tzinfo=None) if order_time.tzinfo is not None else order_time
    
    def _record_exit_pnl(self, symbol, entry_price, exit_price):
        """Helper to record PnL from an exit trade."""
        if entry_price <= 0 or exit_price <= 0:
            return 0
        
        pnl = (exit_price - entry_price) / entry_price
        self._rolling_wins.append(1 if pnl > 0 else 0)
        if pnl > 0:
            self._rolling_win_sizes.append(pnl)
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self._rolling_loss_sizes.append(abs(pnl))
            self.losing_trades += 1
            self.consecutive_losses += 1
        self.total_pnl += pnl
        return pnl

    def EmitInsights(self, *insights):
        return []

    def EmitInsight(self, insight):
        return []

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date = self.Time.date()
        if len(self._session_blacklist) > 0:
            self.Debug(f"Clearing session blacklist ({len(self._session_blacklist)} items)")
            self._session_blacklist.clear()
        persist_state(self)

    def HealthCheck(self):
        if self.IsWarmingUp: return
        # Live resync to catch any holdings the event stream might have missed
        self.ResyncHoldings()
        issues = []
        if self.Portfolio.Cash < 5:
            issues.append(f"Low cash: ${self.Portfolio.Cash:.2f}")
        for symbol in list(self.entry_prices.keys()):
            open_orders = self.Transactions.GetOpenOrders(symbol)
            if len(open_orders) > 0:
                # Check if all orders are stuck (older than timeout)
                all_stale = True
                for o in open_orders:
                    order_time = self._normalize_order_time(o.Time)
                    if (self.Time - order_time).total_seconds() <= self.live_stale_order_timeout_seconds:
                        all_stale = False
                        break
                if not all_stale:
                    continue  # Has non-stale orders, skip this symbol
                
                # All orders are stale - cancel them
                for o in open_orders:
                    try:
                        self.Transactions.CancelOrder(o.Id)
                        issues.append(f"Canceled stale order: {symbol.Value} (order {o.Id})")
                    except Exception as e:
                        self.Debug(f"Error canceling stale order for {symbol.Value}: {e}")
            
            if not is_invested_not_dust(self, symbol):
                issues.append(f"Orphan tracking: {symbol.Value}")
                cleanup_position(self, symbol)
        for kvp in self.Portfolio:
            if is_invested_not_dust(self, kvp.Key) and kvp.Key not in self.entry_prices:
                issues.append(f"Untracked position: {kvp.Key.Value}")
        if len(self._session_blacklist) > 50:
            issues.append(f"Large session blacklist: {len(self._session_blacklist)}")
        open_orders = self.Transactions.GetOpenOrders()
        if len(open_orders) > 0:
            issues.append(f"Open orders: {len(open_orders)}")
        if issues:
            self.Debug("=== HEALTH CHECK ===")
            for issue in issues:
                self.Debug(f"  ⚠️ {issue}")
        else:
            debug_limited(self, "Health check: OK")

    def ResyncHoldings(self):
        """
        Live-only safety: backfills tracking for any holdings that exist in the brokerage
        but were not registered via OnOrderEvent (e.g., missed fill events).
        """
        if self.IsWarmingUp: return
        if not self.LiveMode: return
        
        # Only log periodically to avoid spam (every 30 minutes)
        if not hasattr(self, '_last_resync_log') or (self.Time - self._last_resync_log).total_seconds() > self.resync_log_interval_seconds:
            self.Debug(f"RESYNC CHECK: Portfolio keys={len(list(self.Portfolio.Keys))}, tracked={len(self.entry_prices)}")
            self._last_resync_log = self.Time
        
        missing = []
        for symbol in self.Portfolio.Keys:
            holding = self.Portfolio[symbol]
            if not holding.Invested or holding.Quantity == 0:
                continue
            if symbol in self.entry_prices:
                continue
            # Skip symbols being tracked by VerifyOrderFills to avoid conflicts
            if symbol in self._submitted_orders:
                continue
            if symbol in self._exit_cooldowns and self.Time < self._exit_cooldowns[symbol]:
                continue
            # Check if there are non-stale open orders
            # If all open orders are stale, we should resync anyway
            if has_non_stale_open_orders(self, symbol):
                continue
            missing.append(symbol)
        if missing:
            self.Debug(f"RESYNC: detected {len(missing)} holdings without tracking; backfilling.")
            for symbol in missing:
                try:
                    if symbol not in self.Securities:
                        self.AddCrypto(symbol.Value, Resolution.Hour, Market.Kraken)
                    holding = self.Portfolio[symbol]
                    entry = holding.AveragePrice
                    self.entry_prices[symbol] = entry
                    self.highest_prices[symbol] = entry
                    self.entry_times[symbol] = self.Time
                    current_price = self.Securities[symbol].Price if symbol in self.Securities else holding.Price
                    pnl_pct = (current_price - entry) / entry if entry > 0 else 0
                    self.Debug(f"RESYNCED: {symbol.Value} | Qty: {holding.Quantity} | Entry: ${entry:.4f} | Now: ${current_price:.4f} | PnL: {pnl_pct:+.2%}")
                except Exception as e:
                    self.Debug(f"Resync error {symbol.Value}: {e}")
        
        # Reverse resync: detect positions we're tracking but broker no longer holds
        phantoms = []
        for symbol in list(self.entry_prices.keys()):
            # Skip if order still pending
            if symbol in self._submitted_orders:
                continue
            # Skip if in exit cooldown
            if symbol in self._exit_cooldowns and self.Time < self._exit_cooldowns[symbol]:
                continue
            
            holding = self.Portfolio[symbol] if symbol in self.Portfolio else None
            if holding is None or not holding.Invested or holding.Quantity == 0:
                # We're tracking this position but broker says it's gone
                # Check for open orders first
                open_orders = self.Transactions.GetOpenOrders(symbol)
                if len(open_orders) > 0:
                    # Has open orders - check if all are stuck
                    all_stuck = True
                    for o in open_orders:
                        order_time = self._normalize_order_time(o.Time)
                        if (self.Time - order_time).total_seconds() <= self.live_stale_order_timeout_seconds:
                            all_stuck = False
                            break
                    
                    if not all_stuck:
                        continue  # Has non-stuck orders, wait
                    
                    # All orders are stuck - cancel them
                    for o in open_orders:
                        try:
                            self.Transactions.CancelOrder(o.Id)
                            self.Debug(f"PHANTOM: Canceling stuck order {o.Id} for {symbol.Value}")
                        except Exception as e:
                            self.Debug(f"Error canceling stuck order for {symbol.Value}: {e}")
                
                phantoms.append(symbol)
        
        if phantoms:
            self.Debug(f"⚠️ REVERSE RESYNC: detected {len(phantoms)} phantom positions")
            for symbol in phantoms:
                self.Debug(f"⚠️ PHANTOM POSITION DETECTED: {symbol.Value} — tracked but broker qty=0, cleaning up")
                cleanup_position(self, symbol)

    def VerifyOrderFills(self):
        """Verify submitted orders filled/timed out. Retry once before blacklisting."""
        if self.IsWarmingUp:
            return
        
        current_time = self.Time
        symbols_to_remove = []
        
        # Process pending retries first
        for symbol in list(self._retry_pending.keys()):
            cancel_time = self._retry_pending[symbol]
            if (current_time - cancel_time).total_seconds() >= self.retry_pending_cooldown_seconds:
                # Check if position exists now (cancel failed or order filled after cancel)
                if symbol in self.Portfolio and self.Portfolio[symbol].Invested:
                    # Position exists, don't retry - just track it
                    holding = self.Portfolio[symbol]
                    if symbol not in self.entry_prices:
                        self.entry_prices[symbol] = holding.AveragePrice
                        self.highest_prices[symbol] = holding.AveragePrice
                        self.entry_times[symbol] = current_time
                        self.Debug(f"RETRY SKIPPED (position exists): {symbol.Value}")
                    del self._retry_pending[symbol]
                else:
                    # Safe to retry now
                    self.Debug(f"RETRY NOW: {symbol.Value} - cancel confirmed")
                    del self._retry_pending[symbol]
        
        for symbol, order_info in list(self._submitted_orders.items()):
            order_age_seconds = (current_time - order_info['time']).total_seconds()
            order_id = order_info['order_id']
            
            # Determine timeout based on order type
            if order_info.get('is_limit_entry', False):
                timeout = order_info.get('timeout_seconds', 60)
            elif order_info.get('is_limit_exit', False):
                timeout = 90
            else:
                timeout = self.order_timeout_seconds
            
            # Check for missed fills (order filled but event missed)
            if order_age_seconds > self.order_fill_check_threshold_seconds:
                intent = order_info.get('intent', 'entry')
                
                # Check order status
                try:
                    order = self.Transactions.GetOrderById(order_id)
                    if order is not None and order.Status == OrderStatus.Filled:
                        # Order filled but we missed the event
                        if intent == 'exit':
                            # Exit order filled - position should be gone
                            entry = self.entry_prices.get(symbol, None)
                            if entry:
                                # Calculate PnL using last known price
                                current_price = self.Securities[symbol].Price if symbol in self.Securities else 0
                                if current_price > 0:
                                    pnl = self._record_exit_pnl(symbol, entry, current_price)
                                    self.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} | PnL: {pnl:+.2%} | Entry: ${entry:.4f} | Exit: ${current_price:.4f}")
                                cleanup_position(self, symbol)
                            symbols_to_remove.append(symbol)
                            self._order_retries.pop(order_id, None)
                            continue
                        else:
                            # Entry order filled
                            if symbol in self.Portfolio and self.Portfolio[symbol].Invested:
                                holding = self.Portfolio[symbol]
                                entry_price = holding.AveragePrice
                                current_price = self.Securities[symbol].Price if symbol in self.Securities else holding.Price
                                
                                # Only set tracking if not already tracked (avoid double increment)
                                if symbol not in self.entry_prices:
                                    self.entry_prices[symbol] = entry_price
                                    self.highest_prices[symbol] = max(current_price, entry_price)
                                    self.entry_times[symbol] = order_info['time']
                                    self.daily_trade_count += 1
                                    self.Debug(f"FILL VERIFIED (missed event): {symbol.Value} | Entry: ${entry_price:.4f} | Qty: {holding.Quantity}")
                                
                                symbols_to_remove.append(symbol)
                                self._order_retries.pop(order_id, None)
                                continue
                except Exception as e:
                    self.Debug(f"Error checking order status for {symbol.Value}: {e}")
                
                # Check for exit orders where position is now gone (fill event missed)
                if intent == 'exit':
                    holding = self.Portfolio[symbol] if symbol in self.Portfolio else None
                    if holding is None or not holding.Invested or holding.Quantity == 0:
                        # Position is gone - the exit filled but we missed the event
                        entry = self.entry_prices.get(symbol, None)
                        if entry:
                            # Calculate PnL using last known price
                            current_price = self.Securities[symbol].Price if symbol in self.Securities else 0
                            if current_price > 0:
                                pnl = self._record_exit_pnl(symbol, entry, current_price)
                                self.Debug(f"⚠️ MISSED EXIT FILL (position gone): {symbol.Value} | PnL: {pnl:+.2%} | Entry: ${entry:.4f} | Estimated Exit: ${current_price:.4f}")
                            cleanup_position(self, symbol)
                        symbols_to_remove.append(symbol)
                        self._order_retries.pop(order_id, None)
                        continue
            
            # Handle timeout with retry logic
            if order_age_seconds > timeout:
                retry_count = self._order_retries.get(order_id, 0)
                
                if retry_count == 0:
                    try:
                        self.Transactions.CancelOrder(order_id)
                        self.Debug(f"ORDER TIMEOUT (attempt 1): {symbol.Value} - cancel requested, will retry after cooldown")
                        
                        # Set cooldown flag - do NOT retry immediately
                        self._retry_pending[symbol] = current_time
                        self._order_retries[order_id] = 1
                        symbols_to_remove.append(symbol)
                    except Exception as e:
                        self.Debug(f"Error canceling order for {symbol.Value}: {e}")
                        symbols_to_remove.append(symbol)
                        self._order_retries.pop(order_id, None)
                else:
                    try:
                        self.Transactions.CancelOrder(order_id)
                        self._session_blacklist.add(symbol.Value)
                        symbols_to_remove.append(symbol)
                        self._order_retries.pop(order_id, None)
                        self.Debug(f"ORDER TIMEOUT (attempt 2): {symbol.Value} - blacklisted")
                    except Exception as e:
                        self.Debug(f"Error canceling order {order_id} on second timeout: {e}")
                        symbols_to_remove.append(symbol)
                        self._order_retries.pop(order_id, None)
        
        # Remove processed orders
        for symbol in symbols_to_remove:
            self._submitted_orders.pop(symbol, None)

    def PortfolioSanityCheck(self):
        """
        Check for portfolio value mismatches between QC and tracked positions.
        Fixed: Use CashBook["USD"].Amount for actual USD cash, not Portfolio.Cash
        (which in Cash account mode includes crypto holdings value).
        """
        if self.IsWarmingUp:
            return
        
        total_qc = self.Portfolio.TotalPortfolioValue
        
        # Use actual USD cash, NOT Portfolio.Cash (which double-counts in Cash accounts)
        try:
            usd_cash = self.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            usd_cash = self.Portfolio.Cash
        
        tracked_value = 0.0
        
        for sym in list(self.entry_prices.keys()):
            if sym in self.Securities:
                price = self.Securities[sym].Price
                if sym in self.Portfolio:
                    tracked_value += abs(self.Portfolio[sym].Quantity) * price
        
        expected = usd_cash + tracked_value
        
        # Use both percentage and minimum absolute threshold to avoid spam on small portfolios
        # Also add 1-hour cooldown between warnings
        abs_diff = abs(total_qc - expected)
        if total_qc > 1.0:
            pct_diff = abs_diff / total_qc
            should_warn = pct_diff > self.portfolio_mismatch_threshold and abs_diff > self.portfolio_mismatch_min_dollars
            if should_warn:
                if self._last_mismatch_warning is None or (self.Time - self._last_mismatch_warning).total_seconds() >= self.portfolio_mismatch_cooldown_seconds:
                    self.Debug(f"⚠️ PORTFOLIO MISMATCH: QC total=${total_qc:.2f} but usd_cash+tracked=${expected:.2f} (diff=${abs_diff:.2f})")
                    self._last_mismatch_warning = self.Time

    def ReviewPerformance(self):
        if self.IsWarmingUp or len(self.trade_log) < 10: return
        recent_trades = self.trade_log[-15:] if len(self.trade_log) >= 15 else self.trade_log
        if len(recent_trades) == 0: return
        recent_win_rate = sum(1 for t in recent_trades if t['pnl_pct'] > 0) / len(recent_trades)
        recent_avg_pnl = np.mean([t['pnl_pct'] for t in recent_trades])
        old_max = self.max_positions
        if recent_win_rate < 0.2 and recent_avg_pnl < -0.05:
            self.max_positions = 1
            if old_max != 1:
                self.Debug(f"PERFORMANCE DECAY: max_pos=1 (WR:{recent_win_rate:.0%}, PnL:{recent_avg_pnl:+.2%})")
        elif recent_win_rate > 0.35 and recent_avg_pnl > -0.01:
            self.max_positions = self.base_max_positions
            if old_max != self.base_max_positions:
                self.Debug(f"PERFORMANCE RECOVERY: max_pos={self.base_max_positions}")

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
        if len(self.btc_prices) >= self.medium_period:
            btc_arr = np.array(list(self.btc_prices))
            btc_sma = np.mean(btc_arr[-self.medium_period:])
            current_btc = btc_arr[-1]
            if current_btc > btc_sma * 1.03:
                self.market_regime = "bull"
            elif current_btc < btc_sma * 0.97:
                self.market_regime = "bear"
            else:
                self.market_regime = "sideways"
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
                if l > 0:
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
        except Exception as e:
            return None

    def _calculate_composite_score(self, factors):
        score = sum(factors.get(f, 0.5) * w for f, w in self.weights.items())
        
        # Momentum acceleration bonus
        strong_factors = sum(1 for v in factors.values() if v >= 0.7)
        if strong_factors >= 4:
            score *= 1.10
        elif strong_factors >= 3:
            score *= 1.05
        
        if self.market_regime == "bear":
            score *= 0.85
        if self.volatility_regime == "high":
            score *= 0.90
        if self.market_breadth > 0.7:
            score *= 1.08
        elif self.market_breadth < 0.3:
            score *= 0.90
        return min(score, 1.0)

    def _apply_fee_adjustment(self, score):
        return score - (self.expected_round_trip_fees * 1.1 + self.fee_slippage_buffer)

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        conviction_mult = max(0.8, min(1.2, 0.8 + (score - threshold) * 2))
        vol_floor = max(asset_vol_ann or 0.05, 0.05)
        risk_mult = max(0.8, min(1.2, self.target_position_ann_vol / vol_floor))
        return self.position_size_pct * conviction_mult * risk_mult

    def _kelly_fraction(self):
        return kelly_fraction(self)

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
        if self.daily_trade_count >= self.max_daily_trades:
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
            
            composite_score = self._calculate_composite_score(factor_scores)
            net_score = self._apply_fee_adjustment(composite_score)
            
            # Populate recent_net_scores for persistence filter (Fix 3)
            crypto['recent_net_scores'].append(net_score)
            
            if net_score > threshold_now:
                count_above_thresh += 1
                scores.append({
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'net_score': net_score,
                    'factors': factor_scores,
                    'volatility': crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                    'dollar_volume': list(crypto['dollar_volume'])[-6:] if len(crypto['dollar_volume']) >= 6 else [],
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
        total_reserved = 0
        for o in self.Transactions.GetOpenOrders():
            if o.Direction == OrderDirection.Buy:
                if o.Price > 0:
                    order_price = o.Price
                elif o.Symbol in self.Securities:
                    order_price = self.Securities[o.Symbol].Price
                    if order_price <= 0:
                        continue
                else:
                    continue
                total_reserved += abs(o.Quantity) * order_price
        return total_reserved

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
            if self.daily_trade_count >= self.max_daily_trades:
                break
            if get_actual_position_count(self) >= dynamic_max_pos:
                break
            sym = cand['symbol']
            comp_score = cand.get('composite_score', 0.5)
            net_score = cand.get('net_score', 0.5)
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

            total_value = self.Portfolio.TotalPortfolioValue
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
            if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
                ema_short = crypto['ema_short'].Current.Value
                ema_medium = crypto['ema_medium'].Current.Value

                is_mean_reversion = False
                if len(crypto['zscore']) >= 1 and crypto['rsi'].IsReady:
                    z = crypto['zscore'][-1]
                    rsi = crypto['rsi'].Current.Value
                    if z < -1.5 and rsi < 35:
                        is_mean_reversion = True

                if not is_mean_reversion:
                    if ema_short <= ema_medium:
                        reject_ema_gate += 1
                        continue
                    if len(crypto['returns']) >= 3:
                        recent_return = np.mean(list(crypto['returns'])[-3:])
                        if recent_return <= 0:
                            reject_ema_gate += 1
                            continue
            
            if len(crypto['recent_net_scores']) >= 2:
                above_threshold_count = sum(1 for score in list(crypto['recent_net_scores'])[-2:] if score > threshold_now)
                if above_threshold_count == 0:
                    reject_recent_net_scores += 1
                    continue
            
            vol = self._annualized_vol(crypto)
            size = self._calculate_position_size(net_score, threshold_now, vol)
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
                    if self.LiveMode:
                        self._last_live_trade_time = self.Time
            except Exception as e:
                self.Debug(f"ORDER FAILED: {sym.Value} - {e}")
                self._session_blacklist.add(sym.Value)
                continue
            if self.LiveMode and success_count >= 2:
                break
        
        if success_count > 0 or reject_ema_gate + reject_recent_net_scores + reject_impact_ratio > 5:
            debug_limited(self, f"EXECUTE: {success_count}/{len(candidates)} | rejects: ema={reject_ema_gate} scores={reject_recent_net_scores} impact={reject_impact_ratio}")

    def _get_threshold(self):
        if self.market_regime == "bull" and self.market_breadth > 0.6:
            return self.threshold_bull
        elif self.market_regime == "bear":
            return self.threshold_bear
        elif self.volatility_regime == "high":
            return self.threshold_high_vol
        return self.threshold_sideways

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
            trailing_activation = 0.03
            trailing_stop_pct = 0.02

        tag = ""
        min_notional_usd = get_min_notional_usd(self, symbol)
        trailing_allowed = hours >= self.trailing_grace_hours
        if pnl <= -sl:
            tag = "Stop Loss"
        elif pnl >= tp:
            tag = "Take Profit"
        elif trailing_allowed and pnl > trailing_activation and dd >= trailing_stop_pct:
            tag = "Trailing Stop"
        elif self.market_regime == "bear" and pnl > 0.03:
            tag = "Bear Exit"
        elif hours > 18 and pnl < 0.01:
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
                    factors = self._calculate_factor_scores(symbol, crypto)
                    if factors:
                        comp = self._calculate_composite_score(factors)
                        net = self._apply_fee_adjustment(comp)
                        if net < self._get_threshold() - self.signal_decay_buffer:
                            tag = "Signal Decay"
            except Exception as e:
                pass
        if tag:
            if price * abs(holding.Quantity) < min_notional_usd * 0.9:
                return
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
        if self.IsWarmingUp:
            return
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        avg = self.total_pnl / total if total > 0 else 0
        self.Debug(f"=== DAILY {self.Time.date()} ===")
        self.Debug(f"Portfolio: ${self.Portfolio.TotalPortfolioValue:.2f} | Cash: ${self.Portfolio.Cash:.2f}")
        self.Debug(f"Pos: {get_actual_position_count(self)}/{self.base_max_positions} | {self.market_regime} {self.volatility_regime} {self.market_breadth:.0%}")
        self.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
        if self._session_blacklist:
            self.Debug(f"Blacklist: {len(self._session_blacklist)}")
        for kvp in self.Portfolio:
            if is_invested_not_dust(self, kvp.Key):
                s = kvp.Key
                entry = self.entry_prices.get(s, kvp.Value.AveragePrice)
                cur = self.Securities[s].Price if s in self.Securities else kvp.Value.Price
                pnl = (cur - entry) / entry if entry > 0 else 0
                self.Debug(f"  {s.Value}: ${entry:.4f}→${cur:.4f} ({pnl:+.2%})")
        persist_state(self)
