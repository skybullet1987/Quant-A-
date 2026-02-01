# region imports
from AlgorithmImports import *

import numpy as np
from collections import deque
from datetime import timedelta
# endregion


class SimplifiedCryptoStrategy(QCAlgorithm):
    """
    Simplified 5-Factor Strategy - LIVE READY v2.5 (Fixed & Stable)
    
    Features:
    - Safety hardened against stuck/zombie orders.
    - Fixed Timezone, KeyError, and Missing Attribute crashes.
    - Strict risk management & dust handling.
    """

    SYMBOL_BLACKLIST = {
        # Major / Stablecoins
        "BTCUSD", "ETHUSD", "USDTUSD", "USDCUSD", "PYUSDUSD", "EURCUSD", "USTUSD",
        # Privacy coins
        "SHIBUSD", "XMRUSD", "ZECUSD", "DASHUSD",
        # Other exclusions
        "XNYUSD",
        # Geo-restricted (Canada/Ontario)
        "BDXNUSD", "RAIINUSD", "LUNAUSD", "LUNCUSD", "USTCUSD", "ABORDUSD",
        "BONDUSD", "KEEPUSD", "ORNUSD",
        # Problematic symbols (orders not filling)
        "MUSD", "ICNTUSD",
    }

    # Fallback minimum order quantities
    KRAKEN_MIN_QTY_FALLBACK = {
        'AXSUSD': 5.0, 'SANDUSD': 10.0, 'MANAUSD': 10.0, 'ADAUSD': 10.0,
        'MATICUSD': 10.0, 'DOTUSD': 1.0, 'LINKUSD': 0.5, 'AVAXUSD': 0.2,
        'ATOMUSD': 0.5, 'NEARUSD': 1.0, 'SOLUSD': 0.05, 'XRPUSD': 10.0,
        'ALGOUSD': 10.0, 'XLMUSD': 30.0, 'TRXUSD': 50.0, 'ENJUSD': 10.0,
        'BATUSD': 10.0, 'CRVUSD': 5.0, 'SNXUSD': 3.0, 'COMPUSD': 0.1,
        'AAVEUSD': 0.05, 'MKRUSD': 0.01, 'YFIUSD': 0.001, 'UNIUSD': 1.0,
        'SUSHIUSD': 5.0, '1INCHUSD': 5.0, 'GRTUSD': 10.0, 'FTMUSD': 10.0,
        'IMXUSD': 5.0, 'APEUSD': 2.0, 'GMTUSD': 10.0, 'OPUSD': 5.0,
        'LDOUSD': 5.0, 'ARBUSD': 5.0, 'LPTUSD': 5.0, 'KTAUSD': 10.0,
        'GUNUSD': 50.0, 'BANANAS31USD': 500.0, 'CHILLHOUSEUSD': 500.0,
        'PHAUSD': 50.0, 'MUSD': 50.0, 'ICNTUSD': 50.0,
    }

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(20)
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        # === PARAMETERIZED VALUES ===
        self.threshold_bull = self._get_param("threshold_bull", 0.50)
        self.threshold_bear = self._get_param("threshold_bear", 0.62)
        self.threshold_sideways = self._get_param("threshold_sideways", 0.55)
        self.threshold_high_vol = self._get_param("threshold_high_vol", 0.58)
        self.trailing_activation = self._get_param("trailing_activation", 0.05)
        self.trailing_stop_pct = self._get_param("trailing_stop_pct", 0.03)
        self.base_stop_loss = self._get_param("base_stop_loss", 0.05)
        self.base_take_profit = self._get_param("base_take_profit", 0.12)
        self.correlation_threshold = self._get_param("correlation_threshold", 0.85)
        self.min_conviction_multiplier = self._get_param("min_conviction_mult", 0.8)
        self.max_conviction_multiplier = self._get_param("max_conviction_mult", 1.2)

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.25)
        self.portfolio_vol_cap = self._get_param("portfolio_vol_cap", 0.35)
        self.signal_decay_buffer = self._get_param("signal_decay_buffer", 0.05)
        self.min_signal_age_hours = self._get_param("signal_decay_min_hours", 6)
        self.min_edge_buffer = self._get_param("min_edge_buffer", 0.003)
        self.cost_impact_factor = self._get_param("cost_impact_factor", 0.10)
        self.atr_sl_mult = self._get_param("atr_sl_mult", 1.6)
        self.atr_tp_mult = self._get_param("atr_tp_mult", 3.0)

        # === TIMEFRAMES ===
        self.short_period = 6
        self.medium_period = 24
        self.long_period = 72
        self.lookback = 72
        self.sqrt_annualization = np.sqrt(24 * 365)
        self.min_asset_vol_floor = 0.05

        # === POSITION MANAGEMENT ===
        self.base_max_positions = 2
        self.max_positions = self.base_max_positions
        self.position_size_pct = 0.45
        self.min_notional = 5.0

        # === TRADING COSTS ===
        self.expected_round_trip_fees = 0.0026
        self.expected_holding_return = 0.08

        # === LIVE TRADING SAFEGUARDS ===
        self.max_spread_pct = 0.02
        self.min_hourly_volume = 500
        self.skip_hours_utc = [2, 3, 4, 5]
        self.max_daily_trades = 6
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.stale_order_timeout_seconds = 300

        # === FLAGS AND BLACKLISTS ===
        self._positions_synced = False
        self._session_blacklist = set()
        self._max_session_blacklist_size = 100

        # === FACTOR WEIGHTS ===
        self.weights = {
            'relative_strength': 0.25,
            'volume_momentum': 0.20,
            'trend_strength': 0.20,
            'mean_reversion': 0.10,
            'liquidity': 0.10,
            'risk_adjusted_momentum': 0.15,
        }

        # === DRAWDOWN PROTECTION ===
        self.peak_value = None
        self.max_drawdown_limit = 0.25
        self.drawdown_cooldown = 0
        self.cooldown_hours = 24
        self.consecutive_losses = 0
        self.max_consecutive_losses = 4

        # === DATA STRUCTURES ===
        self.crypto_data = {}
        self.entry_prices = {}
        self.highest_prices = {}
        self.entry_times = {}
        self.trade_count = 0

        # === BTC REFERENCE ===
        self.btc_symbol = None
        self.btc_returns = deque(maxlen=self.long_period)
        self.btc_prices = deque(maxlen=self.long_period)
        self.btc_volatility = deque(maxlen=self.long_period)

        # === MARKET CONTEXT ===
        self.market_regime = "unknown"
        self.volatility_regime = "normal"
        self.market_breadth = 0.5

        # === PERFORMANCE TRACKING ===
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.trade_log = []
        self.log_budget = 0
        self.last_log_time = None

        # === UNIVERSE ===
        self.min_volume_usd = 5000
        self.max_universe_size = 500

        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(CryptoUniverse.Kraken(self.UniverseFilter))

        try:
            btc = self.AddCrypto("BTCUSD", Resolution.Hour, Market.Kraken)
            self.btc_symbol = btc.Symbol
        except:
            self.Debug("Warning: Could not add BTC")

        # === SCHEDULES ===
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=2)), self.Rebalance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.CheckExits)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 1), self.DailyReport)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 0), self.ResetDailyCounters)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=6)), self.ReviewPerformance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(12, 0), self.HealthCheck)

        self.SetWarmUp(timedelta(days=5))
        self.SetSecurityInitializer(lambda security: security.SetSlippageModel(VolumeShareSlippageModel(0.15, 0.02)))
        self.Settings.FreePortfolioValuePercentage = 0.05
        
        # Suppress insights to prevent API Error messages in Live
        self.Settings.InsightScore = False

        if self.LiveMode:
            self.Debug("=" * 50)
            self.Debug("=== LIVE TRADING ACTIVATED v2.5 (SAFE) ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f}")
            self.Debug(f"Max positions: {self.max_positions}")
            self.Debug(f"Position size: {self.position_size_pct:.0%}")
            self.Debug("=" * 50)

    def _get_param(self, name, default):
        try:
            param = self.GetParameter(name)
            if param is not None and param != "":
                return float(param)
            return default
        except:
            return default

    # === DUST CHECK HELPER ===
    def _is_invested_not_dust(self, symbol):
        """Returns True if we have a position larger than the minimum order quantity."""
        if symbol not in self.Portfolio: return False
        holding = self.Portfolio[symbol]
        if not holding.Invested or holding.Quantity == 0: return False
        min_qty = self._get_min_quantity(symbol)
        return abs(holding.Quantity) >= min_qty

    def _get_actual_position_count(self):
        """Get actual position count, ignoring dust."""
        count = 0
        for kvp in self.Portfolio:
            if self._is_invested_not_dust(kvp.Key):
                count += 1
        return count

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date = self.Time.date()
        if len(self._session_blacklist) > 0:
            self.Debug(f"Clearing session blacklist ({len(self._session_blacklist)} items)")
            self._session_blacklist.clear()

    def HealthCheck(self):
        """Periodic health check (Patched for pending orders)."""
        if self.IsWarmingUp: return

        issues = []
        if self.Portfolio.Cash < 5:
            issues.append(f"Low cash: ${self.Portfolio.Cash:.2f}")

        # Check for orphan tracking (tracking but not holding)
        for symbol in list(self.entry_prices.keys()):
            # FIX: Skip check if pending orders exist (Wait for fill)
            if len(self.Transactions.GetOpenOrders(symbol)) > 0:
                continue

            if not self._is_invested_not_dust(symbol):
                issues.append(f"Orphan tracking: {symbol.Value}")
                self._cleanup_position(symbol)

        # Check for untracked positions
        for kvp in self.Portfolio:
            if self._is_invested_not_dust(kvp.Key):
                if kvp.Key not in self.entry_prices:
                    issues.append(f"Untracked position: {kvp.Key.Value}")

        # Check session blacklist
        if len(self._session_blacklist) > 50:
            issues.append(f"Large session blacklist: {len(self._session_blacklist)}")

        # Check open orders
        open_orders = self.Transactions.GetOpenOrders()
        if len(open_orders) > 0:
            issues.append(f"Open orders: {len(open_orders)}")

        if issues:
            self.Debug("=== HEALTH CHECK ===")
            for issue in issues:
                self.Debug(f"  ⚠️ {issue}")
        else:
            self._debug_limited("Health check: OK")

    def ReviewPerformance(self):
        if self.IsWarmingUp or len(self.trade_log) < 5: return
        recent_trades = self.trade_log[-10:] if len(self.trade_log) >= 10 else self.trade_log
        if len(recent_trades) == 0: return

        recent_win_rate = sum(1 for t in recent_trades if t['pnl_pct'] > 0) / len(recent_trades)
        recent_avg_pnl = np.mean([t['pnl_pct'] for t in recent_trades])
        old_max = self.max_positions

        if recent_win_rate < 0.3 or recent_avg_pnl < -0.03:
            self.max_positions = 1
            if old_max != 1:
                self.Debug(f"PERFORMANCE DECAY: max_pos=1 (WR:{recent_win_rate:.0%}, PnL:{recent_avg_pnl:+.2%})")
        elif recent_win_rate > 0.5 and recent_avg_pnl > 0:
            self.max_positions = self.base_max_positions
            if old_max != self.base_max_positions:
                self.Debug(f"PERFORMANCE RECOVERY: max_pos={self.base_max_positions}")

    def _get_min_quantity(self, symbol):
        ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
        try:
            if symbol in self.Securities:
                sec = self.Securities[symbol]
                if hasattr(sec, 'SymbolProperties') and sec.SymbolProperties is not None:
                    min_size = sec.SymbolProperties.MinimumOrderSize
                    if min_size is not None and min_size > 0:
                        return float(min_size)
        except: pass

        if ticker in self.KRAKEN_MIN_QTY_FALLBACK:
            return self.KRAKEN_MIN_QTY_FALLBACK[ticker]
        return self._estimate_min_qty(symbol)

    def _estimate_min_qty(self, symbol):
        try:
            price = self.Securities[symbol].Price if symbol in self.Securities else 0
        except: price = 0

        if price <= 0: return 50.0
        if price < 0.001: return 1000.0
        elif price < 0.01: return 500.0
        elif price < 0.1: return 50.0
        elif price < 1.0: return 10.0
        elif price < 10.0: return 5.0
        elif price < 100.0: return 1.0
        elif price < 1000.0: return 0.1
        else: return 0.01

    def _round_quantity(self, symbol, quantity):
        try:
            lot_size = self.Securities[symbol].SymbolProperties.LotSize
            if lot_size is not None and lot_size > 0:
                return float(int(quantity / lot_size) * lot_size)
            return quantity
        except: return quantity

    def _smart_liquidate(self, symbol, tag="Liquidate"):
        if symbol not in self.Portfolio or self.Portfolio[symbol].Quantity == 0: return
        self.Transactions.CancelOpenOrders(symbol)
        holding_qty = self.Portfolio[symbol].Quantity
        min_qty = self._get_min_quantity(symbol)

        if abs(holding_qty) < min_qty: return
        safe_qty = self._round_quantity(symbol, abs(holding_qty))
        if safe_qty < min_qty: return

        if safe_qty > 0:
            direction_mult = -1 if holding_qty > 0 else 1
            self.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
        else:
            self.Debug(f"Warning: {symbol.Value} holding {holding_qty} rounds to 0")

    def _cancel_stale_orders(self):
        try:
            open_orders = self.Transactions.GetOpenOrders()
            if len(open_orders) > 0:
                self.Debug(f"Found {len(open_orders)} open orders - canceling all...")
                for order in open_orders:
                    self.Transactions.CancelOrder(order.Id)
        except Exception as e:
            self.Debug(f"Error canceling stale orders: {e}")

    def _cancel_stale_new_orders(self):
        """Cancel orders pending too long (FIXED TIMEZONE BUG + ZOMBIE DETECTION)."""
        try:
            open_orders = self.Transactions.GetOpenOrders()
            for order in open_orders:
                # FIX: Handle timezone mismatch
                order_time = order.Time
                if order_time.tzinfo is not None:
                    order_time = order_time.replace(tzinfo=None)
                
                order_age = (self.Time - order_time).total_seconds()
                
                if order_age > self.stale_order_timeout_seconds:
                    ticker = order.Symbol.Value
                    self.Debug(f"Canceling stale: {ticker} (age: {order_age/60:.1f}m)")
                    
                    self.Transactions.CancelOrder(order.Id)
                    
                    # Aggressive blacklist for stuck orders to prevent loop
                    if len(self._session_blacklist) < self._max_session_blacklist_size:
                        self._session_blacklist.add(ticker)
                        
                    self.Debug(f"⚠️ ZOMBIE ORDER DETECTED: {ticker} - Check Brokerage!")
        except Exception as e:
            self.Debug(f"Error in _cancel_stale_new_orders: {e}")

    def _calculate_correlation(self, symbol1, symbol2):
        try:
            if symbol1 not in self.crypto_data or symbol2 not in self.crypto_data: return 0.0
            returns1 = list(self.crypto_data[symbol1]['returns'])
            returns2 = list(self.crypto_data[symbol2]['returns'])
            min_len = min(len(returns1), len(returns2), self.medium_period)
            if min_len < 10: return 0.0
            r1 = np.array(returns1[-min_len:])
            r2 = np.array(returns2[-min_len:])
            std1, std2 = np.std(r1), np.std(r2)
            if std1 < 1e-10 or std2 < 1e-10: return 0.0
            corr = np.corrcoef(r1, r2)[0, 1]
            if np.isnan(corr) or np.isinf(corr): return 0.0
            return corr
        except Exception as e: return 0.0

    def _check_correlation(self, new_symbol):
        for held_symbol in self.entry_prices.keys():
            corr = self._calculate_correlation(new_symbol, held_symbol)
            if abs(corr) > self.correlation_threshold:
                return False, held_symbol, corr
        return True, None, 0.0

    def _sync_existing_positions(self):
        self.Debug("=" * 50)
        self.Debug("=== SYNCING EXISTING POSITIONS ===")
        synced_count = 0
        positions_to_close = []

        for symbol in self.Portfolio.Keys:
            holding = self.Portfolio[symbol]
            if not self._is_invested_not_dust(symbol): continue

            ticker = symbol.Value
            if symbol in self.entry_prices: continue

            if symbol not in self.Securities:
                try: self.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
                except: continue

            self.entry_prices[symbol] = holding.AveragePrice
            self.highest_prices[symbol] = holding.AveragePrice
            self.entry_times[symbol] = self.Time
            synced_count += 1

            current_price = self.Securities[symbol].Price if symbol in self.Securities else holding.Price
            pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
            self.Debug(f"SYNCED: {ticker} | Entry: ${holding.AveragePrice:.4f} | Now: ${current_price:.4f} | PnL: {pnl_pct:+.2%}")

            if current_price > holding.AveragePrice:
                self.highest_prices[symbol] = current_price

            if pnl_pct >= self.base_take_profit:
                positions_to_close.append((symbol, ticker, pnl_pct, "Sync TP"))
            elif pnl_pct <= -self.base_stop_loss:
                positions_to_close.append((symbol, ticker, pnl_pct, "Sync SL"))

        self.Debug(f"Synced {synced_count} positions")
        self.Debug(f"Cash: ${self.Portfolio.Cash:.2f}")
        self.Debug("=" * 50)

        for symbol, ticker, pnl_pct, reason in positions_to_close:
            self.Debug(f"IMMEDIATE {reason}: {ticker} at {pnl_pct:+.2%}")
            self._smart_liquidate(symbol, reason)
            if pnl_pct > 0: self.winning_trades += 1
            else: self.losing_trades += 1
            self._cleanup_position(symbol)

    def UniverseFilter(self, universe):
        selected = []
        for crypto in universe:
            ticker = crypto.Symbol.Value
            if ticker in self.SYMBOL_BLACKLIST or ticker in self._session_blacklist: continue
            if not ticker.endswith("USD"): continue
            if crypto.VolumeInUsd is None or crypto.VolumeInUsd == 0: continue
            if crypto.VolumeInUsd > self.min_volume_usd: selected.append(crypto)
        selected.sort(key=lambda x: x.VolumeInUsd, reverse=True)
        return [c.Symbol for c in selected[:self.max_universe_size]]

    def _initialize_symbol(self, symbol):
        self.crypto_data[symbol] = {
            'prices': deque(maxlen=self.lookback),
            'returns': deque(maxlen=self.lookback),
            'volume': deque(maxlen=self.lookback),
            'volume_ma': deque(maxlen=self.medium_period),
            'dollar_volume': deque(maxlen=self.lookback),
            'ema_short': ExponentialMovingAverage(self.short_period),
            'ema_medium': ExponentialMovingAverage(self.medium_period),
            'ema_long': ExponentialMovingAverage(self.long_period),
            'atr': AverageTrueRange(14),
            'volatility': deque(maxlen=self.medium_period),
            'rsi': RelativeStrengthIndex(14),
            'rs_vs_btc': deque(maxlen=self.medium_period),
            'zscore': deque(maxlen=self.short_period),
            'last_price': 0,
        }

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.crypto_data: self._initialize_symbol(symbol)

        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if not self.IsWarmingUp and self._is_invested_not_dust(symbol):
                self._smart_liquidate(symbol, "Removed from universe")
                self._cleanup_position(symbol)
                self.Debug(f"FORCED EXIT: {symbol.Value} - removed from universe")
            if symbol in self.crypto_data: del self.crypto_data[symbol]

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
            if not data.Bars.ContainsKey(symbol): continue
            try: self._update_symbol_data(symbol, data.Bars[symbol])
            except: pass

        if self.IsWarmingUp: return

        if not self._positions_synced:
            self._cancel_stale_orders()
            self._sync_existing_positions()
            self._positions_synced = True
            ready_count = sum(1 for c in self.crypto_data.values() if self._is_ready(c))
            self.Debug(f"Post-warmup: {ready_count} symbols ready")

        self._update_market_context()

    def _update_symbol_data(self, symbol, bar):
        crypto = self.crypto_data[symbol]
        price = float(bar.Close)
        volume = float(bar.Volume)
        crypto['prices'].append(price)
        if crypto['last_price'] > 0:
            ret = (price - crypto['last_price']) / crypto['last_price']
            crypto['returns'].append(ret)
        crypto['last_price'] = price
        crypto['volume'].append(volume)
        crypto['dollar_volume'].append(price * volume)
        if len(crypto['volume']) >= self.short_period:
            crypto['volume_ma'].append(np.mean(list(crypto['volume'])[-self.short_period:]))
        crypto['ema_short'].Update(bar.EndTime, price)
        crypto['ema_medium'].Update(bar.EndTime, price)
        crypto['ema_long'].Update(bar.EndTime, price)
        crypto['atr'].Update(bar)
        if len(crypto['returns']) >= 10:
            crypto['volatility'].append(np.std(list(crypto['returns'])[-10:]))
        crypto['rsi'].Update(bar.EndTime, price)
        if len(crypto['returns']) >= self.short_period and len(self.btc_returns) >= self.short_period:
            coin_ret = np.sum(list(crypto['returns'])[-self.short_period:])
            btc_ret = np.sum(list(self.btc_returns)[-self.short_period:])
            crypto['rs_vs_btc'].append(coin_ret - btc_ret)
        if len(crypto['prices']) >= self.medium_period:
            prices_arr = np.array(list(crypto['prices']))
            std = np.std(prices_arr)
            if std > 0: crypto['zscore'].append((price - np.mean(prices_arr)) / std)

    def _update_market_context(self):
        if len(self.btc_prices) >= self.medium_period:
            btc_arr = np.array(list(self.btc_prices))
            btc_sma = np.mean(btc_arr[-self.medium_period:])
            current_btc = btc_arr[-1]
            if current_btc > btc_sma * 1.03: self.market_regime = "bull"
            elif current_btc < btc_sma * 0.97: self.market_regime = "bear"
            else: self.market_regime = "sideways"

        if len(self.btc_volatility) >= 5:
            current_vol = self.btc_volatility[-1]
            avg_vol = np.mean(list(self.btc_volatility))
            if current_vol > avg_vol * 1.5: self.volatility_regime = "high"
            elif current_vol < avg_vol * 0.5: self.volatility_regime = "low"
            else: self.volatility_regime = "normal"

        uptrend_count = 0
        total_ready = 0
        for crypto in self.crypto_data.values():
            if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
                total_ready += 1
                if crypto['ema_short'].Current.Value > crypto['ema_medium'].Current.Value:
                    uptrend_count += 1
        if total_ready > 10: self.market_breadth = uptrend_count / total_ready

    def _annualized_vol(self, crypto):
        if crypto is None: return None
        if len(crypto.get('volatility', [])) == 0: return None
        return float(crypto['volatility'][-1]) * self.sqrt_annualization

    def _compute_portfolio_risk_estimate(self):
        total_value = self.Portfolio.TotalPortfolioValue
        if total_value <= 0: return 0.0
        risk = 0.0
        for kvp in self.Portfolio:
            symbol, holding = kvp.Key, kvp.Value
            if not self._is_invested_not_dust(symbol): continue
            crypto = self.crypto_data.get(symbol)
            asset_vol_ann = self._annualized_vol(crypto)
            if asset_vol_ann is None: asset_vol_ann = self.min_asset_vol_floor
            weight = abs(holding.HoldingsValue) / total_value
            risk += weight * asset_vol_ann
        return risk

    # === FIXED: Rebalance now uses _get_threshold() ===
    def Rebalance(self):
        if self.IsWarmingUp: return
        self._cancel_stale_new_orders()

        if self.Time != self.last_log_time:
            self.log_budget = 20
            self.last_log_time = self.Time

        if self.LiveMode and self.Time.hour in self.skip_hours_utc: return
        if self.daily_trade_count >= self.max_daily_trades: return

        val = self.Portfolio.TotalPortfolioValue
        if self.peak_value is None or self.peak_value < 1: self.peak_value = val

        if self.drawdown_cooldown > 0:
            self.drawdown_cooldown -= 2
            if self.drawdown_cooldown <= 0:
                self.peak_value = val
                self.consecutive_losses = 0
            else: return

        self.peak_value = max(self.peak_value, val)
        dd = (self.peak_value - val) / self.peak_value if self.peak_value > 0 else 0

        if dd > self.max_drawdown_limit:
            self.drawdown_cooldown = self.cooldown_hours
            self._notify(f"Drawdown {dd:.1%} exceeded limit. Cooling down.")
            return

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.drawdown_cooldown = 12
            self.consecutive_losses = 0
            return

        pos_count = self._get_actual_position_count()
        if pos_count >= self.max_positions: return

        # CRITICAL SAFETY: BLOCK TRADING IF ORDERS ARE STUCK
        open_orders = self.Transactions.GetOpenOrders()
        if len(open_orders) > 0:
            self._debug_limited(f"Skip Rebalance: {len(open_orders)} orders pending (possible lag)")
            return

        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap: return

        scores = [] 
        # FIX: _get_threshold restored below
        threshold_now = self._get_threshold()

        for symbol in list(self.crypto_data.keys()):
            if symbol.Value in self.SYMBOL_BLACKLIST or symbol.Value in self._session_blacklist: continue
            
            crypto = self.crypto_data[symbol]
            if not self._is_ready(crypto): continue

            factor_scores = self._calculate_factor_scores(symbol, crypto)
            if not factor_scores: continue
            
            composite_score = self._calculate_composite_score(factor_scores)
            net_score = self._apply_fee_adjustment(composite_score)
            
            if net_score > threshold_now:
                scores.append({
                    'symbol': symbol,
                    'composite_score': composite_score, 
                    'net_score': net_score,
                    'factors': factor_scores,
                    'volatility': crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                    'dollar_volume': list(crypto['dollar_volume'])[-6:] if len(crypto['dollar_volume']) >= 6 else [],
                })

        if len(scores) > 0:
            scores.sort(key=lambda x: x['net_score'], reverse=True)
            self._execute_trades(scores, threshold_now)

    def _execute_trades(self, candidates, threshold_now):
        if not self._positions_synced: return
        self._cancel_stale_new_orders()
        
        # Double check orders
        if len(self.Transactions.GetOpenOrders()) > 0: return

        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap: return

        for cand in candidates:
            if self.daily_trade_count >= self.max_daily_trades: break
            if self._get_actual_position_count() >= self.max_positions: break
            
            sym = cand['symbol']
            comp_score = cand.get('composite_score', 0.5)

            if self.Transactions.GetOpenOrders(sym): continue
            if self._is_invested_not_dust(sym): continue
            
            sec = self.Securities[sym]
            cash = self.Portfolio.Cash * 0.95
            
            min_qty = self._get_min_quantity(sym)
            if min_qty * sec.Price > cash * 0.6: continue

            vol = self._annualized_vol(self.crypto_data[sym])
            size = self._calculate_position_size(comp_score, threshold_now, vol)
            val = cash * size
            
            qty = self._round_quantity(sym, val / sec.Price)
            if qty < min_qty:
                qty = self._round_quantity(sym, min_qty)
                val = qty * sec.Price
            
            if val < self.min_notional or val > cash: continue

            try:
                self.MarketOrder(sym, qty)
                self.entry_prices[sym] = sec.Price
                self.highest_prices[sym] = sec.Price
                self.entry_times[sym] = self.Time
                self.trade_count += 1
                self._debug_limited(f"ORDER: {sym.Value} | ${val:.2f}")
            except Exception as e:
                self.Debug(f"ORDER FAILED: {sym.Value} - {e}")
                self._session_blacklist.add(sym.Value)
                continue

            if self.LiveMode: break

    # === RESTORED: _get_threshold ===
    def _get_threshold(self):
        """Get dynamic threshold."""
        if self.market_regime == "bull" and self.market_breadth > 0.6:
            return self.threshold_bull
        elif self.market_regime == "bear":
            return self.threshold_bear
        elif self.volatility_regime == "high":
            return self.threshold_high_vol
        return self.threshold_sideways

    def _is_ready(self, c):
        return len(c['prices']) >= self.medium_period and c['rsi'].IsReady

    def _calculate_factor_scores(self, symbol, crypto):
        try:
            scores = {}
            if len(crypto['rs_vs_btc']) >= 3:
                scores['relative_strength'] = self._normalize(np.mean(list(crypto['rs_vs_btc'])[-3:]), -0.05, 0.05)
            else: scores['relative_strength'] = 0.5
            
            if len(crypto['volume_ma']) >= 2 and len(crypto['returns']) >= 3:
                vol_ma_prev = crypto['volume_ma'][-2] if len(crypto['volume_ma']) >= 2 else 1
                vol_trend = (crypto['volume_ma'][-1] / (vol_ma_prev + 1e-8)) - 1
                price_trend = np.mean(list(crypto['returns'])[-3:])
                if vol_trend > 0 and price_trend > 0:
                    scores['volume_momentum'] = min(0.5 + vol_trend * 5 + price_trend * 25, 1.0)
                elif price_trend > 0: scores['volume_momentum'] = 0.55
                else: scores['volume_momentum'] = 0.3
            else: scores['volume_momentum'] = 0.5

            if crypto['ema_short'].IsReady:
                s, m, l = crypto['ema_short'].Current.Value, crypto['ema_medium'].Current.Value, crypto['ema_long'].Current.Value
                if s>m>l: scores['trend_strength'] = min(0.6 + ((s-l)/l)*10, 1.0)
                elif s>m: scores['trend_strength'] = 0.6
                elif s<m<l: scores['trend_strength'] = 0.2
                else: scores['trend_strength'] = 0.4
            else: scores['trend_strength'] = 0.5

            if len(crypto['zscore']) >= 1:
                z = crypto['zscore'][-1]
                rsi = crypto['rsi'].Current.Value
                if z < -1.5 and rsi < 35: scores['mean_reversion'] = 0.9
                elif z < -1.0 and rsi < 40: scores['mean_reversion'] = 0.75
                elif z > 2.0 or rsi > 75: scores['mean_reversion'] = 0.1
                else: scores['mean_reversion'] = 0.5
            else: scores['mean_reversion'] = 0.5

            if len(crypto['dollar_volume']) >= 12:
                avg = np.mean(list(crypto['dollar_volume'])[-12:])
                if avg > 10000: scores['liquidity'] = 0.9
                elif avg > 5000: scores['liquidity'] = 0.7
                elif avg > 1000: scores['liquidity'] = 0.5
                else: scores['liquidity'] = 0.2
            else: scores['liquidity'] = 0.5

            if len(crypto['returns']) >= self.medium_period:
                std = np.std(list(crypto['returns'])[-self.medium_period:])
                if std > 1e-10:
                    sharpe = np.mean(list(crypto['returns'])[-self.medium_period:]) / std
                    scores['risk_adjusted_momentum'] = self._normalize(sharpe, -1, 1)
                else: scores['risk_adjusted_momentum'] = 0.5
            else: scores['risk_adjusted_momentum'] = 0.5
            
            return scores
        except: return None

    def _normalize(self, v, mn, mx):
        return max(0, min(1, (v - mn)/(mx - mn)))

    def _calculate_composite_score(self, factors):
        score = sum(factors.get(f, 0.5)*w for f, w in self.weights.items())
        if self.market_regime == "bear": score *= 0.75
        if self.volatility_regime == "high": score *= 0.85
        if self.market_breadth > 0.7: score *= 1.05
        elif self.market_breadth < 0.3: score *= 0.85
        return min(score, 1.0)

    def _apply_fee_adjustment(self, score):
        return score - (self.expected_round_trip_fees / self.expected_holding_return)

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        conviction_mult = max(0.8, min(1.2, 0.8 + (score-threshold)*2))
        vol_floor = max(asset_vol_ann if asset_vol_ann else 0.05, 0.05)
        risk_mult = max(0.8, min(1.2, self.target_position_ann_vol / vol_floor))
        return self.position_size_pct * conviction_mult * risk_mult

    def CheckExits(self):
        if self.IsWarmingUp: return
        for kvp in self.Portfolio:
            if not self._is_invested_not_dust(kvp.Key): continue
            self._check_exit(kvp.Key, self.Securities[kvp.Key].Price, kvp.Value)

    def _check_exit(self, symbol, price, holding):
        if symbol not in self.entry_prices:
             self.entry_prices[symbol] = holding.AveragePrice
             self.highest_prices[symbol] = holding.AveragePrice
             self.entry_times[symbol] = self.Time
        
        entry = self.entry_prices[symbol]
        highest = self.highest_prices.get(symbol, entry)
        if price > highest: self.highest_prices[symbol] = price
        
        pnl = (price - entry) / entry
        dd = (highest - price) / highest
        hours = (self.Time - self.entry_times.get(symbol, self.Time)).total_seconds() / 3600
        
        sl, tp = self.base_stop_loss, self.base_take_profit
        if self.volatility_regime == "high":
            sl *= 1.2; tp *= 1.3
        elif self.market_regime == "bear":
            sl *= 0.8; tp *= 0.7

        tag = ""
        if pnl <= -sl: tag = "Stop Loss"
        elif pnl >= tp: tag = "Take Profit"
        elif pnl > self.trailing_activation and dd >= self.trailing_stop_pct: tag = "Trailing Stop"
        elif self.market_regime == "bear" and pnl > 0.03: tag = "Bear Exit"
        elif hours > 72 and pnl < 0.01: tag = "Time Exit"
        
        # Signal Decay
        if not tag and hours >= self.min_signal_age_hours:
            try:
                crypto = self.crypto_data.get(symbol)
                if crypto and self._is_ready(crypto):
                    factors = self._calculate_factor_scores(symbol, crypto)
                    if factors:
                        comp = self._calculate_composite_score(factors)
                        net = self._apply_fee_adjustment(comp)
                        if net < self._get_threshold() - self.signal_decay_buffer:
                            tag = "Signal Decay"
            except: pass

        if tag:
            self._smart_liquidate(symbol, tag)
            self._cleanup_position(symbol)
            if pnl > 0: 
                self.winning_trades += 1
                self.consecutive_losses = 0
            else: 
                self.losing_trades += 1
                self.consecutive_losses += 1
            
            self.total_pnl += pnl
            self.trade_log.append({
                'time': self.Time,
                'symbol': symbol.Value,
                'pnl_pct': pnl,
                'exit_reason': tag,
            })
            self.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.0f}h")

    def _cleanup_position(self, symbol):
        self.entry_prices.pop(symbol, None)
        self.highest_prices.pop(symbol, None)
        self.entry_times.pop(symbol, None)

    def OnOrderEvent(self, event):
        try:
            if event.Status == OrderStatus.Filled:
                if event.Direction == OrderDirection.Buy: 
                    self.daily_trade_count += 1
                    self._debug_limited(f"FILLED BUY: {event.Symbol.Value} @ ${event.FillPrice:.4f}")
                else:
                    self._debug_limited(f"FILLED SELL: {event.Symbol.Value} @ ${event.FillPrice:.4f}")
            elif event.Status == OrderStatus.Invalid:
                self.Debug(f"INVALID: {event.Symbol.Value} - {event.Message}")
                self._session_blacklist.add(event.Symbol.Value)
        except: pass
        
    def OnEndOfAlgorithm(self):
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        self.Debug("=== FINAL REPORT ===")
        self.Debug(f"Trades: {self.trade_count} | WR: {wr:.1%}")
        self.Debug(f"Final: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug(f"Total PnL: {self.total_pnl:+.2%}")

    def DailyReport(self):
        if self.IsWarmingUp: return
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        avg = self.total_pnl / total if total > 0 else 0
        
        self.Debug("=" * 50)
        self.Debug(f"=== DAILY REPORT {self.Time.date()} ===")
        self.Debug(f"Portfolio: ${self.Portfolio.TotalPortfolioValue:.2f} | Cash: ${self.Portfolio.Cash:.2f}")
        self.Debug(f"Positions: {self._get_actual_position_count()}/{self.max_positions}")
        self.Debug(f"Regime: {self.market_regime} | Vol: {self.volatility_regime} | Breadth: {self.market_breadth:.0%}")
        self.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
        if self._session_blacklist: self.Debug(f"Blacklist: {len(self._session_blacklist)} items")
        self.Debug("=" * 50)
        
        for kvp in self.Portfolio:
            if self._is_invested_not_dust(kvp.Key):
                s = kvp.Key
                entry = self.entry_prices.get(s, kvp.Value.AveragePrice)
                cur = self.Securities[s].Price if s in self.Securities else kvp.Value.Price
                pnl = (cur - entry) / entry if entry > 0 else 0
                self.Debug(f"  {s.Value}: ${entry:.4f}→${cur:.4f} ({pnl:+.2%})")

    def _debug_limited(self, msg):
        if self.log_budget > 0:
            self.Debug(msg)
            self.log_budget -= 1
        elif self.LiveMode:
            self.Debug(msg)
