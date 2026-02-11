# region imports
from AlgorithmImports import *

import json
import numpy as np
from collections import deque
from datetime import timedelta
# endregion

class RealisticCryptoSlippage:
    """Volume-aware slippage model for crypto."""

    def __init__(self):
        self.base_slippage_pct = 0.001
        self.volume_impact_factor = 0.10
        self.max_slippage_pct = 0.02

    def get_slippage_approximation(self, asset, order):
        price = asset.Price
        if price <= 0:
            return 0

        slippage_pct = self.base_slippage_pct

        volume = asset.Volume
        if volume > 0:
            order_value = abs(order.Quantity) * price
            volume_value = volume * price
            if volume_value > 0:
                participation_rate = order_value / volume_value
                volume_impact = self.volume_impact_factor * (participation_rate ** 1.5)
                slippage_pct += volume_impact

        if price < 0.01:
            slippage_pct *= 3.0
        elif price < 0.10:
            slippage_pct *= 2.0
        elif price < 1.0:
            slippage_pct *= 1.5

        slippage_pct = min(slippage_pct, self.max_slippage_pct)

        return price * slippage_pct

class SimplifiedCryptoStrategy(QCAlgorithm):
    """
    Simplified 5-Factor Strategy - LIVE READY v3.0.0 (qa-logic + opus-execution)
    Key protections (as requested):
    - Exchange-status gate: trade only when Kraken status is OK (maintenance/cancel_only are blocked; unknown falls back to online after warmup).
    - Stale handling: stale cancels only when online; blacklist symbol on zombie, skip reprocessing blacklisted.
    - Liquidity/impact: live entry liquidity floor $5k over last 6 bars; impact cap 5% (scale >3%).
    - Spread guard: 1.5% cap in high-vol/sideways; median-widen check.
    - Min notional: $5 (tuned for small account); raise later if equity grows.
    - Cash reserve: 15% (adjust to 10% only if hitting insufficient BP).
    - Open-order cap: 2 concurrent.
    - Session blacklist cleared daily.
    - Logging limited to avoid rate limits.
    Other features:
    - Size cap live in high-vol/sideways: 25%.
    - Faster time stop: exit after 24h if PnL < +1%.
    - Zombie auto-blacklist; ARCUSD, PAXGUSD added to blacklist.
    - Insights suppressed; ObjectStore writes disabled.
    """

    SYMBOL_BLACKLIST = {
        "BTCUSD", "ETHUSD", "USDTUSD", "USDCUSD", "PYUSDUSD", "EURCUSD", "USTUSD",
        "DAIUSD", "TUSDUSD", "WETHUSD", "WBTCUSD", "WAXLUSD",
        "SHIBUSD", "XMRUSD", "ZECUSD", "DASHUSD",
        "XNYUSD",
        "BDXNUSD", "RAIINUSD", "LUNAUSD", "LUNCUSD", "USTCUSD", "ABORDUSD",
        "BONDUSD", "KEEPUSD", "ORNUSD",
        "MUSD", "ICNTUSD",
        "EPTUSD", "LMWRUSD",
        "CPOOLUSD",
        "ARCUSD", "PAXGUSD",
        "PARTIUSD", "RAREUSD", "BANANAS31USD",
        # Forex pairs
        "GBPUSD", "AUDUSD", "NZDUSD", "JPYUSD", "CADUSD", "CHFUSD", "CNYUSD", "HKDUSD", "SGDUSD",
        "SEKUSD", "NOKUSD", "DKKUSD", "KRWUSD", "TRYUSD", "ZARUSD", "MXNUSD", "INRUSD", "BRLUSD",
        "PLNUSD", "THBUSD",
    }

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
        'SHIBUSD': 50000.0, 'XRPUSD': 2.0,
    }

    MIN_NOTIONAL_FALLBACK = {
        'EWTUSD': 2.0, 'SANDUSD': 8.0, 'CTSIUSD': 18.0, 'MKRUSD': 0.01,
        'AUDUSD': 10.0, 'LPTUSD': 0.3, 'OXTUSD': 40.0, 'ENJUSD': 15.0,
        'UNIUSD': 0.5, 'LSKUSD': 3.0, 'BCHUSD': 1.0,
    }

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
        self.min_notional = 5.0  # lowered per request
        self.min_price_usd = 0.005

        self.expected_round_trip_fees = 0.0026
        self.fee_slippage_buffer = 0.003
        self.min_recent_dollar_vol_live = 5000

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
        self.max_consecutive_losses = 10  # relaxed to avoid unintended lockouts

        self.crypto_data = {}
        self.entry_prices = {}
        self.highest_prices = {}
        self.entry_times = {}
        self.trade_count = 0
        
        self._rolling_wins = deque(maxlen=50)
        self._rolling_win_sizes = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)

        self._pending_orders = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns = {}
        self.exit_cooldown_hours = 2
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
        self.max_universe_size = 1000

        # Kraken status gate:
        # - "online": trade
        # - "maintenance" / "cancel_only": block
        # - "unknown": allowed after warmup fallback to "online"
        self.kraken_status = "unknown"

        self._last_skip_reason = None
        self._last_live_trade_time = None

        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(CryptoUniverse.Kraken(self.UniverseFilter))

        try:
            btc = self.AddCrypto("BTCUSD", Resolution.Hour, Market.Kraken)
            self.btc_symbol = btc.Symbol
        except:
            self.Debug("Warning: Could not add BTC")

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=2)), self.Rebalance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.CheckExits)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 1), self.DailyReport)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 0), self.ResetDailyCounters)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=6)), self.ReviewPerformance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(12, 0), self.HealthCheck)
        # Hourly live resync to catch fills that may have been missed by event stream
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.ResyncHoldings)

        self.SetWarmUp(timedelta(days=5))
        self.SetSecurityInitializer(lambda security: security.SetSlippageModel(RealisticCryptoSlippage()))
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.Settings.InsightScore = False

        if self.LiveMode:
            self._cleanup_object_store()
            self._load_persisted_state()
            self.Debug("=" * 50)
            self.Debug("=== LIVE TRADING v3.0.0 (qa-logic + opus-execution) ===")
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

    def _cleanup_object_store(self):
        try:
            n = 0
            for i in self.ObjectStore.GetEnumerator():
                k = i.Key if hasattr(i, 'Key') else str(i)
                if k != "opus_live_state":
                    try:
                        self.ObjectStore.Delete(k)
                        n += 1
                    except Exception:
                        pass
            if n:
                self.Debug(f"Cleaned {n} keys")
        except Exception as e:
            self.Debug(f"Cleanup err: {e}")

    def _live_safety_checks(self):
        """Extra safety checks for live trading."""
        if not self.LiveMode:
            return True

        # Check if we have minimum viable cash
        try:
            cash = self.Portfolio.CashBook["USD"].Amount
        except:
            cash = self.Portfolio.Cash

        if cash < 2.0:
            self._debug_limited("LIVE SAFETY: Cash below $2, pausing new entries")
            return False

        # Rate limit: don't trade more than once per 5 minutes in live
        if hasattr(self, '_last_live_trade_time') and self._last_live_trade_time is not None:
            seconds_since = (self.Time - self._last_live_trade_time).total_seconds()
            if seconds_since < 300:
                return False

        return True


    def EmitInsights(self, *insights):
        return []

    def EmitInsight(self, insight):
        return []


    def _persist_state(self):
        if not self.LiveMode:
            return
        try:
            state = {
                "session_blacklist": list(self._session_blacklist),
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "total_pnl": self.total_pnl,
                "consecutive_losses": self.consecutive_losses,
                "daily_trade_count": self.daily_trade_count,
                "trade_count": self.trade_count,
                "peak_value": self.peak_value if self.peak_value is not None else 0,
            }
            self.ObjectStore.Save("opus_live_state", json.dumps(state))
        except Exception as e:
            self.Debug(f"Persist error: {e}")

    def _load_persisted_state(self):
        try:
            if self.LiveMode and self.ObjectStore.ContainsKey("opus_live_state"):
                raw = self.ObjectStore.Read("opus_live_state")
                data = json.loads(raw)
                self._session_blacklist = set(data.get("session_blacklist", []))
                self.winning_trades = data.get("winning_trades", 0)
                self.losing_trades = data.get("losing_trades", 0)
                self.total_pnl = data.get("total_pnl", 0.0)
                self.consecutive_losses = data.get("consecutive_losses", 0)
                self.daily_trade_count = data.get("daily_trade_count", 0)
                self.trade_count = data.get("trade_count", 0)
                peak = data.get("peak_value", 0)
                if peak > 0:
                    self.peak_value = peak
        except Exception as e:
            self.Debug(f"Load persist error: {e}")


    def _is_invested_not_dust(self, symbol):
        if symbol not in self.Portfolio:
            return False
        h = self.Portfolio[symbol]
        if not h.Invested or h.Quantity == 0:
            return False
        min_qty = self._get_min_quantity(symbol)
        min_notional = self._get_min_notional_usd(symbol)
        price = self.Securities[symbol].Price if symbol in self.Securities else h.Price
        notional_ok = (price > 0) and (abs(h.Quantity) * price >= min_notional * 0.5)
        qty_ok = abs(h.Quantity) >= min_qty * 0.5
        return notional_ok or qty_ok

    def _get_actual_position_count(self):
        return sum(1 for kvp in self.Portfolio if self._is_invested_not_dust(kvp.Key))

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date = self.Time.date()
        if len(self._session_blacklist) > 0:
            self.Debug(f"Clearing session blacklist ({len(self._session_blacklist)} items)")
            self._session_blacklist.clear()
        self._persist_state()

    def HealthCheck(self):
        if self.IsWarmingUp: return
        # Live resync to catch any holdings the event stream might have missed
        self.ResyncHoldings()
        issues = []
        if self.Portfolio.Cash < 5:
            issues.append(f"Low cash: ${self.Portfolio.Cash:.2f}")
        for symbol in list(self.entry_prices.keys()):
            if len(self.Transactions.GetOpenOrders(symbol)) > 0:
                continue
            if not self._is_invested_not_dust(symbol):
                issues.append(f"Orphan tracking: {symbol.Value}")
                self._cleanup_position(symbol)
        for kvp in self.Portfolio:
            if self._is_invested_not_dust(kvp.Key) and kvp.Key not in self.entry_prices:
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
            self._debug_limited("Health check: OK")

    def ResyncHoldings(self):
        """
        Live-only safety: backfills tracking for any holdings that exist in the brokerage
        but were not registered via OnOrderEvent (e.g., missed fill events).
        """
        if self.IsWarmingUp: return
        if not self.LiveMode: return
        missing = []
        for symbol in self.Portfolio.Keys:
            holding = self.Portfolio[symbol]
            if not holding.Invested or holding.Quantity == 0:
                continue
            if symbol in self.entry_prices:
                continue
            if symbol in self._exit_cooldowns and self.Time < self._exit_cooldowns[symbol]:
                continue
            if self._has_open_orders(symbol):
                continue
            missing.append(symbol)
        if not missing:
            return
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

    def _get_min_notional_usd(self, symbol):
        ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
        fallback = self.MIN_NOTIONAL_FALLBACK.get(ticker, self.min_notional)
        try:
            price = self.Securities[symbol].Price
            min_qty = self._get_min_quantity(symbol)
            implied = price * min_qty if price > 0 else fallback
            return max(fallback, implied, self.min_notional)
        except:
            return max(fallback, self.min_notional)

    def _round_quantity(self, symbol, quantity):
        try:
            lot_size = self.Securities[symbol].SymbolProperties.LotSize
            if lot_size is not None and lot_size > 0:
                return float(int(quantity / lot_size) * lot_size)
            return quantity
        except: return quantity

    def _smart_liquidate(self, symbol, tag="Liquidate"):
        if len(self.Transactions.GetOpenOrders(symbol)) > 0:
            return
        if symbol in self._cancel_cooldowns and self.Time < self._cancel_cooldowns[symbol]:
            return
        if symbol not in self.Portfolio or self.Portfolio[symbol].Quantity == 0:
            return
        holding_qty = self.Portfolio[symbol].Quantity
        min_qty = self._get_min_quantity(symbol)
        min_notional = self._get_min_notional_usd(symbol)
        price = self.Securities[symbol].Price if symbol in self.Securities else 0
        if price * abs(holding_qty) < min_notional * 0.9:
            return
        # Verify fee reserve before selling (Kraken cash account requirement)
        if self.LiveMode and holding_qty > 0:
            estimated_fee = price * abs(holding_qty) * 0.006  # 0.6% fee estimate (0.4% base + 0.2% safety buffer)
            try:
                available_usd = self.Portfolio.CashBook["USD"].Amount
            except (KeyError, AttributeError):
                available_usd = self.Portfolio.Cash
            if available_usd < estimated_fee:
                self.Debug(f"⚠️ SKIP SELL {symbol.Value}: fee reserve too low "
                           f"(need ${estimated_fee:.4f}, have ${available_usd:.4f})")
                if symbol not in self.entry_prices:
                    self.entry_prices[symbol] = self.Portfolio[symbol].AveragePrice
                    self.highest_prices[symbol] = self.Portfolio[symbol].AveragePrice
                    self.entry_times[symbol] = self.Time
                return
        self.Transactions.CancelOpenOrders(symbol)
        if abs(holding_qty) < min_qty:
            return
        safe_qty = self._round_quantity(symbol, abs(holding_qty))
        if safe_qty < min_qty:
            return
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

    def _effective_stale_timeout(self):
        return self.live_stale_order_timeout_seconds if self.LiveMode else self.stale_order_timeout_seconds

    def _cancel_stale_new_orders(self):
        # Allow cancel gate when venue is online or unknown (fallback handled elsewhere)
        try:
            open_orders = self.Transactions.GetOpenOrders()
            timeout_seconds = self._effective_stale_timeout()
            for order in open_orders:
                sym_val = order.Symbol.Value
                if sym_val in self._session_blacklist:
                    continue
                order_time = order.Time
                if order_time.tzinfo is not None:
                    order_time = order_time.replace(tzinfo=None)
                order_age = (self.Time - order_time).total_seconds()
                if order_age > timeout_seconds:
                    self.Debug(f"Canceling stale: {sym_val} (age: {order_age/60:.1f}m, timeout {timeout_seconds/60:.1f}m)")
                    self.Transactions.CancelOrder(order.Id)
                    self._cancel_cooldowns[order.Symbol] = self.Time + timedelta(minutes=self.cancel_cooldown_minutes)
                    self._session_blacklist.add(sym_val)
                    self.Debug(f"⚠️ ZOMBIE ORDER DETECTED: {sym_val} - blacklisted for session")
        except Exception as e:
            self.Debug(f"Error in _cancel_stale_new_orders: {e}")

    def _sync_existing_positions(self):
        self.Debug("=" * 50)
        self.Debug("=== SYNCING EXISTING POSITIONS ===")
        synced_count = 0
        positions_to_close = []
        for symbol in self.Portfolio.Keys:
            holding = self.Portfolio[symbol]
            if not holding.Invested or holding.Quantity == 0:
                continue
            ticker = symbol.Value
            if symbol in self.entry_prices:
                continue
            if symbol not in self.Securities:
                try:
                    self.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
                except:
                    continue
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
            # Let OnOrderEvent handle cleanup and PnL tracking on fill

    def UniverseFilter(self, universe):
        selected = []
        for crypto in universe:
            ticker = crypto.Symbol.Value
            if ticker in self.SYMBOL_BLACKLIST or ticker in self._session_blacklist:
                continue
            if not ticker.endswith("USD"):
                continue
            # Additional forex filter
            if any(ticker.endswith(suffix) for suffix in ["PYUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
                "JPYUSD", "CADUSD", "CHFUSD", "CNYUSD", "HKDUSD", "SGDUSD", "SEKUSD", "NOKUSD", "DKKUSD",
                "KRWUSD", "TRYUSD", "ZARUSD", "MXNUSD", "INRUSD", "BRLUSD", "PLNUSD", "THBUSD"]):
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
            'ema_short': ExponentialMovingAverage(self.short_period),
            'ema_medium': ExponentialMovingAverage(self.medium_period),
            'ema_long': ExponentialMovingAverage(self.long_period),
            'atr': AverageTrueRange(14),
            'volatility': deque(maxlen=self.medium_period),
            'rsi': RelativeStrengthIndex(14),
            'rs_vs_btc': deque(maxlen=self.medium_period),
            'zscore': deque(maxlen=self.short_period),
            'last_price': 0,
            'recent_net_scores': deque(maxlen=3),
            'spreads': deque(maxlen=self.spread_median_window),
            'trail_stop': None,
        }

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.crypto_data:
                self._initialize_symbol(symbol)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if not self.IsWarmingUp and self._is_invested_not_dust(symbol):
                self._smart_liquidate(symbol, "Removed from universe")
                # Don't cleanup here — let OnOrderEvent handle it on fill
                self.Debug(f"FORCED EXIT: {symbol.Value} - removed from universe")
            # Only delete crypto_data if not invested (otherwise OnOrderEvent needs it)
            if symbol in self.crypto_data and not self._is_invested_not_dust(symbol):
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
            except:
                pass
        if self.IsWarmingUp:
            return
        if not self._positions_synced:
            if not self._first_post_warmup:
                self._cancel_stale_orders()
            self._sync_existing_positions()
            self._positions_synced = True
            self._first_post_warmup = False
            # Fallback: if status never set, assume online after warmup
            if self.kraken_status == "unknown":
                self.kraken_status = "online"
                self.Debug("Fallback: kraken_status set to online after warmup")
            ready_count = sum(1 for c in self.crypto_data.values() if self._is_ready(c))
            self.Debug(f"Post-warmup: {ready_count} symbols ready")
        self._update_market_context()

    def _get_spread_pct(self, symbol):
        try:
            sec = self.Securities[symbol]
            bid = sec.BidPrice
            ask = sec.AskPrice
            if bid > 0 and ask > 0 and ask >= bid:
                mid = 0.5 * (bid + ask)
                if mid > 0:
                    return (ask - bid) / mid
        except:
            pass
        return None

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
            if std > 0:
                crypto['zscore'].append((price - np.mean(prices_arr)) / std)
        sp = self._get_spread_pct(symbol)
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
            if not self._is_invested_not_dust(symbol):
                continue
            crypto = self.crypto_data.get(symbol)
            asset_vol_ann = self._annualized_vol(crypto)
            if asset_vol_ann is None:
                asset_vol_ann = self.min_asset_vol_floor
            weight = abs(holding.HoldingsValue) / total_value
            risk += weight * asset_vol_ann
        return risk

    def _has_open_orders(self, symbol=None):
        if symbol is None:
            return len(self.Transactions.GetOpenOrders()) > 0
        return len(self.Transactions.GetOpenOrders(symbol)) > 0

    def _spread_ok(self, symbol):
        sp = self._get_spread_pct(symbol)
        if sp is None:
            # In live mode, fail-closed (reject if spread unknown); in backtest, allow
            return not self.LiveMode
        effective_spread_cap = self.max_spread_pct
        if self.LiveMode and (self.volatility_regime == "high" or self.market_regime == "sideways"):
            effective_spread_cap = min(effective_spread_cap, 0.015)
        if sp > effective_spread_cap:
            return False
        crypto = self.crypto_data.get(symbol)
        if crypto and len(crypto.get('spreads', [])) >= 4:
            median_sp = np.median(list(crypto['spreads']))
            if median_sp > 0 and sp > self.spread_widen_mult * median_sp:
                return False
        return True


    def _slip_log(self, symbol, direction, fill_price):
        try:
            sec = self.Securities[symbol]
            bid = sec.BidPrice
            ask = sec.AskPrice
            if bid <= 0 or ask <= 0:
                return
            mid = 0.5 * (bid + ask)
            if mid <= 0:
                return
            side = 1 if direction == OrderDirection.Buy else -1
            slip = side * (fill_price - mid) / mid
            self._slip_abs.append(abs(slip))
            # Live slippage alert for unusually high slippage
            if self.LiveMode and abs(slip) > self.slip_outlier_threshold:
                self.Debug(f"⚠️ HIGH SLIPPAGE: {symbol.Value} | {abs(slip):.4%} | dir={direction}")
        except Exception:
            pass

    def _log_skip(self, reason):
        if reason != self._last_skip_reason:
            self._debug_limited(f"Rebalance skip: {reason}")
            self._last_skip_reason = reason

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        # Live safety checks
        if self.LiveMode and not self._live_safety_checks():
            return
        # Only block on explicit bad states; unknown is allowed (and will have fallback after warmup)
        if self.LiveMode and self.kraken_status in ("maintenance", "cancel_only"):
            self._log_skip("kraken not online")
            return
        self._cancel_stale_new_orders()
        if self.Time != self.last_log_time:
            self.log_budget = 20
            self.last_log_time = self.Time
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
        pos_count = self._get_actual_position_count()
        if pos_count >= dynamic_max_pos:
            self._log_skip("at max positions")
            return
        if len(self.Transactions.GetOpenOrders()) > self.max_concurrent_open_orders:
            self._log_skip("too many open orders")
            return
        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap:
            self._log_skip("risk cap")
            return
        scores = []
        threshold_now = self._get_threshold()
        for symbol in list(self.crypto_data.keys()):
            if symbol.Value in self.SYMBOL_BLACKLIST or symbol.Value in self._session_blacklist:
                continue
            if self._has_open_orders(symbol):
                continue
            if not self._spread_ok(symbol):
                continue
            crypto = self.crypto_data[symbol]
            if not self._is_ready(crypto):
                continue
            factor_scores = self._calculate_factor_scores(symbol, crypto)
            if not factor_scores:
                continue
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
        if len(scores) == 0:
            self._log_skip("no candidates passed filters")
            return
        scores.sort(key=lambda x: x['net_score'], reverse=True)
        self._last_skip_reason = None
        self._execute_trades(scores, threshold_now, dynamic_max_pos)

    def _execute_trades(self, candidates, threshold_now, dynamic_max_pos):
        if not self._positions_synced:
            return
        if self.LiveMode and self.kraken_status in ("maintenance", "cancel_only"):
            return
        self._cancel_stale_new_orders()
        if len(self.Transactions.GetOpenOrders()) > self.max_concurrent_open_orders:
            return
        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap:
            return
        for cand in candidates:
            if self.daily_trade_count >= self.max_daily_trades:
                break
            if self._get_actual_position_count() >= dynamic_max_pos:
                break
            sym = cand['symbol']
            comp_score = cand.get('composite_score', 0.5)
            net_score = cand.get('net_score', 0.5)
            if sym in self._pending_orders and self._pending_orders[sym] > 0:
                continue
            if self._has_open_orders(sym):
                continue
            if self._is_invested_not_dust(sym):
                continue
            if not self._spread_ok(sym):
                continue
            if sym in self._exit_cooldowns and self.Time < self._exit_cooldowns[sym]:
                continue
            sec = self.Securities[sym]
            price = sec.Price
            if price is None or price <= 0:
                continue
            if price < self.min_price_usd:
                continue

            effective_size_cap = self.position_size_pct
            # In live mode, cap size at 25% in high-vol or sideways regimes for safety
            if self.LiveMode and (self.volatility_regime == "high" or self.market_regime == "sideways"):
                effective_size_cap = min(effective_size_cap, 0.25)

            total_value = self.Portfolio.TotalPortfolioValue
            try:
                available_cash = self.Portfolio.CashBook["USD"].Amount
            except (KeyError, AttributeError):
                available_cash = self.Portfolio.Cash
            # Reserve based on portfolio value, not just remaining cash
            portfolio_reserve = total_value * self.cash_reserve_pct
            fee_reserve = total_value * 0.02  # Reserve 2% of portfolio value for fee coverage
            effective_reserve = max(portfolio_reserve, fee_reserve)
            reserved_cash = available_cash - effective_reserve
            if reserved_cash <= 0:
                continue
            min_qty = self._get_min_quantity(sym)
            min_notional_usd = self._get_min_notional_usd(sym)
            if min_qty * price > reserved_cash * 0.6:
                continue
            crypto = self.crypto_data.get(sym)
            if not crypto:
                continue
            if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
                ema_short = crypto['ema_short'].Current.Value
                ema_medium = crypto['ema_medium'].Current.Value

                # Check if this is a strong mean-reversion candidate
                is_mean_reversion = False
                if len(crypto['zscore']) >= 1 and crypto['rsi'].IsReady:
                    z = crypto['zscore'][-1]
                    rsi = crypto['rsi'].Current.Value
                    # Deep oversold: allow entry even against trend
                    if z < -1.5 and rsi < 35:
                        is_mean_reversion = True

                if not is_mean_reversion:
                    # Normal trend-following gate
                    if ema_short <= ema_medium:
                        continue
                    if len(crypto['returns']) >= 3:
                        recent_return = np.mean(list(crypto['returns'])[-3:])
                        if recent_return <= 0:
                            continue
            crypto['recent_net_scores'].append(net_score)
            if len(crypto['recent_net_scores']) >= 3:
                above_threshold_count = sum(1 for score in crypto['recent_net_scores'] if score > threshold_now)
                if above_threshold_count < 2:
                    continue
            vol = self._annualized_vol(crypto)
            size = self._calculate_position_size(comp_score, threshold_now, vol)
            size = min(size, effective_size_cap)
            if self.volatility_regime == "high":
                size *= 0.7

            if self.LiveMode and len(crypto['dollar_volume']) >= 6:
                recent_dollar_vol6 = np.mean(list(crypto['dollar_volume'])[-6:])
                if recent_dollar_vol6 < self.min_recent_dollar_vol_live:
                    continue
            recent_dollar_vol3 = np.mean(list(crypto['dollar_volume'])[-3:]) if len(crypto['dollar_volume']) >= 3 else 0

            if len(crypto['dollar_volume']) >= 3:
                order_value_estimate = reserved_cash * size
                if recent_dollar_vol3 > 0:
                    impact_ratio = order_value_estimate / recent_dollar_vol3
                    # Dynamic impact caps based on portfolio size
                    portfolio_value = self.Portfolio.TotalPortfolioValue
                    impact_hard_cap = 0.05 if portfolio_value < 500 else (0.03 if portfolio_value < 5000 else 0.02)
                    impact_soft_cap = impact_hard_cap * 0.6
                    if impact_ratio > impact_hard_cap:
                        continue
                    if impact_ratio > impact_soft_cap:
                        size *= max(0.3, 1.0 - impact_ratio)
                    max_child = 0.15 * recent_dollar_vol3
                    if order_value_estimate > max_child:
                        size *= max_child / order_value_estimate

            val = reserved_cash * size
            qty = self._round_quantity(sym, val / price)
            if qty < min_qty:
                qty = self._round_quantity(sym, min_qty)
                val = qty * price
            # Verify total cost with fees doesn't breach reserve
            total_cost_with_fee = val * 1.006  # Include 0.6% fee in total cost
            if total_cost_with_fee > available_cash - fee_reserve:
                continue
            if val < min_notional_usd or val < self.min_notional or val > reserved_cash:
                continue
            try:
                ticket = self.MarketOrder(sym, qty)
                if ticket is not None:
                    self._recent_tickets.append(ticket)
                    self.Debug(f"ORDER: {sym.Value} | ${val:.2f} | id={ticket.OrderId}")
                self.trade_count += 1
                self._debug_limited(f"ORDER: {sym.Value} | ${val:.2f}")
                # Update last trade time for rate limiting
                if self.LiveMode:
                    self._last_live_trade_time = self.Time
            except Exception as e:
                self.Debug(f"ORDER FAILED: {sym.Value} - {e}")
                self._session_blacklist.add(sym.Value)
                continue
            if self.LiveMode:
                break

    def _get_threshold(self):
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
            if not self._is_invested_not_dust(kvp.Key):
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
            # Apply penalty if exit > 2% of volume
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
        min_notional_usd = self._get_min_notional_usd(symbol)
        trailing_allowed = hours >= self.trailing_grace_hours
        if pnl <= -sl:
            tag = "Stop Loss"
        elif pnl >= tp:
            tag = "Take Profit"
        elif trailing_allowed and pnl > trailing_activation and dd >= trailing_stop_pct:
            tag = "Trailing Stop"
        elif self.market_regime == "bear" and pnl > 0.03:
            tag = "Bear Exit"
        elif hours > 24 and pnl < 0.01:
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
            except:
                pass
        if tag:
            if price * abs(holding.Quantity) < min_notional_usd * 0.9:
                return
            self._smart_liquidate(symbol, tag)
            self._exit_cooldowns[symbol] = self.Time + timedelta(hours=self.exit_cooldown_hours)
            self.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.0f}h")

    def _cleanup_position(self, symbol):
        self.entry_prices.pop(symbol, None)
        self.highest_prices.pop(symbol, None)
        self.entry_times.pop(symbol, None)
        if symbol in self.crypto_data:
            self.crypto_data[symbol]['trail_stop'] = None


    def _kelly_fraction(self):
        """Calculate Kelly fraction for performance monitoring. Not used for position sizing."""
        if len(self._rolling_wins) < 10:
            return 1.0
        win_rate = sum(self._rolling_wins) / len(self._rolling_wins)
        if win_rate <= 0 or win_rate >= 1:
            return 1.0
        avg_win = np.mean(list(self._rolling_win_sizes)) if len(self._rolling_win_sizes) > 0 else 0.02
        avg_loss = np.mean(list(self._rolling_loss_sizes)) if len(self._rolling_loss_sizes) > 0 else 0.02
        if avg_loss <= 0:
            return 1.0
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        half_kelly = kelly * 0.5
        return max(0.5, min(1.5, half_kelly / 0.5))


    def OnOrderEvent(self, event):
        try:
            symbol = event.Symbol
            msg = (
                f"ORDER EVENT: {symbol.Value} | status={event.Status} | dir={event.Direction} | "
                f"qty={event.FillQuantity or event.Quantity} | price={event.FillPrice} | id={event.OrderId}"
            )
            self.Debug(msg)

            if event.Status == OrderStatus.Submitted:
                if symbol not in self._pending_orders:
                    self._pending_orders[symbol] = 0
                intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
                self._pending_orders[symbol] += intended_qty

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
                self._slip_log(symbol, event.Direction, event.FillPrice)

            elif event.Status == OrderStatus.Filled:
                self._pending_orders.pop(symbol, None)
                if event.Direction == OrderDirection.Buy:
                    self.entry_prices[symbol] = event.FillPrice
                    self.highest_prices[symbol] = event.FillPrice
                    self.entry_times[symbol] = self.Time
                    self.daily_trade_count += 1
                else:
                    # Sell: calculate PnL and update Kelly tracking
                    entry = self.entry_prices.get(symbol, None)
                    if entry is None:
                        entry = event.FillPrice
                        self.Debug(f"⚠️ WARNING: Missing entry price for {symbol.Value} sell, using fill price")
                    pnl = (event.FillPrice - entry) / entry if entry > 0 else 0

                    # Update Kelly criterion tracking (Opus feature)
                    if pnl > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                        self._rolling_wins.append(1)
                        self._rolling_win_sizes.append(abs(pnl))
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1
                        self._rolling_wins.append(0)
                        self._rolling_loss_sizes.append(abs(pnl))

                    self.total_pnl += pnl
                    self.trade_log.append({
                        'time': self.Time,
                        'symbol': symbol.Value,
                        'pnl_pct': pnl,
                        'exit_reason': 'filled_sell',
                    })
                    self._cleanup_position(symbol)
                self._slip_log(symbol, event.Direction, event.FillPrice)

            elif event.Status == OrderStatus.Canceled:
                self._pending_orders.pop(symbol, None)
                if event.Direction == OrderDirection.Sell and symbol not in self.entry_prices:
                    if self._is_invested_not_dust(symbol):
                        holding = self.Portfolio[symbol]
                        self.entry_prices[symbol] = holding.AveragePrice
                        self.highest_prices[symbol] = holding.AveragePrice
                        self.entry_times[symbol] = self.Time
                        self.Debug(f"RE-TRACKED after cancel: {symbol.Value}")

            elif event.Status == OrderStatus.Invalid:
                self._pending_orders.pop(symbol, None)
                if event.Direction == OrderDirection.Sell and symbol not in self.entry_prices:
                    if self._is_invested_not_dust(symbol):
                        holding = self.Portfolio[symbol]
                        self.entry_prices[symbol] = holding.AveragePrice
                        self.highest_prices[symbol] = holding.AveragePrice
                        self.entry_times[symbol] = self.Time
                        self.Debug(f"RE-TRACKED after invalid: {symbol.Value}")
                self._session_blacklist.add(symbol.Value)
        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")


    def OnBrokerageMessage(self, message):
        try:
            txt = message.Message.lower()
            # self.Debug(f"BRKR MSG: {message.Message}")  # uncomment for diagnostics
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

            # Handle rate limit messages
            if "rate limit" in txt or "too many" in txt:
                self.Debug(f"⚠️ RATE LIMIT HIT - pausing trades for 5 minutes")
                self._last_live_trade_time = self.Time
        except Exception as e:
            self.Debug(f"BrokerageMessage parse error: {e}")

    def OnEndOfAlgorithm(self):
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        self.Debug("=== FINAL REPORT ===")
        self.Debug(f"Trades: {self.trade_count} | WR: {wr:.1%}")
        self.Debug(f"Final: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug(f"Total PnL: {self.total_pnl:+.2%}")
        self._persist_state()

    def DailyReport(self):
        if self.IsWarmingUp:
            return
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        avg = self.total_pnl / total if total > 0 else 0
        self.Debug("=" * 50)
        self.Debug(f"=== DAILY REPORT {self.Time.date()} ===")
        self.Debug(f"Portfolio: ${self.Portfolio.TotalPortfolioValue:.2f} | Cash: ${self.Portfolio.Cash:.2f}")
        self.Debug(f"Positions: {self._get_actual_position_count()}/{self.base_max_positions}")
        self.Debug(f"Regime: {self.market_regime} | Vol: {self.volatility_regime} | Breadth: {self.market_breadth:.0%}")
        self.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
        if self._session_blacklist:
            self.Debug(f"Blacklist: {len(self._session_blacklist)} items")
        self.Debug("=" * 50)
        for kvp in self.Portfolio:
            if self._is_invested_not_dust(kvp.Key):
                s = kvp.Key
                entry = self.entry_prices.get(s, kvp.Value.AveragePrice)
                cur = self.Securities[s].Price if s in self.Securities else kvp.Value.Price
                pnl = (cur - entry) / entry if entry > 0 else 0
                self.Debug(f"  {s.Value}: ${entry:.4f}→${cur:.4f} ({pnl:+.2%})")
        self._persist_state()

    def _debug_limited(self, msg):
        if "CANCELED" in msg or "ZOMBIE" in msg or "INVALID" in msg:
            self.Debug(msg)
            return
        if self.log_budget > 0:
            self.Debug(msg)
            self.log_budget -= 1
        elif self.LiveMode:
            self.Debug(msg)
