# region imports
from AlgorithmImports import *
from collections import deque
import statistics
import json
# endregion

# =============================================================================
# SLIPPAGE MODEL
# =============================================================================

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
        
        if price < LOW_PRICE_THRESHOLD:
            slippage_pct *= LOW_PRICE_SLIPPAGE_MULT
        elif price < MID_PRICE_THRESHOLD:
            slippage_pct *= MID_PRICE_SLIPPAGE_MULT
        elif price < HIGH_PRICE_THRESHOLD:
            slippage_pct *= HIGH_PRICE_SLIPPAGE_MULT
        
        slippage_pct = min(slippage_pct, self.max_slippage_pct)
        
        return price * slippage_pct


# =============================================================================
# SYMBOL BLACKLIST
# =============================================================================

SYMBOL_BLACKLIST = {
    # Major pairs (excluded for strategy focus)
    "BTCUSD", "ETHUSD",
    # Stablecoins
    "USDTUSD", "USDCUSD", "PYUSDUSD", "EURCUSD", "USTUSD", "DAIUSD", "TUSDUSD",
    # Wrapped assets
    "WETHUSD", "WBTCUSD", "WAXLUSD",
    # Privacy coins
    "SHIBUSD", "XMRUSD", "ZECUSD", "DASHUSD",
    # Delisted or problematic
    "XNYUSD", "BDXNUSD", "RAIINUSD", "LUNAUSD", "LUNCUSD", "USTCUSD", "ABORDUSD",
    "BONDUSD", "KEEPUSD", "ORNUSD", "MUSD", "ICNTUSD", "EPTUSD", "LMWRUSD",
    "CPOOLUSD", "ARCUSD", "PAXGUSD", "PARTIUSD", "RAREUSD", "BANANAS31USD",
    # Forex pairs
    "GBPUSD", "AUDUSD", "NZDUSD", "JPYUSD", "CADUSD", "CHFUSD", "CNYUSD", "HKDUSD", 
    "SGDUSD", "SEKUSD", "NOKUSD", "DKKUSD", "KRWUSD", "TRYUSD", "ZARUSD", "MXNUSD", 
    "INRUSD", "BRLUSD", "PLNUSD", "THBUSD",
}


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum quantity and notional defaults
KRAKEN_MIN_QTY_FALLBACK = 0.0001
MIN_NOTIONAL_FALLBACK = 0.5

# Trading parameters
KRAKEN_MAX_FEE_RATE = 0.006  # 0.6% fee estimate (0.4% base + 0.2% safety buffer)
MIN_LIVE_CASH_USD = 2.0  # Minimum cash required to continue trading in live mode
LIVE_TRADE_RATE_LIMIT_SECONDS = 300  # Rate limit between trades in live mode (5 minutes)
MIN_KELLY_SAMPLE_SIZE = 10  # Minimum number of trades needed for Kelly fraction calculation

# Kelly fraction parameters
HALF_KELLY_DIVISOR = 0.5  # Divisor for half-kelly calculation
MIN_KELLY_MULTIPLIER = 0.5  # Minimum kelly fraction multiplier
MAX_KELLY_MULTIPLIER = 1.5  # Maximum kelly fraction multiplier

# Slippage model parameters
LOW_PRICE_THRESHOLD = 0.01  # Price below which to apply 3x slippage
MID_PRICE_THRESHOLD = 0.10  # Price below which to apply 2x slippage
HIGH_PRICE_THRESHOLD = 1.0  # Price below which to apply 1.5x slippage
LOW_PRICE_SLIPPAGE_MULT = 3.0
MID_PRICE_SLIPPAGE_MULT = 2.0
HIGH_PRICE_SLIPPAGE_MULT = 1.5


# =============================================================================
# EXECUTION MIXIN
# =============================================================================

class ExecutionMixin:
    """
    Mixin class providing execution-related methods for trading strategies.
    Handles order execution, position management, state persistence, and safety checks.
    """
    
    # -------------------------------------------------------------------------
    # ORDER MANAGEMENT
    # -------------------------------------------------------------------------
    
    def _smart_liquidate(self, symbol, tag="Liquidate"):
        """
        Intelligently liquidate a position with multiple safety checks.
        Handles open orders, cooldowns, minimum quantity/notional requirements, and fee reserves.
        """
        if len(self.Transactions.GetOpenOrders(symbol)) > 0:
            self.Debug(symbol.Value+"o")
            return
        if symbol in self._cancel_cooldowns and self.Time < self._cancel_cooldowns[symbol]:
            self.Debug(symbol.Value+"c")
            return
        if symbol not in self.Portfolio or self.Portfolio[symbol].Quantity == 0:
            self.Debug(symbol.Value+"p")
            return
        
        holding_qty = self.Portfolio[symbol].Quantity
        min_qty = self._get_min_quantity(symbol)
        min_notional = self._get_min_notional_usd(symbol)
        price = self.Securities[symbol].Price if symbol in self.Securities else 0
        
        if price * abs(holding_qty) < min_notional * 0.9:
            self.Debug(symbol.Value+"n")
            return
        
        # Verify fee reserve before selling (Kraken cash account requirement)
        if self.LiveMode and holding_qty > 0:
            estimated_fee = price * abs(holding_qty) * KRAKEN_MAX_FEE_RATE
            try:
                available_usd = self.Portfolio.CashBook["USD"].Amount
            except (KeyError, AttributeError):
                available_usd = self.Portfolio.Cash
            if available_usd < estimated_fee:
                self.Debug(symbol.Value+"f")
                if symbol not in self.entry_prices:
                    self.entry_prices[symbol] = self.Portfolio[symbol].AveragePrice
                    self.highest_prices[symbol] = self.Portfolio[symbol].AveragePrice
                    self.entry_times[symbol] = self.Time
                return
        
        self.Transactions.CancelOpenOrders(symbol)
        if abs(holding_qty) < min_qty:
            self.Debug(symbol.Value+"q")
            return
        
        safe_qty = self._round_quantity(symbol, abs(holding_qty))
        if safe_qty < min_qty:
            self.Debug(symbol.Value+"r")
            return
        
        if safe_qty > 0:
            direction_mult = -1 if holding_qty > 0 else 1
            self.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
    
    def _cancel_stale_orders(self):
        """Cancel all open orders (typically called during initialization)."""
        try:
            open_orders = self.Transactions.GetOpenOrders()
            if len(open_orders) > 0:
                for order in open_orders:
                    self.Transactions.CancelOrder(order.Id)
        except Exception as e:
            self.Debug(f"Error canceling stale orders: {e}")
    
    def _cancel_stale_new_orders(self):
        """
        Cancel orders that have been open longer than the stale timeout.
        Blacklists symbols with zombie orders.
        """
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
                    self.Transactions.CancelOrder(order.Id)
                    self._cancel_cooldowns[order.Symbol] = self.Time + timedelta(minutes=self.cancel_cooldown_minutes)
                    self._session_blacklist.add(sym_val)
        except Exception as e:
            self.Debug(f"Error in _cancel_stale_new_orders: {e}")
    
    def _effective_stale_timeout(self):
        """Get the appropriate stale order timeout based on live vs backtest mode."""
        return self.live_stale_order_timeout_seconds if self.LiveMode else self.stale_order_timeout_seconds
    
    # -------------------------------------------------------------------------
    # QUANTITY AND NOTIONAL CALCULATIONS
    # -------------------------------------------------------------------------
    
    def _get_min_quantity(self, symbol):
        """
        Get minimum order quantity for a symbol.
        Tries SymbolProperties first, then fallback dictionary, then estimation.
        """
        ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
        try:
            if symbol in self.Securities:
                sec = self.Securities[symbol]
                if hasattr(sec, 'SymbolProperties') and sec.SymbolProperties is not None:
                    min_size = sec.SymbolProperties.MinimumOrderSize
                    if min_size is not None and min_size > 0:
                        return float(min_size)
        except:
            pass
        if hasattr(self, 'KRAKEN_MIN_QTY_FALLBACK') and ticker in self.KRAKEN_MIN_QTY_FALLBACK:
            return self.KRAKEN_MIN_QTY_FALLBACK[ticker]
        return self._estimate_min_qty(symbol)
    
    def _estimate_min_qty(self, symbol):
        """Estimate minimum quantity based on price tiers."""
        try:
            price = self.Securities[symbol].Price if symbol in self.Securities else 0
        except:
            price = 0
        if price <= 0:
            return 50.0
        if price < 0.001:
            return 1000.0
        elif price < 0.01:
            return 500.0
        elif price < 0.1:
            return 50.0
        elif price < 1.0:
            return 10.0
        elif price < 10.0:
            return 5.0
        elif price < 100.0:
            return 1.0
        elif price < 1000.0:
            return 0.1
        else:
            return 0.01
    
    def _get_min_notional_usd(self, symbol):
        """
        Get minimum notional value (USD) required for an order.
        Uses fallback dictionary and calculated implied minimum.
        """
        ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
        fallback = (self.MIN_NOTIONAL_FALLBACK.get(ticker, self.min_notional) 
                   if hasattr(self, 'MIN_NOTIONAL_FALLBACK') else self.min_notional)
        try:
            price = self.Securities[symbol].Price
            min_qty = self._get_min_quantity(symbol)
            implied = price * min_qty if price > 0 else fallback
            return max(fallback, implied, self.min_notional)
        except:
            return max(fallback, self.min_notional)
    
    def _round_quantity(self, symbol, quantity):
        """Round quantity to appropriate lot size for the symbol."""
        try:
            lot_size = self.Securities[symbol].SymbolProperties.LotSize
            if lot_size is not None and lot_size > 0:
                return float(int(quantity / lot_size) * lot_size)
            return quantity
        except:
            return quantity
    
    # -------------------------------------------------------------------------
    # ORDER AND POSITION CHECKS
    # -------------------------------------------------------------------------
    
    def _has_open_orders(self, symbol=None):
        """Check if there are open orders for a symbol or any symbol."""
        if symbol is None:
            return len(self.Transactions.GetOpenOrders()) > 0
        return len(self.Transactions.GetOpenOrders(symbol)) > 0
    
    def _spread_ok(self, symbol):
        """
        Check if bid-ask spread is within acceptable limits.
        Applies stricter limits in high volatility or sideways regimes.
        """
        sp = self._get_spread_pct(symbol)
        if sp is None:
            # In live mode, fail-closed (reject if spread unknown); in backtest, allow
            return not self.LiveMode
        
        effective_spread_cap = self.max_spread_pct
        if self.LiveMode and (self.volatility_regime == "high" or self.market_regime == "sideways"):
            effective_spread_cap = min(effective_spread_cap, 0.015)
        
        if sp > effective_spread_cap:
            return False
        
        # Check if spread is abnormally widened compared to median
        crypto = self.crypto_data.get(symbol)
        if crypto and len(crypto.get('spreads', [])) >= 4:
            median_sp = statistics.median(list(crypto['spreads']))
            if median_sp > 0 and sp > self.spread_widen_mult * median_sp:
                return False
        
        return True
    
    def _get_spread_pct(self, symbol):
        """Calculate bid-ask spread as percentage of mid price."""
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
    
    # -------------------------------------------------------------------------
    # SLIPPAGE AND FILL LOGGING
    # -------------------------------------------------------------------------
    
    def _slip_log(self, symbol, direction, fill_price):
        """
        Log slippage from fills and alert if unusually high in live trading.
        Calculates slippage as deviation from mid price.
        """
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
        except:
            pass
    
    # -------------------------------------------------------------------------
    # POSITION HELPERS
    # -------------------------------------------------------------------------
    
    def _is_invested_not_dust(self, symbol):
        """
        Check if position is truly invested and not just dust.
        Considers both quantity and notional value thresholds.
        """
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
        """Get count of actual positions (excluding dust)."""
        return sum(1 for kvp in self.Portfolio if self._is_invested_not_dust(kvp.Key))
    
    # -------------------------------------------------------------------------
    # LIVE SAFETY CHECKS
    # -------------------------------------------------------------------------
    
    def _live_safety_checks(self):
        """
        Extra safety checks for live trading.
        Verifies minimum cash and enforces rate limiting.
        """
        if not self.LiveMode:
            return True
        
        # Check if we have minimum viable cash
        try:
            cash = self.Portfolio.CashBook["USD"].Amount
        except:
            cash = self.Portfolio.Cash
        
        if cash < MIN_LIVE_CASH_USD:
            self._debug_limited(f"LIVE SAFETY: Cash below ${MIN_LIVE_CASH_USD}, pausing new entries")
            return False
        
        # Rate limit: don't trade more than once per specified interval in live
        if hasattr(self, '_last_live_trade_time') and self._last_live_trade_time is not None:
            seconds_since = (self.Time - self._last_live_trade_time).total_seconds()
            if seconds_since < LIVE_TRADE_RATE_LIMIT_SECONDS:
                return False
        
        return True
    
    # -------------------------------------------------------------------------
    # STATE PERSISTENCE
    # -------------------------------------------------------------------------
    
    def _cleanup_object_store(self):
        """Clean up old ObjectStore keys (keeping only live state)."""
        try:
            n = 0
            for i in self.ObjectStore.GetEnumerator():
                k = i.Key if hasattr(i, 'Key') else str(i)
                if k != "opus_live_state":
                    try:
                        self.ObjectStore.Delete(k)
                        n += 1
                    except:
                        pass
            if n:
                self.Debug(f"Cleaned {n} keys")
        except Exception as e:
            self.Debug(f"Cleanup err: {e}")
    
    def _persist_state(self):
        """
        Persist important state to ObjectStore for live trading.
        Includes trade statistics, blacklist, and risk tracking.
        """
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
        """
        Load persisted state from ObjectStore on algorithm restart.
        Restores trade statistics, blacklist, and risk tracking.
        """
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
    
    # -------------------------------------------------------------------------
    # POSITION SYNCHRONIZATION
    # -------------------------------------------------------------------------
    
    def _sync_existing_positions(self):
        """
        Sync existing positions at algorithm start.
        Backfills tracking for any untracked holdings and immediately closes positions
        that hit stop-loss or take-profit thresholds.
        """
        self.Debug("=" * 50)
        self.Debug("=== SYNCING EXISTING POSITIONS ===")
        synced_count = 0
        positions_to_close = []
        
        for symbol in self.Portfolio.Keys:
            holding = self.Portfolio[symbol]
            if not holding.Invested or holding.Quantity == 0:
                continue
            if symbol in self.entry_prices:
                continue
            
            if symbol not in self.Securities:
                try:
                    self.AddCrypto(symbol.Value, Resolution.Hour, Market.Kraken)
                except:
                    continue
            
            self.entry_prices[symbol] = holding.AveragePrice
            self.highest_prices[symbol] = holding.AveragePrice
            self.entry_times[symbol] = self.Time
            synced_count += 1
            
            current_price = self.Securities[symbol].Price if symbol in self.Securities else holding.Price
            pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
            
            if current_price > holding.AveragePrice:
                self.highest_prices[symbol] = current_price
            
            if pnl_pct >= self.base_take_profit:
                positions_to_close.append((symbol, symbol.Value, pnl_pct, "Sync TP"))
            elif pnl_pct <= -self.base_stop_loss:
                positions_to_close.append((symbol, symbol.Value, pnl_pct, "Sync SL"))
        
        self.Debug(f"Synced {synced_count} positions")
        self.Debug(f"Cash: ${self.Portfolio.Cash:.2f}")
        self.Debug("=" * 50)
        
        for symbol, ticker, pnl_pct, reason in positions_to_close:
            self._smart_liquidate(symbol, reason)
    
    def ResyncHoldings(self):
        """
        Resync holdings during live trading to catch positions missed by event stream.
        Only runs in live mode.
        """
        if self.IsWarmingUp:
            return
        if not self.LiveMode:
            return
        
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
            except Exception as e:
                self.Debug(f"Resync error {symbol.Value}: {e}")
    
    # -------------------------------------------------------------------------
    # POSITION CLEANUP
    # -------------------------------------------------------------------------
    
    def _cleanup_position(self, symbol):
        """Remove all tracking data for a closed position."""
        self.entry_prices.pop(symbol, None)
        self.highest_prices.pop(symbol, None)
        self.entry_times.pop(symbol, None)
        if symbol in self.crypto_data:
            self.crypto_data[symbol]['trail_stop'] = None
    
    # -------------------------------------------------------------------------
    # KELLY FRACTION
    # -------------------------------------------------------------------------
    
    def _kelly_fraction(self):
        """
        Calculate Kelly fraction for position sizing based on recent win/loss history.
        Returns a multiplier between 0.5 and 1.5.
        """
        if len(self._rolling_wins) < MIN_KELLY_SAMPLE_SIZE:
            return 1.0
        
        win_rate = sum(self._rolling_wins) / len(self._rolling_wins)
        if win_rate <= 0 or win_rate >= 1:
            return 1.0
        
        avg_win = statistics.mean(list(self._rolling_win_sizes)) if len(self._rolling_win_sizes) > 0 else 0.02
        avg_loss = statistics.mean(list(self._rolling_loss_sizes)) if len(self._rolling_loss_sizes) > 0 else 0.02
        
        if avg_loss <= 0:
            return 1.0
        
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        half_kelly = kelly * HALF_KELLY_DIVISOR
        return max(MIN_KELLY_MULTIPLIER, min(MAX_KELLY_MULTIPLIER, half_kelly / HALF_KELLY_DIVISOR))
    
    # -------------------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------------------
    
    def _debug_limited(self, msg):
        """
        Rate-limited debug logging to avoid hitting API limits.
        Always logs critical messages (CANCELED, ZOMBIE, INVALID).
        """
        if "CANCELED" in msg or "ZOMBIE" in msg or "INVALID" in msg:
            self.Debug(msg)
            return
        if self.log_budget > 0:
            self.Debug(msg)
            self.log_budget -= 1
        elif self.LiveMode:
            self.Debug(msg)
