# region imports
from AlgorithmImports import *
import json
import numpy as np
from collections import deque
from datetime import timedelta
# endregion

# Constants from main_qa.py lines 31-68
SYMBOL_BLACKLIST = {
    "USDTUSD", "USDCUSD", "PYUSDUSD", "EURCUSD", "USTUSD",
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
    "PLNUSD", "THBUSD","BTCUSD", "ETHUSD"
}

KRAKEN_MIN_QTY_FALLBACK = {
    'AXSUSD': 5.0, 'SANDUSD': 10.0, 'MANAUSD': 10.0, 'ADAUSD': 10.0,
    'MATICUSD': 10.0, 'DOTUSD': 1.0, 'LINKUSD': 0.5, 'AVAXUSD': 0.2,
    'ATOMUSD': 0.5, 'NEARUSD': 1.0, 'SOLUSD': 0.05,
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


def get_min_quantity(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    try:
        if symbol in algo.Securities:
            sec = algo.Securities[symbol]
            if hasattr(sec, 'SymbolProperties') and sec.SymbolProperties is not None:
                min_size = sec.SymbolProperties.MinimumOrderSize
                if min_size is not None and min_size > 0:
                    return float(min_size)
    except Exception as e:
        algo.Debug(f"Error getting min quantity for {ticker}: {e}")
        pass
    if ticker in KRAKEN_MIN_QTY_FALLBACK:
        return KRAKEN_MIN_QTY_FALLBACK[ticker]
    return estimate_min_qty(algo, symbol)


def estimate_min_qty(algo, symbol):
    try:
        price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    except Exception as e:
        algo.Debug(f"Error getting price for min qty estimate: {e}")
        price = 0
    if price <= 0: return 50.0
    if price < 0.001: return 1000.0
    elif price < 0.01: return 500.0
    elif price < 0.1: return 50.0
    elif price < 1.0: return 10.0
    elif price < 10.0: return 5.0
    elif price < 100.0: return 1.0
    elif price < 1000.0: return 0.1
    else: return 0.01


def get_min_notional_usd(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    fallback = MIN_NOTIONAL_FALLBACK.get(ticker, algo.min_notional)
    try:
        price = algo.Securities[symbol].Price
        min_qty = get_min_quantity(algo, symbol)
        implied = price * min_qty if price > 0 else fallback
        return max(fallback, implied, algo.min_notional)
    except Exception as e:
        algo.Debug(f"Error in get_min_notional_usd for {symbol.Value}: {e}")
        return max(fallback, algo.min_notional)


def round_quantity(algo, symbol, quantity):
    try:
        lot_size = algo.Securities[symbol].SymbolProperties.LotSize
        if lot_size is not None and lot_size > 0:
            return float(int(quantity / lot_size) * lot_size)
        return quantity
    except Exception as e:
        algo.Debug(f"Error rounding quantity for {symbol.Value}: {e}")
        return quantity


def smart_liquidate(algo, symbol, tag="Liquidate"):
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return
    if symbol in algo._cancel_cooldowns and algo.Time < algo._cancel_cooldowns[symbol]:
        return
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return
    holding_qty = algo.Portfolio[symbol].Quantity
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price * abs(holding_qty) < min_notional * 0.9:
        return
    # Verify fee reserve before selling (Kraken cash account requirement)
    if algo.LiveMode and holding_qty > 0:
        estimated_fee = price * abs(holding_qty) * 0.006  # 0.6% fee estimate (0.4% base + 0.2% safety buffer)
        try:
            available_usd = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_usd = algo.Portfolio.Cash
        is_stop_loss = "Stop Loss" in tag or "Stop" in tag
        if available_usd < estimated_fee and not is_stop_loss:
            algo.Debug(f"⚠️ SKIP SELL {symbol.Value}: fee reserve too low "
                       f"(need ${estimated_fee:.4f}, have ${available_usd:.4f})")
            if symbol not in algo.entry_prices:
                algo.entry_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.highest_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.entry_times[symbol] = algo.Time
            return
    algo.Transactions.CancelOpenOrders(symbol)
    if abs(holding_qty) < min_qty:
        return
    safe_qty = round_quantity(algo, symbol, abs(holding_qty))
    if safe_qty < min_qty:
        return
    if safe_qty > 0:
        direction_mult = -1 if holding_qty > 0 else 1
        # Spread-aware exit logic
        is_stop_loss = "Stop Loss" in tag
        if not is_stop_loss:
            spread_pct = get_spread_pct(algo, symbol)
            if spread_pct is not None:
                if spread_pct > 0.03:  # 3% spread - log warning but still exit with market
                    algo.Debug(f"⚠️ WIDE SPREAD EXIT: {symbol.Value} spread={spread_pct:.2%}, using market order")
                    algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                elif algo.LiveMode and spread_pct > 0.015:  # 1.5% spread - use limit order with fallback (live mode only)
                    try:
                        sec = algo.Securities[symbol]
                        bid = sec.BidPrice
                        ask = sec.AskPrice
                        if bid > 0 and ask > 0:
                            mid = 0.5 * (bid + ask)
                            limit_order = algo.LimitOrder(symbol, safe_qty * direction_mult, mid, tag=tag)
                            # Track for fallback in VerifyOrderFills (90 second timeout handled there)
                            if hasattr(algo, '_submitted_orders'):
                                algo._submitted_orders[symbol] = {
                                    'order_id': limit_order.OrderId,
                                    'time': algo.Time,
                                    'quantity': safe_qty * direction_mult,  # Store signed quantity
                                    'is_limit_exit': True,
                                    'intent': 'exit'
                                }
                            algo.Debug(f"LIMIT EXIT: {symbol.Value} at mid ${mid:.4f} (spread={spread_pct:.2%})")
                        else:
                            algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    except Exception as e:
                        algo.Debug(f"Error placing limit exit for {symbol.Value}: {e}")
                        algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                else:
                    # Spread is acceptable, use market order
                    algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
            else:
                # Spread unknown, use market order
                algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
        else:
            # Stop loss - always use market order
            algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
    else:
        algo.Debug(f"Warning: {symbol.Value} holding {holding_qty} rounds to 0")


def cancel_stale_orders(algo):
    try:
        open_orders = algo.Transactions.GetOpenOrders()
        if len(open_orders) > 0:
            algo.Debug(f"Found {len(open_orders)} open orders - canceling all...")
            for order in open_orders:
                algo.Transactions.CancelOrder(order.Id)
    except Exception as e:
        algo.Debug(f"Error canceling stale orders: {e}")


def effective_stale_timeout(algo):
    return algo.live_stale_order_timeout_seconds if algo.LiveMode else algo.stale_order_timeout_seconds


def cancel_stale_new_orders(algo):
    # Allow cancel gate when venue is online or unknown (fallback handled elsewhere)
    try:
        open_orders = algo.Transactions.GetOpenOrders()
        timeout_seconds = effective_stale_timeout(algo)
        for order in open_orders:
            sym_val = order.Symbol.Value
            if sym_val in algo._session_blacklist:
                continue
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age > timeout_seconds:
                algo.Debug(f"Canceling stale: {sym_val} (age: {order_age/60:.1f}m, timeout {timeout_seconds/60:.1f}m)")
                
                # Check if this order actually filled (fill event missed)
                if is_invested_not_dust(algo, order.Symbol):
                    # Position exists — order filled, we just missed the event
                    # Re-track instead of blacklisting
                    algo.Debug(f"STALE ORDER but position exists: {sym_val} — re-tracking")
                    holding = algo.Portfolio[order.Symbol]
                    algo.entry_prices[order.Symbol] = holding.AveragePrice
                    algo.highest_prices[order.Symbol] = holding.AveragePrice
                    algo.entry_times[order.Symbol] = algo.Time
                    algo.Transactions.CancelOrder(order.Id)  # Cancel the stale order
                    continue  # Don't blacklist
                
                algo.Transactions.CancelOrder(order.Id)
                algo._cancel_cooldowns[order.Symbol] = algo.Time + timedelta(minutes=algo.cancel_cooldown_minutes)
                
                # Only blacklist stale ENTRY orders, not EXIT orders
                # Exit orders that are stale just get cooldown to allow retry
                has_position_or_tracked = order.Symbol in algo.entry_prices or (
                    order.Symbol in algo.Portfolio and algo.Portfolio[order.Symbol].Quantity != 0
                )
                
                if has_position_or_tracked:
                    algo.Debug(f"STALE EXIT: {sym_val} - cooldown only, not blacklisted")
                else:
                    algo._session_blacklist.add(sym_val)
                    algo.Debug(f"⚠️ ZOMBIE ORDER DETECTED: {sym_val} - blacklisted for session")
    except Exception as e:
        algo.Debug(f"Error in _cancel_stale_new_orders: {e}")


def is_invested_not_dust(algo, symbol):
    if symbol not in algo.Portfolio:
        return False
    h = algo.Portfolio[symbol]
    if not h.Invested or h.Quantity == 0:
        return False
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else h.Price
    notional_ok = (price > 0) and (abs(h.Quantity) * price >= min_notional * 0.5)
    qty_ok = abs(h.Quantity) >= min_qty * 0.5
    return notional_ok or qty_ok


def get_actual_position_count(algo):
    return sum(1 for kvp in algo.Portfolio if is_invested_not_dust(algo, kvp.Key))


def has_open_orders(algo, symbol=None):
    if symbol is None:
        return len(algo.Transactions.GetOpenOrders()) > 0
    return len(algo.Transactions.GetOpenOrders(symbol)) > 0


def has_non_stale_open_orders(algo, symbol):
    """Check if symbol has open orders that are NOT stale (younger than timeout)."""
    try:
        orders = algo.Transactions.GetOpenOrders(symbol)
        if len(orders) == 0:
            return False
        timeout_seconds = effective_stale_timeout(algo)
        for order in orders:
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age <= timeout_seconds:
                return True  # At least one order is not stale
        return False  # All orders are stale
    except Exception:
        return False


def get_spread_pct(algo, symbol):
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid > 0 and ask > 0 and ask >= bid:
            mid = 0.5 * (bid + ask)
            if mid > 0:
                return (ask - bid) / mid
    except Exception as e:
        algo.Debug(f"Error getting spread for {symbol.Value}: {e}")
        pass
    return None


def spread_ok(algo, symbol):
    sp = get_spread_pct(algo, symbol)
    if sp is None:
        # In live mode with small accounts (< $100), allow unknown spreads
        # Otherwise in live mode, fail-closed (reject if spread unknown); in backtest, allow
        if algo.LiveMode:
            portfolio_value = algo.Portfolio.TotalPortfolioValue
            if portfolio_value < 100:
                return True
            return False
        return True
    effective_spread_cap = algo.max_spread_pct
    if algo.LiveMode and (algo.volatility_regime == "high" or algo.market_regime == "sideways"):
        effective_spread_cap = min(effective_spread_cap, 0.025)
    if sp > effective_spread_cap:
        return False
    crypto = algo.crypto_data.get(symbol)
    if crypto and len(crypto.get('spreads', [])) >= 4:
        median_sp = np.median(list(crypto['spreads']))
        if median_sp > 0 and sp > algo.spread_widen_mult * median_sp:
            return False
    return True


def cleanup_position(algo, symbol):
    algo.entry_prices.pop(symbol, None)
    algo.highest_prices.pop(symbol, None)
    algo.entry_times.pop(symbol, None)
    if symbol in algo.crypto_data:
        algo.crypto_data[symbol]['trail_stop'] = None


def sync_existing_positions(algo):
    algo.Debug("=" * 50)
    algo.Debug("=== SYNCING EXISTING POSITIONS ===")
    synced_count = 0
    positions_to_close = []
    for symbol in algo.Portfolio.Keys:
        holding = algo.Portfolio[symbol]
        if not holding.Invested or holding.Quantity == 0:
            continue
        ticker = symbol.Value
        if symbol in algo.entry_prices:
            continue
        if symbol not in algo.Securities:
            try:
                algo.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
            except Exception as e:
                algo.Debug(f"Error adding crypto {ticker}: {e}")
                continue
        algo.entry_prices[symbol] = holding.AveragePrice
        algo.highest_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol] = algo.Time
        synced_count += 1
        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
        pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
        algo.Debug(f"SYNCED: {ticker} | Entry: ${holding.AveragePrice:.4f} | Now: ${current_price:.4f} | PnL: {pnl_pct:+.2%}")
        if current_price > holding.AveragePrice:
            algo.highest_prices[symbol] = current_price
        if pnl_pct >= algo.base_take_profit:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync TP"))
        elif pnl_pct <= -algo.base_stop_loss:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync SL"))
    algo.Debug(f"Synced {synced_count} positions")
    algo.Debug(f"Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug("=" * 50)
    for symbol, ticker, pnl_pct, reason in positions_to_close:
        algo.Debug(f"IMMEDIATE {reason}: {ticker} at {pnl_pct:+.2%}")
        smart_liquidate(algo, symbol, reason)
        # Let OnOrderEvent handle cleanup and PnL tracking on fill


def resync_holdings(algo):
    """
    Live-only safety: backfills tracking for any holdings that exist in the brokerage
    but were not registered via OnOrderEvent (e.g., missed fill events).
    """
    if algo.IsWarmingUp: return
    if not algo.LiveMode: return
    missing = []
    for symbol in algo.Portfolio.Keys:
        holding = algo.Portfolio[symbol]
        if not holding.Invested or holding.Quantity == 0:
            continue
        if symbol in algo.entry_prices:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        # Check if there are non-stale open orders
        # If all open orders are stale, we should resync anyway
        if has_non_stale_open_orders(algo, symbol):
            continue
        missing.append(symbol)
    if not missing:
        return
    algo.Debug(f"RESYNC: detected {len(missing)} holdings without tracking; backfilling.")
    for symbol in missing:
        try:
            if symbol not in algo.Securities:
                algo.AddCrypto(symbol.Value, Resolution.Hour, Market.Kraken)
            holding = algo.Portfolio[symbol]
            entry = holding.AveragePrice
            algo.entry_prices[symbol] = entry
            algo.highest_prices[symbol] = entry
            algo.entry_times[symbol] = algo.Time
            current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
            pnl_pct = (current_price - entry) / entry if entry > 0 else 0
            algo.Debug(f"RESYNCED: {symbol.Value} | Qty: {holding.Quantity} | Entry: ${entry:.4f} | Now: ${current_price:.4f} | PnL: {pnl_pct:+.2%}")
        except Exception as e:
            algo.Debug(f"Resync error {symbol.Value}: {e}")


def debug_limited(algo, msg):
    if "CANCELED" in msg or "ZOMBIE" in msg or "INVALID" in msg:
        algo.Debug(msg)
        return
    if algo.log_budget > 0:
        algo.Debug(msg)
        algo.log_budget -= 1
    elif algo.LiveMode:
        algo.Debug(msg)


def slip_log(algo, symbol, direction, fill_price):
    """Enhanced slip_log with live outlier alert and symbol-level slippage tracking."""
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid <= 0 or ask <= 0:
            return
        mid = 0.5 * (bid + ask)
        if mid <= 0:
            return
        side = 1 if direction == OrderDirection.Buy else -1
        slip = side * (fill_price - mid) / mid
        abs_slip = abs(slip)
        algo._slip_abs.append(abs_slip)
        
        # Track slippage per symbol
        if hasattr(algo, '_symbol_slippage_history'):
            ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
            if ticker not in algo._symbol_slippage_history:
                algo._symbol_slippage_history[ticker] = deque(maxlen=10)
            algo._symbol_slippage_history[ticker].append(abs_slip)
        
        # Live slippage alert for unusually high slippage
        if algo.LiveMode and abs(slip) > algo.slip_outlier_threshold:
            algo.Debug(f"⚠️ HIGH SLIPPAGE: {symbol.Value} | {abs(slip):.4%} | dir={direction}")
    except Exception as e:
        algo.Debug(f"Error in slip_log for {symbol.Value}: {e}")
        pass


def persist_state(algo):
    """Enhanced persist_state with trade_count and peak_value from main_opus."""
    if not algo.LiveMode:
        return
    try:
        state = {
            "session_blacklist": list(algo._session_blacklist),
            "winning_trades": algo.winning_trades,
            "losing_trades": algo.losing_trades,
            "total_pnl": algo.total_pnl,
            "consecutive_losses": algo.consecutive_losses,
            "daily_trade_count": algo.daily_trade_count,
            "trade_count": algo.trade_count,
            "peak_value": algo.peak_value if algo.peak_value is not None else 0,
        }
        algo.ObjectStore.Save("live_state", json.dumps(state))
    except Exception as e:
        algo.Debug(f"Persist error: {e}")


def load_persisted_state(algo):
    """Enhanced load_persisted_state with trade_count and peak_value from main_opus."""
    try:
        if algo.LiveMode and algo.ObjectStore.ContainsKey("live_state"):
            raw = algo.ObjectStore.Read("live_state")
            data = json.loads(raw)
            algo._session_blacklist = set(data.get("session_blacklist", []))
            algo.winning_trades = data.get("winning_trades", 0)
            algo.losing_trades = data.get("losing_trades", 0)
            algo.total_pnl = data.get("total_pnl", 0.0)
            algo.consecutive_losses = data.get("consecutive_losses", 0)
            algo.daily_trade_count = data.get("daily_trade_count", 0)
            algo.trade_count = data.get("trade_count", 0)
            peak = data.get("peak_value", 0)
            if peak > 0:
                algo.peak_value = peak
            algo.Debug(f"Loaded persisted state: blacklist {len(algo._session_blacklist)}, "
                       f"trades W:{algo.winning_trades}/L:{algo.losing_trades}")
    except Exception as e:
        algo.Debug(f"Load persist error: {e}")


def cleanup_object_store(algo):
    """From main_opus._cleanup_object_store."""
    try:
        n = 0
        for i in algo.ObjectStore.GetEnumerator():
            k = i.Key if hasattr(i, 'Key') else str(i)
            if k != "live_state":
                try:
                    algo.ObjectStore.Delete(k)
                    n += 1
                except Exception as e:
                    algo.Debug(f"Error deleting key {k}: {e}")
                    pass
        if n:
            algo.Debug(f"Cleaned {n} keys")
    except Exception as e:
        algo.Debug(f"Cleanup err: {e}")


def live_safety_checks(algo):
    """Extra safety checks for live trading from main_opus."""
    if not algo.LiveMode:
        return True
    
    # Check if we have minimum viable cash
    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except Exception as e:
        algo.Debug(f"Error getting cash from CashBook, using Portfolio.Cash: {e}")
        cash = algo.Portfolio.Cash
    
    if cash < 2.0:
        debug_limited(algo, "LIVE SAFETY: Cash below $2, pausing new entries")
        return False
    
    # Rate limit: don't trade more than once per 5 minutes in live
    if hasattr(algo, '_last_live_trade_time') and algo._last_live_trade_time is not None:
        seconds_since = (algo.Time - algo._last_live_trade_time).total_seconds()
        if seconds_since < 300:
            return False
    
    return True


def kelly_fraction(algo):
    """Kelly criterion-based position sizing from main_opus."""
    if len(algo._rolling_wins) < 10:
        return 1.0
    win_rate = sum(algo._rolling_wins) / len(algo._rolling_wins)
    if win_rate <= 0 or win_rate >= 1:
        return 1.0
    avg_win = np.mean(list(algo._rolling_win_sizes)) if len(algo._rolling_win_sizes) > 0 else 0.02
    avg_loss = np.mean(list(algo._rolling_loss_sizes)) if len(algo._rolling_loss_sizes) > 0 else 0.02
    if avg_loss <= 0:
        return 1.0
    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b
    half_kelly = kelly * 0.5
    return max(0.5, min(1.5, half_kelly / 0.5))


def get_slippage_penalty(algo, symbol):
    """
    Calculate position size multiplier based on historical slippage for a symbol.
    Returns a value between 0.3 and 1.0.
    """
    if not hasattr(algo, '_symbol_slippage_history'):
        return 1.0
    
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    if ticker not in algo._symbol_slippage_history:
        return 1.0
    
    slippage_history = algo._symbol_slippage_history[ticker]
    if len(slippage_history) == 0:
        return 1.0
    
    avg_slippage = sum(slippage_history) / len(slippage_history)
    
    # Apply penalties based on average slippage
    if avg_slippage > 0.010:  # > 1.0%
        return 0.3
    elif avg_slippage > 0.005:  # > 0.5%
        return 0.6
    elif avg_slippage > 0.003:  # > 0.3%
        return 0.8
    else:
        return 1.0


def place_limit_or_market(algo, symbol, quantity, timeout_seconds=60, tag="Entry"):
    """
    Place a limit order at mid-price with fallback to market order after timeout.
    In backtest mode, use market orders directly.
    Returns the ticket from the order placement.
    """
    # In backtest mode, use market orders directly
    if not algo.LiveMode:
        return algo.MarketOrder(symbol, quantity, tag=tag)
    
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        
        # If bid/ask unavailable, use market order immediately
        if bid <= 0 or ask <= 0:
            algo.Debug(f"BID/ASK unavailable for {symbol.Value}, using market order")
            return algo.MarketOrder(symbol, quantity, tag=tag)
        
        # Calculate mid price
        mid = 0.5 * (bid + ask)
        
        # Place limit order at mid price
        limit_ticket = algo.LimitOrder(symbol, quantity, mid, tag=tag)
        
        # Track for fallback in VerifyOrderFills
        if hasattr(algo, '_submitted_orders'):
            algo._submitted_orders[symbol] = {
                'order_id': limit_ticket.OrderId,
                'time': algo.Time,
                'quantity': quantity,  # Store signed quantity
                'is_limit_entry': True,
                'timeout_seconds': timeout_seconds,
                'intent': 'entry'
            }
        
        algo.Debug(f"LIMIT ORDER: {symbol.Value} | qty={quantity} | mid=${mid:.4f} | timeout={timeout_seconds}s")
        return limit_ticket
        
    except Exception as e:
        algo.Debug(f"Error placing limit order for {symbol.Value}: {e}, falling back to market")
        return algo.MarketOrder(symbol, quantity, tag=tag)
