# region imports
from AlgorithmImports import *
from collections import deque
import statistics
import json
# endregion

# =============================================================================
# CONSTANTS
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

# Trading parameters
KRAKEN_MAX_FEE_RATE = 0.006
MIN_LIVE_CASH_USD = 2.0
LIVE_TRADE_RATE_LIMIT_SECONDS = 300
MIN_KELLY_SAMPLE_SIZE = 10
HALF_KELLY_DIVISOR = 0.5
MIN_KELLY_MULTIPLIER = 0.5
MAX_KELLY_MULTIPLIER = 1.5

# Slippage model parameters
LOW_PRICE_THRESHOLD = 0.01
MID_PRICE_THRESHOLD = 0.10
HIGH_PRICE_THRESHOLD = 1.0
LOW_PRICE_SLIPPAGE_MULT = 3.0
MID_PRICE_SLIPPAGE_MULT = 2.0
HIGH_PRICE_SLIPPAGE_MULT = 1.5

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
# EXECUTION FUNCTIONS
# =============================================================================

def smart_liquidate(algo, symbol, tag="Liquidate"):
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        algo.Debug(symbol.Value+"o")
        return
    if symbol in algo._cancel_cooldowns and algo.Time < algo._cancel_cooldowns[symbol]:
        algo.Debug(symbol.Value+"c")
        return
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        algo.Debug(symbol.Value+"p")
        return
    holding_qty = algo.Portfolio[symbol].Quantity
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price * abs(holding_qty) < min_notional * 0.9:
        algo.Debug(symbol.Value+"n")
        return
    if algo.LiveMode and holding_qty > 0:
        estimated_fee = price * abs(holding_qty) * KRAKEN_MAX_FEE_RATE
        try:
            available_usd = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_usd = algo.Portfolio.Cash
        if available_usd < estimated_fee:
            algo.Debug(symbol.Value+"f")
            if symbol not in algo.entry_prices:
                algo.entry_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.highest_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.entry_times[symbol] = algo.Time
            return
    algo.Transactions.CancelOpenOrders(symbol)
    if abs(holding_qty) < min_qty:
        algo.Debug(symbol.Value+"q")
        return
    safe_qty = round_quantity(algo, symbol, abs(holding_qty))
    if safe_qty < min_qty:
        algo.Debug(symbol.Value+"r")
        return
    if safe_qty > 0:
        direction_mult = -1 if holding_qty > 0 else 1
        algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)

def cancel_stale_orders(algo):
    try:
        open_orders = algo.Transactions.GetOpenOrders()
        if len(open_orders) > 0:
            for order in open_orders:
                algo.Transactions.CancelOrder(order.Id)
    except Exception as e:
        algo.Debug(f"Error canceling stale orders: {e}")

def cancel_stale_new_orders(algo):
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
                algo.Transactions.CancelOrder(order.Id)
                algo._cancel_cooldowns[order.Symbol] = algo.Time + timedelta(minutes=algo.cancel_cooldown_minutes)
                algo._session_blacklist.add(sym_val)
    except Exception as e:
        algo.Debug(f"Error in cancel_stale_new_orders: {e}")

def effective_stale_timeout(algo):
    return algo.live_stale_order_timeout_seconds if algo.LiveMode else algo.stale_order_timeout_seconds

def get_min_quantity(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    try:
        if symbol in algo.Securities:
            sec = algo.Securities[symbol]
            if hasattr(sec, 'SymbolProperties') and sec.SymbolProperties is not None:
                min_size = sec.SymbolProperties.MinimumOrderSize
                if min_size is not None and min_size > 0:
                    return float(min_size)
    except:
        pass
    if ticker in KRAKEN_MIN_QTY_FALLBACK:
        return KRAKEN_MIN_QTY_FALLBACK[ticker]
    return estimate_min_qty(algo, symbol)

def estimate_min_qty(algo, symbol):
    try:
        price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
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

def get_min_notional_usd(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    fallback = MIN_NOTIONAL_FALLBACK.get(ticker, algo.min_notional)
    try:
        price = algo.Securities[symbol].Price
        min_qty = get_min_quantity(algo, symbol)
        implied = price * min_qty if price > 0 else fallback
        return max(fallback, implied, algo.min_notional)
    except:
        return max(fallback, algo.min_notional)

def round_quantity(algo, symbol, quantity):
    try:
        lot_size = algo.Securities[symbol].SymbolProperties.LotSize
        if lot_size is not None and lot_size > 0:
            return float(int(quantity / lot_size) * lot_size)
        return quantity
    except:
        return quantity

def has_open_orders(algo, symbol=None):
    if symbol is None:
        return len(algo.Transactions.GetOpenOrders()) > 0
    return len(algo.Transactions.GetOpenOrders(symbol)) > 0

def spread_ok(algo, symbol):
    sp = get_spread_pct(algo, symbol)
    if sp is None:
        return not algo.LiveMode
    effective_spread_cap = algo.max_spread_pct
    if algo.LiveMode and (algo.volatility_regime == "high" or algo.market_regime == "sideways"):
        effective_spread_cap = min(effective_spread_cap, 0.015)
    if sp > effective_spread_cap:
        return False
    crypto = algo.crypto_data.get(symbol)
    if crypto and len(crypto.get('spreads', [])) >= 4:
        median_sp = statistics.median(list(crypto['spreads']))
        if median_sp > 0 and sp > algo.spread_widen_mult * median_sp:
            return False
    return True

def get_spread_pct(algo, symbol):
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid > 0 and ask > 0 and ask >= bid:
            mid = 0.5 * (bid + ask)
            if mid > 0:
                return (ask - bid) / mid
    except:
        pass
    return None

def slip_log(algo, symbol, direction, fill_price):
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
        algo._slip_abs.append(abs(slip))
        if algo.LiveMode and abs(slip) > algo.slip_outlier_threshold:
            algo.Debug(f"⚠️ HIGH SLIPPAGE: {symbol.Value} | {abs(slip):.4%} | dir={direction}")
    except:
        pass

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

def live_safety_checks(algo):
    if not algo.LiveMode:
        return True
    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except:
        cash = algo.Portfolio.Cash
    if cash < MIN_LIVE_CASH_USD:
        debug_limited(algo, f"LIVE SAFETY: Cash below ${MIN_LIVE_CASH_USD}, pausing new entries")
        return False
    if hasattr(algo, '_last_live_trade_time') and algo._last_live_trade_time is not None:
        seconds_since = (algo.Time - algo._last_live_trade_time).total_seconds()
        if seconds_since < LIVE_TRADE_RATE_LIMIT_SECONDS:
            return False
    return True

def cleanup_object_store(algo):
    try:
        n = 0
        for i in algo.ObjectStore.GetEnumerator():
            k = i.Key if hasattr(i, 'Key') else str(i)
            if k != "opus_live_state":
                try:
                    algo.ObjectStore.Delete(k)
                    n += 1
                except:
                    pass
        if n:
            algo.Debug(f"Cleaned {n} keys")
    except Exception as e:
        algo.Debug(f"Cleanup err: {e}")

def persist_state(algo):
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
        algo.ObjectStore.Save("opus_live_state", json.dumps(state))
    except Exception as e:
        algo.Debug(f"Persist error: {e}")

def load_persisted_state(algo):
    try:
        if algo.LiveMode and algo.ObjectStore.ContainsKey("opus_live_state"):
            raw = algo.ObjectStore.Read("opus_live_state")
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
    except Exception as e:
        algo.Debug(f"Load persist error: {e}")

def sync_existing_positions(algo):
    algo.Debug("=" * 50)
    algo.Debug("=== SYNCING EXISTING POSITIONS ===")
    synced_count = 0
    positions_to_close = []
    for symbol in algo.Portfolio.Keys:
        holding = algo.Portfolio[symbol]
        if not holding.Invested or holding.Quantity == 0:
            continue
        if symbol in algo.entry_prices:
            continue
        if symbol not in algo.Securities:
            try:
                algo.AddCrypto(symbol.Value, Resolution.Hour, Market.Kraken)
            except:
                continue
        algo.entry_prices[symbol] = holding.AveragePrice
        algo.highest_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol] = algo.Time
        synced_count += 1
        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
        pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
        if current_price > holding.AveragePrice:
            algo.highest_prices[symbol] = current_price
        if pnl_pct >= algo.base_take_profit:
            positions_to_close.append((symbol, symbol.Value, pnl_pct, "Sync TP"))
        elif pnl_pct <= -algo.base_stop_loss:
            positions_to_close.append((symbol, symbol.Value, pnl_pct, "Sync SL"))
    algo.Debug(f"Synced {synced_count} positions")
    algo.Debug(f"Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug("=" * 50)
    for symbol, ticker, pnl_pct, reason in positions_to_close:
        smart_liquidate(algo, symbol, reason)

def resync_holdings(algo):
    if algo.IsWarmingUp:
        return
    if not algo.LiveMode:
        return
    missing = []
    for symbol in algo.Portfolio.Keys:
        holding = algo.Portfolio[symbol]
        if not holding.Invested or holding.Quantity == 0:
            continue
        if symbol in algo.entry_prices:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        if has_open_orders(algo, symbol):
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
        except Exception as e:
            algo.Debug(f"Resync error {symbol.Value}: {e}")

def cleanup_position(algo, symbol):
    algo.entry_prices.pop(symbol, None)
    algo.highest_prices.pop(symbol, None)
    algo.entry_times.pop(symbol, None)
    if symbol in algo.crypto_data:
        algo.crypto_data[symbol]['trail_stop'] = None

def kelly_fraction(algo):
    if len(algo._rolling_wins) < MIN_KELLY_SAMPLE_SIZE:
        return 1.0
    win_rate = sum(algo._rolling_wins) / len(algo._rolling_wins)
    if win_rate <= 0 or win_rate >= 1:
        return 1.0
    avg_win = statistics.mean(list(algo._rolling_win_sizes)) if len(algo._rolling_win_sizes) > 0 else 0.02
    avg_loss = statistics.mean(list(algo._rolling_loss_sizes)) if len(algo._rolling_loss_sizes) > 0 else 0.02
    if avg_loss <= 0:
        return 1.0
    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b
    half_kelly = kelly * HALF_KELLY_DIVISOR
    return max(MIN_KELLY_MULTIPLIER, min(MAX_KELLY_MULTIPLIER, half_kelly / HALF_KELLY_DIVISOR))

def debug_limited(algo, msg):
    if "CANCELED" in msg or "ZOMBIE" in msg or "INVALID" in msg:
        algo.Debug(msg)
        return
    if algo.log_budget > 0:
        algo.Debug(msg)
        algo.log_budget -= 1
    elif algo.LiveMode:
        algo.Debug(msg)
