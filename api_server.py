"""
MSNR Trading Bot — Malaysian Support and Resistance
Markets: Crypto (CoinGecko) + Futures NQ/ES (Yahoo Finance)
Strategy: Key Level → Engulfing → CHoCH → FVG Entry | 15× Leverage
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("msnr")

HISTORY_FILE   = 'trade_history.json'
DEMO_BALANCE   = 1_000.0
TRADE_SIZE_USD = 100.0
LEVERAGE       = 15

MS_5M  =  5 * 60 * 1_000
MS_15M = 15 * 60 * 1_000
MS_1H  = 60 * 60 * 1_000
MS_4H  =  4 * 60 * 60 * 1_000
MS_1D  = 24 * 60 * 60 * 1_000

# ── Crypto (CoinGecko) ────────────────────────────────────────────────────────
COINS = {
    'bitcoin':     'BTC',
    'ethereum':    'ETH',
    'binancecoin': 'BNB',
    'solana':      'SOL',
    'ripple':      'XRP',
    'dogecoin':    'DOGE',
    'cardano':     'ADA',
    'avalanche-2': 'AVAX',
}
SYMBOLS  = list(COINS.values())
COIN_IDS = {v: k for k, v in COINS.items()}

# ── Futures (Yahoo Finance) ───────────────────────────────────────────────────
FUTURES = {
    'NQ': 'NQ=F',   # Nasdaq 100 E-mini futures
    'ES': 'ES=F',   # S&P 500 E-mini futures
}
FUTURES_SYMBOLS = list(FUTURES.keys())

ALL_SYMBOLS = SYMBOLS + FUTURES_SYMBOLS

# ── State ─────────────────────────────────────────────────────────────────────
def _blank_state() -> dict:
    return {
        "position": None, "entry": 0.0, "amount": 0.0,
        "sl": 0.0, "tp": 0.0, "liq_price": 0.0,
        "last_price": 0.0, "last_signal": "hold",
        "fvg": None, "score": 0,
        "at_level": False, "key_level": 0.0,
        "engulfing": None, "choch": False, "trend": "range",
    }

coin_state: dict[str, dict] = {sym: _blank_state() for sym in ALL_SYMBOLS}

bot_state = {
    "running":      True,
    "mode":         "demo",
    "demo_balance": DEMO_BALANCE,
}


# ─── DATA LAYER ───────────────────────────────────────────────────────────────

YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}


async def fetch_market_chart(coin_id: str, days: int) -> list:
    """CoinGecko price ticks [[ts_ms, price]] for crypto."""
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days={days}")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url)
        return r.json().get('prices', [])


def to_ohlc(prices: list, interval_ms: int) -> list[dict]:
    """Resample price ticks → OHLC candles."""
    if not prices:
        return []
    candles: dict[int, dict] = {}
    for ts, price in prices:
        b = (int(ts) // interval_ms) * interval_ms
        if b not in candles:
            candles[b] = {'ts': b, 'open': price, 'high': price,
                          'low': price, 'close': price}
        else:
            cv = candles[b]
            if price > cv['high']: cv['high'] = price
            if price < cv['low']:  cv['low']  = price
            cv['close'] = price
    return sorted(candles.values(), key=lambda x: x['ts'])


def resample_ohlc(candles: list[dict], interval_ms: int) -> list[dict]:
    """Resample true OHLC candles to a larger timeframe (preserves H/L correctly)."""
    if not candles:
        return []
    buckets: dict[int, dict] = {}
    for c in candles:
        b = (c['ts'] // interval_ms) * interval_ms
        if b not in buckets:
            buckets[b] = {'ts': b, 'open': c['open'], 'high': c['high'],
                          'low': c['low'], 'close': c['close']}
        else:
            bc = buckets[b]
            if c['high'] > bc['high']: bc['high'] = c['high']
            if c['low']  < bc['low']:  bc['low']  = c['low']
            bc['close'] = c['close']
    return sorted(buckets.values(), key=lambda x: x['ts'])


async def fetch_yahoo_ohlc(yahoo_sym: str, interval: str, range_: str) -> list[dict]:
    """
    True OHLC candles from Yahoo Finance (used for NQ=F / ES=F).
    interval: '1d' | '60m' | '15m' | '5m'
    range_:   '3mo' | '7d' | '1d'
    """
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_sym}"
           f"?interval={interval}&range={range_}&includePrePost=false")
    async with httpx.AsyncClient(timeout=15, headers=YAHOO_HEADERS) as c:
        r = await c.get(url)
        data = r.json()

    result = data['chart']['result'][0]
    timestamps = result.get('timestamp', [])
    q = result['indicators']['quote'][0]
    opens  = q.get('open',  [])
    highs  = q.get('high',  [])
    lows   = q.get('low',   [])
    closes = q.get('close', [])

    candles = []
    for i, ts in enumerate(timestamps):
        c = closes[i] if i < len(closes) else None
        if c is None:
            continue
        candles.append({
            'ts':    ts * 1_000,
            'open':  (opens[i]  or c),
            'high':  (highs[i]  or c),
            'low':   (lows[i]   or c),
            'close': c,
        })
    return candles


async def fetch_yahoo_price(yahoo_sym: str) -> float:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_sym}"
           f"?interval=1m&range=1d")
    async with httpx.AsyncClient(timeout=10, headers=YAHOO_HEADERS) as c:
        r = await c.get(url)
        data = r.json()
    return float(data['chart']['result'][0]['meta']['regularMarketPrice'])


async def fetch_all_prices() -> dict[str, float]:
    """Batch-fetch all crypto prices in one CoinGecko call."""
    ids = ','.join(COINS.keys())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url)
        data = r.json()
    return {sym: data[cid]['usd'] for cid, sym in COINS.items()
            if cid in data and 'usd' in data[cid]}


async def fetch_futures_prices() -> dict[str, float]:
    """Fetch NQ and ES prices from Yahoo Finance (parallel)."""
    async def _get(sym: str, yahoo: str) -> tuple[str, float]:
        try:
            p = await fetch_yahoo_price(yahoo)
            return sym, p
        except Exception as e:
            log.warning(f"futures price {sym}: {e}")
            return sym, 0.0
    results = await asyncio.gather(*[_get(s, y) for s, y in FUTURES.items()])
    return {sym: p for sym, p in results if p > 0}


# ─── MSNR ANALYSIS ENGINES ────────────────────────────────────────────────────

def detect_swing_levels(candles: list[dict], lookback: int = 3) -> tuple[list[float], list[float]]:
    resistances, supports = [], []
    n = len(candles)
    for i in range(lookback, n - lookback):
        h = candles[i]['high']
        l = candles[i]['low']
        if all(h >= candles[j]['high'] for j in range(i - lookback, i + lookback + 1) if j != i):
            resistances.append(h)
        if all(l <= candles[j]['low']  for j in range(i - lookback, i + lookback + 1) if j != i):
            supports.append(l)
    return resistances, supports


def cluster_levels(levels: list[float], tol: float = 0.006) -> list[float]:
    clustered, seen = [], []
    for lvl in sorted(levels):
        if not any(abs(lvl - s) / max(s, 1) < tol for s in seen):
            clustered.append(lvl)
            seen.append(lvl)
    return clustered


def nearest_level(price: float, levels: list[float], tol: float = 0.018) -> tuple[bool, float]:
    best_dist, best_lvl = float('inf'), 0.0
    for lvl in levels:
        dist = abs(price - lvl) / max(lvl, 1)
        if dist < tol and dist < best_dist:
            best_dist, best_lvl = dist, lvl
    return (best_lvl > 0), best_lvl


def detect_engulfing(candles: list[dict]) -> str | None:
    if len(candles) < 2:
        return None
    prev, curr = candles[-2], candles[-1]
    if abs(prev['close'] - prev['open']) < 1e-9:
        return None
    curr_body = abs(curr['close'] - curr['open'])
    prev_body = abs(prev['close'] - prev['open'])
    if (curr['close'] > curr['open'] and prev['close'] < prev['open'] and
            curr['open'] <= prev['close'] and curr['close'] >= prev['open'] and
            curr_body > prev_body * 0.8):
        return 'bullish'
    if (curr['close'] < curr['open'] and prev['close'] > prev['open'] and
            curr['open'] >= prev['close'] and curr['close'] <= prev['open'] and
            curr_body > prev_body * 0.8):
        return 'bearish'
    return None


def detect_choch(candles: list[dict]) -> bool:
    """Bullish CHoCH: last close breaks above a recent swing high."""
    if len(candles) < 8:
        return False
    recent = candles[-14:]
    close  = recent[-1]['close']
    swing_highs = [
        recent[i]['high']
        for i in range(2, len(recent) - 2)
        if (recent[i]['high'] >= recent[i-1]['high'] and
            recent[i]['high'] >= recent[i+1]['high'] and
            recent[i]['high'] >= recent[i-2]['high'])
    ]
    return bool(swing_highs) and close > swing_highs[-1]


def detect_fvg(candles: list[dict]) -> dict | None:
    for i in range(len(candles) - 1, 2, -1):
        c1, c3 = candles[i - 2], candles[i]
        if c3['low'] > c1['high']:
            return {"type": "bullish", "low": c1['high'], "high": c3['low'],
                    "mid": (c1['high'] + c3['low']) / 2}
        if c3['high'] < c1['low']:
            return {"type": "bearish", "low": c3['high'], "high": c1['low'],
                    "mid": (c3['high'] + c1['low']) / 2}
    return None


def get_trend_1h(candles: list[dict]) -> str:
    if len(candles) < 4:
        return "range"
    highs = [c['high'] for c in candles[-8:]]
    lows  = [c['low']  for c in candles[-8:]]
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]: return "bullish"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]: return "bearish"
    return "range"


# ─── MSNR SIGNAL ─────────────────────────────────────────────────────────────

def generate_signal_msnr(at_level: bool, key_level: float,
                          engulfing: str | None, choch: bool,
                          fvg: dict | None, price: float,
                          trend: str) -> tuple[str, int]:
    """
    MSNR Score /10, threshold 6:
      +3  At key S/R level (RBS zone)
      +2  Bullish engulfing candle
      +2  CHoCH on 1H
      +2  Inside bullish FVG  (+1 if FVG nearby but not inside)
      +1  1H trend bullish
    """
    score = 0
    if at_level:                  score += 3
    if engulfing == 'bullish':    score += 2
    elif engulfing == 'bearish':  score -= 1
    if choch:                     score += 2
    if fvg and fvg['type'] == 'bullish':
        score += 2 if fvg['low'] <= price <= fvg['high'] else 1
    if trend == 'bullish':        score += 1
    elif trend == 'bearish':      score -= 1
    return ("buy", score) if score >= 6 else ("hold", score)


# ─── RISK MANAGEMENT ─────────────────────────────────────────────────────────

def calc_sl_tp_msnr(price: float, key_level: float, all_levels: list[float],
                    engulf_low: float = 0.0) -> tuple[float, float]:
    """SL below key level; TP at next resistance; min 1:2.5 RR."""
    if key_level > 0 and key_level < price:
        sl = key_level * 0.997
    elif engulf_low > 0:
        sl = engulf_low * 0.999
    else:
        sl = price * 0.98

    risk   = price - sl
    min_tp = price + risk * 2.5
    next_res = sorted(lvl for lvl in all_levels if lvl > price * 1.005)
    tp = max(next_res[0], min_tp) if next_res else min_tp
    return sl, tp


# ─── DEMO EXECUTION ──────────────────────────────────────────────────────────

def demo_tick(sym: str, price: float, signal: str, sl: float, tp: float, score: int,
              trend: str, at_level: bool, fvg, choch: bool,
              engulfing: str | None, key_level: float):
    cs = coin_state[sym]
    cs.update({"last_price": price, "last_signal": signal, "score": score,
                "trend": trend, "at_level": at_level, "key_level": key_level,
                "engulfing": engulfing, "choch": choch, "fvg": fvg})

    if not bot_state["running"]:
        return

    if cs["position"] == "long":
        if cs["liq_price"] > 0 and price <= cs["liq_price"]:
            _close(sym, price, "liquidated")
        elif price <= cs["sl"]:
            _close(sym, price, "stop-loss")
        elif price >= cs["tp"]:
            _close(sym, price, "take-profit")
        elif signal == "hold" and price < cs["entry"] * 0.985:
            _close(sym, price, "signal-exit")
        return

    if signal == "buy" and cs["position"] is None and bot_state["demo_balance"] >= TRADE_SIZE_USD:
        notional  = TRADE_SIZE_USD * LEVERAGE
        amount    = notional / price
        liq_price = price * (1 - 1 / LEVERAGE + 0.001)
        cs.update({"amount": amount, "entry": price, "position": "long",
                   "sl": sl, "tp": tp, "liq_price": liq_price})
        bot_state["demo_balance"] -= TRADE_SIZE_USD
        log.info(f"[BUY 15×] {sym} @ ${price:,.2f}  notional=${notional:.0f}  "
                 f"SL=${sl:.2f}  TP=${tp:.2f}  LIQ=${liq_price:.2f}  score={score}/10")


def _close(sym: str, price: float, reason: str):
    cs = coin_state[sym]
    profit_pct = (price - cs["entry"]) / cs["entry"]
    cost_usd   = cs["amount"] * cs["entry"]
    profit_usd = cs["amount"] * (price - cs["entry"])
    bot_state["demo_balance"] += cs["amount"] * price
    h = load_history()
    h.append({
        "time": datetime.utcnow().isoformat(), "symbol": sym, "reason": reason,
        "entry": round(cs["entry"], 6), "exit": round(price, 6),
        "sl": round(cs["sl"], 6), "tp": round(cs["tp"], 6),
        "cost_usd": round(cost_usd, 2), "profit_usd": round(profit_usd, 2),
        "profit": round(profit_pct, 6),
        "balance": round(bot_state["demo_balance"], 2),
        "score": cs.get("score", 0), "leverage": LEVERAGE,
    })
    save_history(h)
    cs.update({"position": None, "entry": 0.0, "amount": 0.0,
               "sl": 0.0, "tp": 0.0, "liq_price": 0.0})
    log.info(f"[{'✅' if profit_pct>0 else '❌'}] {sym} {reason}  "
             f"{profit_pct:+.2%}  balance=${bot_state['demo_balance']:.2f}")


# ─── ANALYSIS ────────────────────────────────────────────────────────────────

async def _run_msnr(sym: str, tf_1d: list[dict], tf_4h: list[dict],
                    tf_1h: list[dict], tf_15m: list[dict], tf_5m: list[dict]):
    """Shared MSNR logic — called by both crypto and futures analyzers."""
    if not tf_1d or not tf_4h or not tf_1h:
        return

    price = coin_state[sym]["last_price"]
    if price == 0.0:
        return

    # 1. HTF key S/R levels
    res_1d, sup_1d = detect_swing_levels(tf_1d,  lookback=2)
    res_4h, sup_4h = detect_swing_levels(tf_4h,  lookback=3)
    all_levels = cluster_levels(res_1d + res_4h + sup_1d + sup_4h)

    # 2. At level?
    at_level, key_level = nearest_level(price, all_levels)

    # 3. 1H trend
    trend = get_trend_1h(tf_1h)

    # 4. CHoCH on 1H
    choch = detect_choch(tf_1h)

    # 5. Engulfing on 15M
    engulfing = detect_engulfing(tf_15m[-5:]) if tf_15m else None

    # 6. FVG: 15M → 5M
    fvg = ((detect_fvg(tf_15m[-20:]) if tf_15m else None) or
           (detect_fvg(tf_5m[-20:])  if tf_5m  else None))

    # 7. Signal
    signal, score = generate_signal_msnr(at_level, key_level,
                                          engulfing, choch, fvg, price, trend)

    # 8. SL / TP
    engulf_low = tf_15m[-1]['low'] if (engulfing == 'bullish' and tf_15m) else 0.0
    sl, tp = calc_sl_tp_msnr(price, key_level, all_levels, engulf_low)

    # 9. Execute
    demo_tick(sym, price, signal, sl, tp, score,
              trend, at_level, fvg, choch, engulfing, key_level)

    log.info(f"{sym} ${price:,.2f} | level={'✓' if at_level else '✗'}@{key_level:.2f} "
             f"trend={trend} choch={choch} engulf={engulfing} "
             f"fvg={'✓' if fvg else '✗'} | score={score}/10 → {signal}")


async def analyze_crypto(sym: str):
    """Crypto MSNR: CoinGecko market_chart → resample → MSNR."""
    coin_id = COIN_IDS[sym]
    try:
        chart_30d, chart_7d, chart_1d = await asyncio.gather(
            fetch_market_chart(coin_id, 30),
            fetch_market_chart(coin_id, 7),
            fetch_market_chart(coin_id, 1),
        )
        tf_1d  = to_ohlc(chart_30d, MS_1D)
        tf_4h  = to_ohlc(chart_30d, MS_4H)
        tf_1h  = to_ohlc(chart_7d,  MS_1H)
        tf_15m = to_ohlc(chart_1d,  MS_15M)
        tf_5m  = to_ohlc(chart_1d,  MS_5M)
        await _run_msnr(sym, tf_1d, tf_4h, tf_1h, tf_15m, tf_5m)
    except Exception as e:
        log.warning(f"analyze_crypto {sym}: {e}")


async def analyze_futures(sym: str):
    """Futures MSNR: Yahoo Finance true OHLC → MSNR."""
    yahoo = FUTURES[sym]
    try:
        # 3 concurrent Yahoo calls
        tf_daily, tf_1h_raw, tf_15m = await asyncio.gather(
            fetch_yahoo_ohlc(yahoo, '1d',  '3mo'),   # daily  → 1D bias
            fetch_yahoo_ohlc(yahoo, '60m', '7d'),    # hourly → 1H + resample to 4H
            fetch_yahoo_ohlc(yahoo, '15m', '1d'),    # 15M    → entry + FVG
        )
        tf_4h  = resample_ohlc(tf_1h_raw, MS_4H)
        tf_5m  = resample_ohlc(tf_15m,    MS_5M) if tf_15m else []

        # Update price from latest candle if price_loop hasn't set it
        if coin_state[sym]["last_price"] == 0.0 and tf_daily:
            coin_state[sym]["last_price"] = tf_daily[-1]['close']

        await _run_msnr(sym, tf_daily, tf_4h, tf_1h_raw, tf_15m, tf_5m)
    except Exception as e:
        log.warning(f"analyze_futures {sym}: {e}")


async def analyze_coin(sym: str):
    if sym in FUTURES_SYMBOLS:
        await analyze_futures(sym)
    else:
        await analyze_crypto(sym)


# ─── LOOPS ───────────────────────────────────────────────────────────────────

async def price_loop():
    """Batch-update crypto + futures prices every 30s."""
    log.info("Price loop started  (crypto: CoinGecko | futures: Yahoo Finance)")
    while True:
        try:
            crypto_p, futures_p = await asyncio.gather(
                fetch_all_prices(),
                fetch_futures_prices(),
            )
            for sym, p in {**crypto_p, **futures_p}.items():
                if p > 0:
                    coin_state[sym]["last_price"] = p
            log.info(
                "Crypto: " + " ".join(f"{s}=${p:,.0f}" for s, p in crypto_p.items()) + "  |  "
                "Futures: " + " ".join(f"{s}=${p:,.0f}" for s, p in futures_p.items())
            )
        except Exception as e:
            log.warning(f"price_loop: {e}")
        await asyncio.sleep(30)


async def trading_loop():
    """MSNR analysis — all 10 symbols (8 crypto + NQ + ES), 15s spacing."""
    log.info("MSNR Trading loop started  [Crypto × 8 + NQ + ES]")
    try:
        cp, fp = await asyncio.gather(fetch_all_prices(), fetch_futures_prices())
        for sym, p in {**cp, **fp}.items():
            if p > 0: coin_state[sym]["last_price"] = p
    except Exception as e:
        log.warning(f"Initial price fetch: {e}")

    while True:
        for sym in ALL_SYMBOLS:
            await analyze_coin(sym)
            await asyncio.sleep(15)
        await asyncio.sleep(30)


# ─── APP ─────────────────────────────────────────────────────────────────────

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(h: list):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(h, f, indent=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t1 = asyncio.create_task(price_loop())
    t2 = asyncio.create_task(trading_loop())
    yield
    t1.cancel(); t2.cancel()


app = FastAPI(title="MSNR Trading Bot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/status")
def status():
    h    = load_history()
    wins = [t for t in h if t.get("profit", 0) > 0]
    return {
        **bot_state,
        "open_positions": [s for s, cs in coin_state.items() if cs["position"]],
        "total_trades":   len(h),
        "win_rate":       round(len(wins) / len(h), 3) if h else 0,
        "demo_pnl":       round(bot_state["demo_balance"] - DEMO_BALANCE, 2),
        "total_profit":   round(sum(t.get("profit_usd", 0) for t in h), 2),
        "leverage":       LEVERAGE,
        "trade_size":     TRADE_SIZE_USD,
        "strategy":       "MSNR",
        "markets":        f"Crypto ×{len(SYMBOLS)} + Futures ×{len(FUTURES_SYMBOLS)}",
    }


@app.post("/start")
def start():
    bot_state["running"] = True
    return {"status": "started"}


@app.post("/stop")
def stop():
    bot_state["running"] = False
    return {"status": "stopped"}


@app.post("/reset-demo")
def reset_demo():
    bot_state.update({"demo_balance": DEMO_BALANCE, "running": True})
    for cs in coin_state.values():
        cs.update({"position": None, "entry": 0.0, "amount": 0.0,
                   "sl": 0.0, "tp": 0.0, "liq_price": 0.0})
    save_history([])
    return {"status": "reset"}


@app.get("/positions")
def positions_endpoint():
    result = []
    for sym, cs in coin_state.items():
        if cs["position"] == "long":
            price = cs["last_price"]
            entry = cs["entry"]
            amt   = cs["amount"]
            pnl   = amt * (price - entry)
            pct   = (price - entry) / entry if entry else 0
            liq   = cs["liq_price"]
            result.append({
                "symbol":        sym,
                "market":        "futures" if sym in FUTURES_SYMBOLS else "crypto",
                "entry":         round(entry,             4),
                "current_price": round(price,             4),
                "margin_used":   round(TRADE_SIZE_USD,    2),
                "notional":      round(TRADE_SIZE_USD * LEVERAGE, 2),
                "current_value": round(amt * price,       2),
                "pnl_usd":       round(pnl,               2),
                "pnl_pct":       round(pct * 100,         2),
                "sl":            round(cs["sl"],           4),
                "tp":            round(cs["tp"],           4),
                "liq_price":     round(liq,                4),
                "dist_liq_pct":  round((price - liq) / price * 100 if liq else 0, 2),
                "score":         cs["score"],
                "leverage":      LEVERAGE,
                "key_level":     round(cs["key_level"],   4),
            })
    return result


@app.get("/coins")
def coins_endpoint():
    return [
        {
            "symbol":    s,
            "market":    "futures" if s in FUTURES_SYMBOLS else "crypto",
            "price":     coin_state[s]["last_price"],
            "signal":    coin_state[s]["last_signal"],
            "position":  coin_state[s]["position"],
            "entry":     coin_state[s]["entry"],
            "sl":        coin_state[s]["sl"],
            "tp":        coin_state[s]["tp"],
            "score":     coin_state[s]["score"],
            "trend":     coin_state[s]["trend"],
            "at_level":  coin_state[s]["at_level"],
            "key_level": coin_state[s]["key_level"],
            "engulfing": coin_state[s]["engulfing"],
            "choch":     coin_state[s]["choch"],
            "fvg":       coin_state[s]["fvg"] is not None,
        }
        for s in ALL_SYMBOLS
    ]


@app.get("/history")
def history_endpoint():
    return load_history()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if os.path.exists("dashboard.html"):
        return open("dashboard.html").read()
    return HTMLResponse("<h1>MSNR Bot running</h1>")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
