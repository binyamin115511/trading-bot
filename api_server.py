"""
MSNR Trading Bot — Malaysian Support and Resistance
Strategy: Key Level Detection → RBS/SBR Zone → Engulfing Confirmation → CHoCH + FVG Entry
Timeframes: 1D · 4H · 1H · 15M · 5M
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

HISTORY_FILE  = 'trade_history.json'
DEMO_BALANCE  = 1_000.0
TRADE_SIZE_USD = 100.0   # $100 מרג'ין קבוע
LEVERAGE       = 15       # 15× → פוזיציה $1,500

# ─── CANDLE INTERVAL CONSTANTS (milliseconds) ────────────────────────────────
MS_5M  =  5 * 60 * 1_000
MS_15M = 15 * 60 * 1_000
MS_30M = 30 * 60 * 1_000
MS_1H  = 60 * 60 * 1_000
MS_4H  =  4 * 60 * 60 * 1_000
MS_1D  = 24 * 60 * 60 * 1_000

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

coin_state: dict[str, dict] = {
    sym: {
        "position":   None,  "entry":    0.0, "amount":    0.0,
        "sl":         0.0,   "tp":       0.0, "liq_price": 0.0,
        "last_price": 0.0,   "last_signal": "hold",
        "fvg":        None,  "score":    0,
        # MSNR specific
        "at_level":   False, "key_level": 0.0,
        "engulfing":  None,  # 'bullish' | 'bearish' | None
        "choch":      False,
        "trend":      "range",
    }
    for sym in SYMBOLS
}

bot_state = {
    "running":      True,
    "mode":         "demo",
    "demo_balance": DEMO_BALANCE,
}


# ─── DATA LAYER ───────────────────────────────────────────────────────────────

async def fetch_market_chart(coin_id: str, days: int) -> list:
    """
    CoinGecko market_chart → [[ts_ms, price], ...].
      days=1   → ~5-min  intervals (~288 pts)
      days≤90  → hourly  intervals
    """
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days={days}")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        data = r.json()
    return data.get('prices', [])


def to_ohlc(prices: list, interval_ms: int) -> list[dict]:
    """Resample [[ts_ms, price]] ticks → OHLC candles."""
    if not prices:
        return []
    candles: dict[int, dict] = {}
    for ts, price in prices:
        b = (int(ts) // interval_ms) * interval_ms
        if b not in candles:
            candles[b] = {'ts': b, 'open': price, 'high': price, 'low': price, 'close': price}
        else:
            c = candles[b]
            if price > c['high']: c['high'] = price
            if price < c['low']:  c['low']  = price
            c['close'] = price
    return sorted(candles.values(), key=lambda x: x['ts'])


async def fetch_price(coin_id: str) -> float:
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        return r.json()[coin_id]['usd']


async def fetch_all_prices() -> dict[str, float]:
    ids = ','.join(COINS.keys())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        data = r.json()
    return {sym: data[cid]['usd'] for cid, sym in COINS.items()
            if cid in data and 'usd' in data[cid]}


# ─── MSNR ENGINES ─────────────────────────────────────────────────────────────

def detect_swing_levels(candles: list[dict], lookback: int = 3) -> tuple[list[float], list[float]]:
    """
    Identify swing highs (resistance) and swing lows (support) — the MSNR key levels.
    Returns (resistances, supports).
    """
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
    """Merge nearby levels (within tol%) into one — keeps only significant levels."""
    if not levels:
        return []
    clustered, seen = [], []
    for lvl in sorted(levels):
        if not any(abs(lvl - s) / max(s, 1) < tol for s in seen):
            clustered.append(lvl)
            seen.append(lvl)
    return clustered


def nearest_level(price: float, levels: list[float], tol: float = 0.018) -> tuple[bool, float]:
    """Returns (at_level, level) if price is within tol% of any key level."""
    best_dist, best_lvl = float('inf'), 0.0
    for lvl in levels:
        dist = abs(price - lvl) / max(lvl, 1)
        if dist < tol and dist < best_dist:
            best_dist = dist
            best_lvl  = lvl
    return (best_lvl > 0), best_lvl


def detect_engulfing(candles: list[dict]) -> str | None:
    """
    Detect engulfing candle pattern on the last 2 candles.
    Returns 'bullish', 'bearish', or None.
    """
    if len(candles) < 2:
        return None
    prev, curr = candles[-2], candles[-1]
    prev_body = abs(prev['close'] - prev['open'])
    curr_body = abs(curr['close'] - curr['open'])
    if prev_body < 1e-9:
        return None
    # Bullish engulfing: curr fully engulfs prev bearish candle
    if (curr['close'] > curr['open'] and
        prev['close'] < prev['open'] and
        curr['open']  <= prev['close'] and
        curr['close'] >= prev['open']  and
        curr_body > prev_body * 0.8):
        return 'bullish'
    # Bearish engulfing
    if (curr['close'] < curr['open'] and
        prev['close'] > prev['open'] and
        curr['open']  >= prev['close'] and
        curr['close'] <= prev['open']  and
        curr_body > prev_body * 0.8):
        return 'bearish'
    return None


def detect_choch(candles: list[dict]) -> bool:
    """
    Bullish Change of Character (CHoCH):
    In a downtrend context, the last close breaks above the most recent
    swing high → market structure shifts bullish.
    """
    if len(candles) < 8:
        return False
    recent = candles[-14:]
    close  = recent[-1]['close']
    # Find swing highs in the middle section (not the last 2 candles)
    swing_highs = []
    for i in range(2, len(recent) - 2):
        if (recent[i]['high'] >= recent[i-1]['high'] and
            recent[i]['high'] >= recent[i+1]['high'] and
            recent[i]['high'] >= recent[i-2]['high']):
            swing_highs.append(recent[i]['high'])
    if not swing_highs:
        return False
    # CHoCH: current close breaks above the last swing high
    return close > swing_highs[-1]


def detect_fvg(candles: list[dict]) -> dict | None:
    """Fair Value Gap — unfilled price imbalance area."""
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
    """1H trend: bullish (HH+HL), bearish (LH+LL), range."""
    if len(candles) < 4:
        return "range"
    highs = [c['high'] for c in candles[-8:]]
    lows  = [c['low']  for c in candles[-8:]]
    hh = highs[-1] > highs[-2]
    hl = lows[-1]  > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1]  < lows[-2]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "range"


# ─── MSNR SIGNAL ENGINE ───────────────────────────────────────────────────────

def generate_signal_msnr(
    at_level:  bool,
    key_level: float,
    engulfing: str | None,
    choch:     bool,
    fvg:       dict | None,
    price:     float,
    trend:     str,
) -> tuple[str, int]:
    """
    MSNR Score (max 10), threshold 6 → LONG entry:

      At Key Level (RBS/Support zone):
        +3  Price at HTF swing high/low level — the core MSNR zone

      Candle Confirmation:
        +2  Bullish engulfing at the level

      Structure:
        +2  CHoCH on 1H (Change of Character — structure shifts bullish)

      Entry Precision:
        +2  Price inside bullish FVG (15M/5M)
        +1  Bullish FVG exists nearby (not in it)

      Alignment:
        +1  1H trend bullish

    Bearish signals reduce score (prevents longs in downtrends).
    """
    score = 0

    # ── Key Level (RBS zone) ──
    if at_level:
        score += 3

    # ── Engulfing ──
    if engulfing == 'bullish':
        score += 2
    elif engulfing == 'bearish':
        score -= 1

    # ── CHoCH ──
    if choch:
        score += 2

    # ── FVG ──
    if fvg and fvg['type'] == 'bullish':
        if fvg['low'] <= price <= fvg['high']:
            score += 2   # inside FVG
        else:
            score += 1   # FVG nearby

    # ── Trend alignment ──
    if trend == 'bullish':
        score += 1
    elif trend == 'bearish':
        score -= 1

    return ("buy", score) if score >= 6 else ("hold", score)


# ─── RISK MANAGEMENT ──────────────────────────────────────────────────────────

def calc_sl_tp_msnr(price: float, key_level: float, all_levels: list[float],
                    engulf_low: float = 0.0) -> tuple[float, float]:
    """
    MSNR Risk Management:
      SL: just below the key support level (0.3% buffer), or below engulfing candle low
      TP: next resistance level above price — minimum 1:2.5 RR
    """
    # SL placement
    if key_level > 0 and key_level < price:
        sl = key_level * 0.997   # 0.3% below the key level
    elif engulf_low > 0:
        sl = engulf_low * 0.999  # just below engulfing candle low
    else:
        sl = price * 0.98

    risk   = price - sl
    min_tp = price + risk * 2.5  # minimum 1:2.5 RR

    # TP at next resistance level above price
    next_res = sorted([lvl for lvl in all_levels if lvl > price * 1.005])
    if next_res:
        tp = next_res[0]
        if tp < min_tp:
            tp = min_tp
    else:
        tp = min_tp

    return sl, tp


# ─── DEMO EXECUTION ───────────────────────────────────────────────────────────

def demo_tick(sym: str, price: float, signal: str, sl: float, tp: float, score: int,
              trend: str, at_level: bool, fvg, choch: bool,
              engulfing: str | None, key_level: float):
    cs = coin_state[sym]
    cs.update({
        "last_price":  price,
        "last_signal": signal,
        "score":       score,
        "trend":       trend,
        "at_level":    at_level,
        "key_level":   key_level,
        "engulfing":   engulfing,
        "choch":       choch,
        "fvg":         fvg,
    })

    if not bot_state["running"]:
        return

    # ── Exit checks ──
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

    # ── Entry ──
    if signal == "buy" and cs["position"] is None and bot_state["demo_balance"] >= TRADE_SIZE_USD:
        notional  = TRADE_SIZE_USD * LEVERAGE
        amount    = notional / price
        liq_price = price * (1 - 1 / LEVERAGE + 0.001)
        cs.update({
            "amount": amount, "entry": price, "position": "long",
            "sl": sl, "tp": tp, "liq_price": liq_price,
        })
        bot_state["demo_balance"] -= TRADE_SIZE_USD
        log.info(
            f"[BUY 15×] {sym} @ ${price:.4f}  "
            f"notional=${notional:.0f}  SL=${sl:.4f}  TP=${tp:.4f}  "
            f"LIQ=${liq_price:.4f}  score={score}/10"
        )


def _close(sym: str, price: float, reason: str):
    cs = coin_state[sym]
    profit_pct = (price - cs["entry"]) / cs["entry"]
    cost_usd   = cs["amount"] * cs["entry"]
    profit_usd = cs["amount"] * (price - cs["entry"])
    proceeds   = cs["amount"] * price
    bot_state["demo_balance"] += proceeds

    h = load_history()
    h.append({
        "time":       datetime.utcnow().isoformat(),
        "symbol":     sym,
        "reason":     reason,
        "entry":      round(cs["entry"],  6),
        "exit":       round(price,        6),
        "sl":         round(cs["sl"],     6),
        "tp":         round(cs["tp"],     6),
        "cost_usd":   round(cost_usd,     2),
        "profit_usd": round(profit_usd,   2),
        "profit":     round(profit_pct,   6),
        "balance":    round(bot_state["demo_balance"], 2),
        "score":      cs.get("score", 0),
        "leverage":   LEVERAGE,
    })
    save_history(h)
    cs.update({"position": None, "entry": 0.0, "amount": 0.0,
               "sl": 0.0, "tp": 0.0, "liq_price": 0.0})
    emoji = "✅" if profit_pct > 0 else "❌"
    log.info(f"[{emoji}] {sym} {reason}  {profit_pct:+.2%}  balance=${bot_state['demo_balance']:.2f}")


# ─── MAIN TRADING LOOP ────────────────────────────────────────────────────────

async def analyze_coin(sym: str):
    """
    MSNR Full Analysis:
      chart_30d (hourly) → 1D / 4H  → swing-based key S/R levels
      chart_7d  (hourly) → 1H        → trend + CHoCH
      chart_1d  (5-min)  → 15M / 5M  → engulfing + FVG entry
    """
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

        if not tf_1d or not tf_4h or not tf_1h:
            return

        price = coin_state[sym]["last_price"]
        if price == 0.0:
            price = await fetch_price(coin_id)
            coin_state[sym]["last_price"] = price

        # ── 1. HTF Key Levels (1D + 4H swing highs/lows) ─────────────────
        res_1d, sup_1d = detect_swing_levels(tf_1d,  lookback=2)
        res_4h, sup_4h = detect_swing_levels(tf_4h,  lookback=3)
        all_levels = cluster_levels(res_1d + res_4h + sup_1d + sup_4h)

        # ── 2. Is price at an MSNR key level? ────────────────────────────
        at_level, key_level = nearest_level(price, all_levels)

        # ── 3. 1H Trend ───────────────────────────────────────────────────
        trend = get_trend_1h(tf_1h)

        # ── 4. CHoCH on 1H ────────────────────────────────────────────────
        choch = detect_choch(tf_1h)

        # ── 5. Engulfing on 15M ───────────────────────────────────────────
        engulfing = detect_engulfing(tf_15m[-5:]) if tf_15m else None

        # ── 6. FVG: 15M → 5M ─────────────────────────────────────────────
        fvg = (detect_fvg(tf_15m[-20:]) if tf_15m else None) or \
              (detect_fvg(tf_5m[-20:])  if tf_5m  else None)

        # ── 7. MSNR Signal ────────────────────────────────────────────────
        signal, score = generate_signal_msnr(
            at_level, key_level, engulfing, choch, fvg, price, trend
        )

        # ── 8. SL / TP ────────────────────────────────────────────────────
        engulf_low = tf_15m[-1]['low'] if (engulfing == 'bullish' and tf_15m) else 0.0
        sl, tp = calc_sl_tp_msnr(price, key_level, all_levels, engulf_low)

        # ── 9. Execute ────────────────────────────────────────────────────
        demo_tick(sym, price, signal, sl, tp, score,
                  trend, at_level, fvg, choch, engulfing, key_level)

        log.info(
            f"{sym} ${price:,.2f} | "
            f"level={key_level:.2f}({'✓' if at_level else '✗'}) | "
            f"trend={trend} choch={choch} engulf={engulfing} fvg={'✓' if fvg else '✗'} | "
            f"score={score}/10 → {signal}"
        )

    except Exception as e:
        log.warning(f"analyze {sym}: {e}")


async def price_loop():
    """Batch-fetch all prices every 30s — keeps dashboard live."""
    log.info("Price loop started")
    while True:
        try:
            prices = await fetch_all_prices()
            for sym, p in prices.items():
                if p > 0:
                    coin_state[sym]["last_price"] = p
            log.info(f"Prices: { {s: f'${p:,.2f}' for s, p in prices.items()} }")
        except Exception as e:
            log.warning(f"price_loop error: {e}")
        await asyncio.sleep(30)


async def trading_loop():
    """MSNR analysis — 1D/4H/1H/15M/5M per coin, 15s spacing."""
    log.info("MSNR Trading loop started  [1D · 4H · 1H · 15M · 5M]")
    try:
        prices = await fetch_all_prices()
        for sym, p in prices.items():
            if p > 0:
                coin_state[sym]["last_price"] = p
    except Exception as e:
        log.warning(f"Initial price fetch: {e}")

    while True:
        for sym in SYMBOLS:
            await analyze_coin(sym)
            await asyncio.sleep(15)
        await asyncio.sleep(30)


# ─── APP ──────────────────────────────────────────────────────────────────────

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
    h = load_history()
    wins     = [t for t in h if t.get("profit", 0) > 0]
    total_pnl = sum(t.get("profit_usd", 0) for t in h)
    return {
        **bot_state,
        "open_positions": [s for s, cs in coin_state.items() if cs["position"]],
        "total_trades":   len(h),
        "win_rate":       round(len(wins) / len(h), 3) if h else 0,
        "demo_pnl":       round(bot_state["demo_balance"] - DEMO_BALANCE, 2),
        "total_profit":   round(total_pnl, 2),
        "leverage":       LEVERAGE,
        "trade_size":     TRADE_SIZE_USD,
        "strategy":       "MSNR",
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
            price       = cs["last_price"]
            entry       = cs["entry"]
            amount      = cs["amount"]
            margin      = TRADE_SIZE_USD
            current_val = amount * price
            cost_val    = amount * entry
            pnl_usd     = current_val - cost_val
            pnl_pct     = (price - entry) / entry if entry else 0
            liq         = cs["liq_price"]
            dist_liq    = (price - liq) / price * 100 if liq else 0
            result.append({
                "symbol":        sym,
                "entry":         round(entry,              4),
                "current_price": round(price,              4),
                "margin_used":   round(margin,             2),
                "notional":      round(margin * LEVERAGE,  2),
                "current_value": round(current_val,        2),
                "pnl_usd":       round(pnl_usd,            2),
                "pnl_pct":       round(pnl_pct * 100,      2),
                "sl":            round(cs["sl"],            4),
                "tp":            round(cs["tp"],            4),
                "liq_price":     round(liq,                 4),
                "dist_liq_pct":  round(dist_liq,            2),
                "score":         cs["score"],
                "leverage":      LEVERAGE,
                "key_level":     round(cs["key_level"],     4),
            })
    return result


@app.get("/coins")
def coins_endpoint():
    return [
        {
            "symbol":    s,
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
        for s in SYMBOLS
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
