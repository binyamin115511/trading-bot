"""
ICT Trading Bot — Full Implementation
Strategy: HTF Bias → Liquidity Sweep → Structure Shift → FVG Entry → Next Liquidity TP
Timeframes: 30D · 7D · 1D · 4H · 1H · 30M · 15M · 5M
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
log = logging.getLogger("ict")

HISTORY_FILE = 'trade_history.json'
DEMO_BALANCE = 1_000.0
TRADE_SIZE_USD = 100.0  # $100 קבוע לכל עסקה

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
SYMBOLS = list(COINS.values())
COIN_IDS = {v: k for k, v in COINS.items()}

coin_state: dict[str, dict] = {
    sym: {
        "position": None, "entry": 0.0, "amount": 0.0, "sl": 0.0, "tp": 0.0,
        "last_price": 0.0, "last_signal": "hold",
        "fvg": None, "sweep": False, "bias": "range", "bos": False,
        "score": 0,
        # Per-timeframe breakdown
        "bias_30d": "range", "bias_7d": "range", "bias_1d": "range",
        "bos_4h": False, "bos_1h": False,
    }
    for sym in SYMBOLS
}

bot_state = {
    "running": True,
    "mode": "demo",
    "demo_balance": DEMO_BALANCE,
}


# ─── DATA LAYER ───────────────────────────────────────────────────────────────

async def fetch_market_chart(coin_id: str, days: int) -> list:
    """
    CoinGecko market_chart → [[ts_ms, price], ...].
    Auto granularity (free tier):
      days=1   → ~5-min intervals  (~288 pts) → use for 5M / 15M / 30M
      days≤90  → hourly intervals             → use for 1H / 4H / 1D / 7D / 30D
    """
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days={days}")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        data = r.json()
    return data.get('prices', [])


def to_ohlc(prices: list, interval_ms: int) -> list[dict]:
    """Resample price ticks [[ts_ms, price]] into OHLC candles."""
    if not prices:
        return []
    candles: dict[int, dict] = {}
    for ts, price in prices:
        bucket = (int(ts) // interval_ms) * interval_ms
        if bucket not in candles:
            candles[bucket] = {
                'ts': bucket, 'open': price,
                'high': price, 'low': price, 'close': price,
            }
        else:
            c = candles[bucket]
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
    """Single CoinGecko call → prices for all 8 coins."""
    ids = ','.join(COINS.keys())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        data = r.json()
    result = {}
    for coin_id, sym in COINS.items():
        if coin_id in data and 'usd' in data[coin_id]:
            result[sym] = data[coin_id]['usd']
    return result


# ─── ICT ENGINES ──────────────────────────────────────────────────────────────

def get_bias(candles: list[dict]) -> str:
    """Bullish if HH+HL, bearish if LH+LL, else range."""
    if len(candles) < 4:
        return "range"
    highs = [c['high'] for c in candles]
    lows  = [c['low']  for c in candles]
    hh = highs[-1] > highs[-2]
    hl = lows[-1]  > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1]  < lows[-2]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "range"


def swing_highs_lows(candles: list[dict], lookback: int = 3):
    highs, lows = [], []
    n = len(candles)
    for i in range(lookback, n - lookback):
        h = candles[i]['high']
        l = candles[i]['low']
        if all(h >= candles[j]['high'] for j in range(i - lookback, i + lookback + 1) if j != i):
            highs.append((i, h))
        if all(l <= candles[j]['low']  for j in range(i - lookback, i + lookback + 1) if j != i):
            lows.append((i, l))
    return highs, lows


def detect_bos(candles: list[dict], swing_highs, swing_lows, bias: str) -> bool:
    """Break of Structure: close > last swing high (bullish) or < last swing low (bearish)."""
    if not candles:
        return False
    close = candles[-1]['close']
    if bias == "bullish" and swing_highs:
        return close > swing_highs[-1][1]
    if bias == "bearish" and swing_lows:
        return close < swing_lows[-1][1]
    return False


def detect_fvg(candles: list[dict]) -> dict | None:
    """Latest Fair Value Gap in candle list."""
    for i in range(len(candles) - 1, 2, -1):
        c1, c3 = candles[i - 2], candles[i]
        if c3['low'] > c1['high']:      # bullish FVG
            return {"type": "bullish", "low": c1['high'], "high": c3['low'],
                    "mid": (c1['high'] + c3['low']) / 2}
        if c3['high'] < c1['low']:      # bearish FVG
            return {"type": "bearish", "low": c3['high'], "high": c1['low'],
                    "mid": (c3['high'] + c1['low']) / 2}
    return None


def detect_equal_levels(values: list[float], tolerance: float = 0.002) -> list[float]:
    levels = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) / max(values[i], 1) < tolerance:
                levels.append((values[i] + values[j]) / 2)
    return levels


def detect_sweep(price: float, levels: list[float], tolerance: float = 0.003) -> tuple[bool, float]:
    for lvl in levels:
        if abs(price - lvl) / max(lvl, 1) < tolerance or price > lvl:
            return True, lvl
    return False, 0.0


# ─── MULTI-TF SIGNAL ENGINE ───────────────────────────────────────────────────

def generate_signal_mtf(
    bias_30d: str, bias_7d: str, bias_1d: str,
    bos_4h: bool, bos_1h: bool,
    swept: bool, fvg: dict | None, price: float,
) -> tuple[str, int]:
    """
    ICT multi-timeframe score (max 10), threshold 6 → LONG entry.

    Bias alignment (top-down):
      +2  30D daily bias bullish   (dominant monthly trend)
      +1  7D  bias bullish         (weekly alignment)
      +1  1D  bias bullish         (daily alignment)

    Structure:
      +2  4H Break of Structure    (confirmed market shift)
      +1  1H Break of Structure    (intraday confirmation)

    Liquidity:
      +2  Sweep of equal highs/lows on 4H or 1H

    Entry precision:
      +1  Price inside bullish FVG on 30M / 15M / 5M

    Bearish bias at any level subtracts points → prevents longs in downtrends.
    """
    score = 0

    # Bias
    if bias_30d == "bullish":   score += 2
    elif bias_30d == "bearish": score -= 2
    if bias_7d  == "bullish":   score += 1
    elif bias_7d  == "bearish": score -= 1
    if bias_1d  == "bullish":   score += 1
    elif bias_1d  == "bearish": score -= 1

    # Structure
    if bos_4h: score += 2
    if bos_1h: score += 1

    # Liquidity sweep
    if swept: score += 2

    # FVG entry
    if fvg and fvg['type'] == "bullish" and fvg['low'] <= price <= fvg['high']:
        score += 1

    return ("buy", score) if score >= 6 else ("hold", score)


# ─── RISK MANAGEMENT ──────────────────────────────────────────────────────────

def calc_sl_tp(price: float, sweep_level: float, next_liq: float, bias: str):
    """SL below liquidity sweep, TP at next liquidity target."""
    if bias == "bullish":
        sl = sweep_level * 0.998 if sweep_level else price * 0.98
        tp = next_liq if next_liq > price else price * 1.04
    else:
        sl = price * 1.02
        tp = price * 0.96
    return sl, tp


# ─── DEMO EXECUTION ───────────────────────────────────────────────────────────

def demo_tick(sym: str, price: float, signal: str, sl: float, tp: float, score: int,
              bias: str, sweep: bool, fvg, bos: bool,
              bias_30d: str, bias_7d: str, bias_1d: str,
              bos_4h: bool, bos_1h: bool):
    cs = coin_state[sym]
    cs.update({
        "last_price": price, "last_signal": signal, "score": score,
        "bias": bias, "sweep": sweep, "fvg": fvg, "bos": bos,
        "bias_30d": bias_30d, "bias_7d": bias_7d, "bias_1d": bias_1d,
        "bos_4h": bos_4h, "bos_1h": bos_1h,
    })

    if not bot_state["running"]:
        return

    # Exit check
    if cs["position"] == "long":
        if price <= cs["sl"]:
            _close(sym, price, "stop-loss")
        elif price >= cs["tp"]:
            _close(sym, price, "take-profit")
        elif signal == "hold" and price < cs["entry"] * 0.985:
            _close(sym, price, "signal-exit")
        return

    # Entry
    if signal == "buy" and cs["position"] is None and bot_state["demo_balance"] >= TRADE_SIZE_USD:
        amount = TRADE_SIZE_USD / price
        cs.update({"amount": amount, "entry": price, "position": "long", "sl": sl, "tp": tp})
        bot_state["demo_balance"] -= TRADE_SIZE_USD
        log.info(f"[BUY] {sym} @ ${price:.4f}  SL=${sl:.4f}  TP=${tp:.4f}  score={score}/10")


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
    })
    save_history(h)
    cs.update({"position": None, "entry": 0.0, "amount": 0.0, "sl": 0.0, "tp": 0.0})
    emoji = "✅" if profit_pct > 0 else "❌"
    log.info(f"[{emoji}] {sym} {reason}  {profit_pct:+.2%}  balance=${bot_state['demo_balance']:.2f}")


# ─── MAIN TRADING LOOP ────────────────────────────────────────────────────────

async def analyze_coin(sym: str):
    """
    8-timeframe ICT analysis per coin:
      chart_30d (hourly ticks) → daily OHLC  → 30D / 7D / 1D bias
      chart_7d  (hourly ticks) → 4H / 1H     → BOS + liquidity sweep
      chart_1d  (5-min  ticks) → 30M / 15M / 5M → FVG entry
    """
    coin_id = COIN_IDS[sym]
    try:
        # 3 concurrent API calls cover all 8 timeframes
        chart_30d, chart_7d, chart_1d = await asyncio.gather(
            fetch_market_chart(coin_id, 30),   # hourly → 1D/7D/30D
            fetch_market_chart(coin_id, 7),    # hourly → 4H/1H
            fetch_market_chart(coin_id, 1),    # 5-min  → 30M/15M/5M
        )

        # ── Resample to all 8 timeframes ────────────────────────────────────
        tf_1d  = to_ohlc(chart_30d, MS_1D)   # daily  (30 candles)
        tf_4h  = to_ohlc(chart_7d,  MS_4H)   # 4-hour (42 candles)
        tf_1h  = to_ohlc(chart_7d,  MS_1H)   # 1-hour (168 candles)
        tf_30m = to_ohlc(chart_1d,  MS_30M)  # 30-min (48 candles)
        tf_15m = to_ohlc(chart_1d,  MS_15M)  # 15-min (96 candles)
        tf_5m  = to_ohlc(chart_1d,  MS_5M)   # 5-min  (288 candles)

        if not tf_1d or not tf_4h or not tf_1h:
            return

        # Cached price (updated every 30s by price_loop)
        price = coin_state[sym]["last_price"]
        if price == 0.0:
            price = await fetch_price(coin_id)
            coin_state[sym]["last_price"] = price

        # ── BIAS: top-down across 30D / 7D / 1D ─────────────────────────────
        bias_30d = get_bias(tf_1d)                                         # all 30 daily candles
        bias_7d  = get_bias(tf_1d[-7:]  if len(tf_1d)  >= 7  else tf_1d)  # last 7 daily
        bias_1d  = get_bias(tf_4h[-6:]  if len(tf_4h)  >= 6  else tf_4h)  # last 6×4H ≈ 1 day

        # Majority vote for dominant bias
        votes = [bias_30d, bias_7d, bias_1d]
        bias  = max(set(votes), key=votes.count)

        # ── STRUCTURE: BOS on 4H and 1H ──────────────────────────────────────
        sh_4h, sl_4h = swing_highs_lows(tf_4h)
        sh_1h, sl_1h = swing_highs_lows(tf_1h)
        bos_4h = detect_bos(tf_4h, sh_4h, sl_4h, bias)
        bos_1h = detect_bos(tf_1h, sh_1h, sl_1h, bias)

        # ── LIQUIDITY: equal highs/lows + sweep on 4H (primary) + 1H ────────
        highs_4h = [c['high'] for c in tf_4h[-40:]]
        lows_4h  = [c['low']  for c in tf_4h[-40:]]
        eq_highs_4h = detect_equal_levels(highs_4h)
        eq_lows_4h  = detect_equal_levels(lows_4h)

        highs_1h = [c['high'] for c in tf_1h[-30:]]
        lows_1h  = [c['low']  for c in tf_1h[-30:]]
        eq_highs_1h = detect_equal_levels(highs_1h)
        eq_lows_1h  = detect_equal_levels(lows_1h)

        liq_4h = eq_highs_4h if bias == "bullish" else eq_lows_4h
        liq_1h = eq_highs_1h if bias == "bullish" else eq_lows_1h

        swept_4h, sweep_lvl_4h = detect_sweep(price, liq_4h)
        swept_1h, sweep_lvl_1h = detect_sweep(price, liq_1h)
        swept       = swept_4h or swept_1h
        sweep_level = sweep_lvl_4h if swept_4h else sweep_lvl_1h

        # ── FVG: 30M → 15M → 5M (best available precision) ──────────────────
        fvg = (
            detect_fvg(tf_30m[-30:]) or
            detect_fvg(tf_15m[-30:]) or
            detect_fvg(tf_5m[-30:])
        )

        # ── SIGNAL ───────────────────────────────────────────────────────────
        signal, score = generate_signal_mtf(
            bias_30d, bias_7d, bias_1d,
            bos_4h, bos_1h,
            swept, fvg, price,
        )

        # ── SL / TP ──────────────────────────────────────────────────────────
        all_eq_highs = eq_highs_4h + eq_highs_1h
        next_liq = max(all_eq_highs) if all_eq_highs else price * 1.04
        sl, tp = calc_sl_tp(price, sweep_level, next_liq, bias)

        # ── EXECUTE ──────────────────────────────────────────────────────────
        demo_tick(
            sym, price, signal, sl, tp, score,
            bias, swept, fvg, bos_4h or bos_1h,
            bias_30d, bias_7d, bias_1d, bos_4h, bos_1h,
        )

        log.info(
            f"{sym} ${price:,.2f} | "
            f"30D={bias_30d} 7D={bias_7d} 1D={bias_1d} | "
            f"4H-BOS={bos_4h} 1H-BOS={bos_1h} sweep={swept} | "
            f"FVG={'✓' if fvg else '✗'} | score={score}/10 → {signal}"
        )

    except Exception as e:
        log.warning(f"analyze {sym}: {e}")


async def price_loop():
    """Fast loop — batch price update for all 8 coins every 30 seconds."""
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
    """ICT analysis loop — 8 timeframes per coin, 15s spacing to respect rate limits."""
    log.info("ICT Trading loop started  [30D · 7D · 1D · 4H · 1H · 30M · 15M · 5M]")
    # Prices before first analysis cycle
    try:
        prices = await fetch_all_prices()
        for sym, p in prices.items():
            if p > 0:
                coin_state[sym]["last_price"] = p
    except Exception as e:
        log.warning(f"Initial price fetch failed: {e}")

    while True:
        for sym in SYMBOLS:
            await analyze_coin(sym)
            await asyncio.sleep(15)   # 15s × 8 coins = 2 min per full cycle
        await asyncio.sleep(30)       # rest between cycles


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
    t1 = asyncio.create_task(price_loop())    # fast  — prices every 30s
    t2 = asyncio.create_task(trading_loop())  # slow  — ICT analysis per coin
    yield
    t1.cancel()
    t2.cancel()


app = FastAPI(title="ICT Trading Bot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/status")
def status():
    h = load_history()
    wins = [t for t in h if t.get("profit", 0) > 0]
    return {
        **bot_state,
        "open_positions": [s for s, cs in coin_state.items() if cs["position"]],
        "total_trades": len(h),
        "win_rate": round(len(wins) / len(h), 3) if h else 0,
        "demo_pnl": round(bot_state["demo_balance"] - DEMO_BALANCE, 2),
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
        cs.update({"position": None, "entry": 0.0, "amount": 0.0, "sl": 0.0, "tp": 0.0})
    save_history([])
    return {"status": "reset"}


@app.get("/positions")
def positions_endpoint():
    """פוזיציות פתוחות עם P&L חי בדולרים."""
    result = []
    for sym, cs in coin_state.items():
        if cs["position"] == "long":
            price = cs["last_price"]
            entry = cs["entry"]
            amount = cs["amount"]
            cost_usd    = amount * entry
            current_val = amount * price
            pnl_usd     = current_val - cost_usd
            pnl_pct     = (price - entry) / entry if entry else 0
            result.append({
                "symbol":        sym,
                "entry":         round(entry,       4),
                "current_price": round(price,       4),
                "cost_usd":      round(cost_usd,    2),
                "current_value": round(current_val, 2),
                "pnl_usd":       round(pnl_usd,     2),
                "pnl_pct":       round(pnl_pct * 100, 2),
                "sl":            round(cs["sl"],    4),
                "tp":            round(cs["tp"],    4),
                "score":         cs["score"],
            })
    return result


@app.get("/coins")
def coins_endpoint():
    return [
        {
            "symbol":   s,
            "price":    coin_state[s]["last_price"],
            "signal":   coin_state[s]["last_signal"],
            "position": coin_state[s]["position"],
            "entry":    coin_state[s]["entry"],
            "sl":       coin_state[s]["sl"],
            "tp":       coin_state[s]["tp"],
            "bias":     coin_state[s]["bias"],
            "bias_30d": coin_state[s]["bias_30d"],
            "bias_7d":  coin_state[s]["bias_7d"],
            "bias_1d":  coin_state[s]["bias_1d"],
            "sweep":    coin_state[s]["sweep"],
            "bos":      coin_state[s]["bos"],
            "bos_4h":   coin_state[s]["bos_4h"],
            "bos_1h":   coin_state[s]["bos_1h"],
            "score":    coin_state[s]["score"],
            "fvg":      coin_state[s]["fvg"] is not None,
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
    return HTMLResponse("<h1>ICT Bot running</h1>")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
