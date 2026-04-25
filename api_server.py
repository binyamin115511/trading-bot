"""
ICT Trading Bot — Full Implementation
Strategy: HTF Bias → Liquidity Sweep → Structure Shift → FVG Entry → Next Liquidity TP
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ict")

HISTORY_FILE = 'trade_history.json'
DEMO_BALANCE = 1_000.0
TRADE_SIZE_USD = 100.0  # $100 קבוע לכל עסקה

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
    }
    for sym in SYMBOLS
}

bot_state = {
    "running": True,
    "mode": "demo",
    "demo_balance": DEMO_BALANCE,
}


# ─── DATA LAYER ───────────────────────────────────────────────────────────────

async def fetch_ohlc(coin_id: str, days: int) -> list[dict]:
    """CoinGecko OHLC — returns list of {open,high,low,close} dicts."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        raw = r.json()
    # raw: [[ts, open, high, low, close], ...]
    return [{"ts": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4]} for c in raw]


async def fetch_price(coin_id: str) -> float:
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        return r.json()[coin_id]['usd']


async def fetch_all_prices() -> dict[str, float]:
    """Batch-fetch all 8 coin prices in a single CoinGecko call."""
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
    """HTF Bias: bullish if HH+HL, bearish if LH+LL, else range."""
    if len(candles) < 4:
        return "range"
    highs = [c['high'] for c in candles]
    lows  = [c['low']  for c in candles]
    hh = highs[-1] > highs[-2]
    hl = lows[-1]  > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1]  < lows[-2]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
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
    """Break of Structure — close breaks last swing high (bullish) or low (bearish)."""
    if not candles:
        return False
    close = candles[-1]['close']
    if bias == "bullish" and swing_highs:
        last_high = swing_highs[-1][1]
        return close > last_high
    if bias == "bearish" and swing_lows:
        last_low = swing_lows[-1][1]
        return close < last_low
    return False


def detect_fvg(candles: list[dict]) -> dict | None:
    """Latest Fair Value Gap (last 30 candles)."""
    for i in range(len(candles) - 1, 2, -1):
        c1 = candles[i - 2]
        c3 = candles[i]
        # Bullish FVG: gap up (c3.low > c1.high)
        if c3['low'] > c1['high']:
            return {"type": "bullish", "low": c1['high'], "high": c3['low'], "mid": (c1['high'] + c3['low']) / 2}
        # Bearish FVG: gap down (c3.high < c1.low)
        if c3['high'] < c1['low']:
            return {"type": "bearish", "low": c3['high'], "high": c1['low'], "mid": (c3['high'] + c1['low']) / 2}
    return None


def detect_equal_levels(values: list[float], tolerance: float = 0.002) -> list[float]:
    levels = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) / max(values[i], 1) < tolerance:
                levels.append((values[i] + values[j]) / 2)
    return levels


def detect_sweep(price: float, levels: list[float], tolerance: float = 0.003) -> tuple[bool, float]:
    """Returns (swept, swept_level) if price came within tolerance of a level."""
    for lvl in levels:
        if abs(price - lvl) / max(lvl, 1) < tolerance or price > lvl:
            return True, lvl
    return False, 0.0


def is_killzone() -> bool:
    """London: 07-10 UTC | New York: 13-16 UTC."""
    hour = datetime.now(timezone.utc).hour
    return 7 <= hour <= 10 or 13 <= hour <= 16


def fake_breakout(candle: dict, level: float) -> bool:
    return candle['high'] > level and candle['close'] < level


# ─── SIGNAL ENGINE ────────────────────────────────────────────────────────────

def generate_signal(bias: str, sweep: bool, fvg: dict | None,
                    bos: bool, price: float, killzone: bool) -> tuple[str, int]:
    score = 0

    if bias == "bullish":
        score += 2
    elif bias == "bearish":
        score -= 2  # bearish bias → don't buy

    if sweep:
        score += 2

    if fvg and fvg['type'] == "bullish" and fvg['low'] <= price <= fvg['high']:
        score += 1  # price inside FVG

    if bos:
        score += 2

    if killzone:
        score += 1  # session bonus

    if score >= 5:
        return "buy", score

    return "hold", score


# ─── RISK MANAGEMENT ──────────────────────────────────────────────────────────

def calc_sl_tp(price: float, sweep_level: float, next_liq: float, bias: str):
    """SL below liquidity sweep, TP at next liquidity level."""
    if bias == "bullish":
        sl = sweep_level * 0.998 if sweep_level else price * 0.98
        tp = next_liq if next_liq > price else price * 1.04
    else:
        sl = price * 1.02
        tp = price * 0.96
    return sl, tp


# ─── DEMO EXECUTION ───────────────────────────────────────────────────────────

def demo_tick(sym: str, price: float, signal: str, sl: float, tp: float, score: int,
              bias: str, sweep: bool, fvg, bos: bool):
    cs = coin_state[sym]
    cs.update({"last_price": price, "last_signal": signal, "score": score,
                "bias": bias, "sweep": sweep, "fvg": fvg, "bos": bos})

    if not bot_state["running"]:
        return

    # Check exit first
    if cs["position"] == "long":
        entry = cs["entry"]
        if price <= cs["sl"]:
            _close(sym, price, "stop-loss")
        elif price >= cs["tp"]:
            _close(sym, price, "take-profit")
        elif signal == "hold" and price < entry * 0.985:
            _close(sym, price, "signal-exit")
        return

    # Entry
    if signal == "buy" and cs["position"] is None and bot_state["demo_balance"] >= TRADE_SIZE_USD:
        amount_usd = TRADE_SIZE_USD
        amount = amount_usd / price
        cs.update({"amount": amount, "entry": price, "position": "long",
                   "sl": sl, "tp": tp})
        bot_state["demo_balance"] -= amount_usd
        log.info(f"[BUY] {sym} @ ${price:.4f} SL=${sl:.4f} TP=${tp:.4f} score={score}")


def _close(sym: str, price: float, reason: str):
    cs = coin_state[sym]
    profit_pct = (price - cs["entry"]) / cs["entry"]
    cost_usd   = cs["amount"] * cs["entry"]
    profit_usd = cs["amount"] * (price - cs["entry"])
    proceeds   = cs["amount"] * price
    bot_state["demo_balance"] += proceeds

    h = load_history()
    h.append({
        "time": datetime.utcnow().isoformat(),
        "symbol": sym, "reason": reason,
        "entry": round(cs["entry"], 6), "exit": round(price, 6),
        "sl": round(cs["sl"], 6), "tp": round(cs["tp"], 6),
        "cost_usd": round(cost_usd, 2),
        "profit_usd": round(profit_usd, 2),
        "profit": round(profit_pct, 6),
        "balance": round(bot_state["demo_balance"], 2),
        "score": cs.get("score", 0),
    })
    save_history(h)
    cs.update({"position": None, "entry": 0.0, "amount": 0.0, "sl": 0.0, "tp": 0.0})
    emoji = "✅" if profit_pct > 0 else "❌"
    log.info(f"[{emoji}] {sym} {reason} {profit_pct:.2%} → ${bot_state['demo_balance']:.2f}")


# ─── MAIN TRADING LOOP ────────────────────────────────────────────────────────

async def analyze_coin(sym: str):
    coin_id = COIN_IDS[sym]
    try:
        # Fetch HTF (4h candles via 7 days) and LTF (30m candles via 1 day)
        # Price already updated by price_loop — use cached value
        htf_candles, ltf_candles = await asyncio.gather(
            fetch_ohlc(coin_id, 7),
            fetch_ohlc(coin_id, 1),
        )
        if not htf_candles or not ltf_candles:
            return

        # Use cached price (updated every 30s by price_loop)
        price = coin_state[sym]["last_price"]
        if price == 0.0:
            # Fallback: fetch individually if price_loop hasn't run yet
            price = await fetch_price(coin_id)
            coin_state[sym]["last_price"] = price

        # 1. HTF Bias
        bias = get_bias(htf_candles[-20:])

        # 2. Swing structure on LTF
        swing_h, swing_l = swing_highs_lows(ltf_candles)

        # 3. Break of Structure
        bos = detect_bos(ltf_candles, swing_h, swing_l, bias)

        # 4. Liquidity levels (equal highs)
        highs_vals = [c['high'] for c in ltf_candles[-30:]]
        lows_vals  = [c['low']  for c in ltf_candles[-30:]]
        eq_highs = detect_equal_levels(highs_vals)
        eq_lows  = detect_equal_levels(lows_vals)
        liq_levels = eq_highs if bias == "bullish" else eq_lows

        # 5. Liquidity sweep
        swept, sweep_level = detect_sweep(price, liq_levels)

        # 6. FVG on LTF
        fvg = detect_fvg(ltf_candles[-20:])

        # 7. Killzone
        kz = is_killzone()

        # 8. Signal
        signal, score = generate_signal(bias, swept, fvg, bos, price, kz)

        # 9. SL / TP
        next_liq = max(eq_highs) if eq_highs else price * 1.04
        sl, tp = calc_sl_tp(price, sweep_level, next_liq, bias)

        # 10. Execute
        demo_tick(sym, price, signal, sl, tp, score, bias, swept, fvg, bos)

        log.info(f"{sym} ${price:.4f} bias={bias} sweep={swept} bos={bos} fvg={'✓' if fvg else '✗'} kz={kz} score={score} → {signal}")

    except Exception as e:
        log.warning(f"analyze {sym}: {e}")


async def price_loop():
    """Fast loop: batch-fetch all prices every 30 seconds so dashboard always shows values."""
    log.info("Price loop started")
    while True:
        try:
            prices = await fetch_all_prices()
            for sym, price in prices.items():
                if price > 0:
                    coin_state[sym]["last_price"] = price
            log.info(f"Prices updated: { {s: f'${p:.2f}' for s, p in prices.items()} }")
        except Exception as e:
            log.warning(f"price_loop error: {e}")
        await asyncio.sleep(30)


async def trading_loop():
    """ICT analysis loop — runs full analysis per coin with spacing to avoid rate limits."""
    log.info("ICT Trading loop started")
    # Initial price fetch so dashboard shows values immediately
    try:
        prices = await fetch_all_prices()
        for sym, price in prices.items():
            if price > 0:
                coin_state[sym]["last_price"] = price
    except Exception as e:
        log.warning(f"Initial price fetch failed: {e}")

    while True:
        for sym in SYMBOLS:
            await analyze_coin(sym)
            await asyncio.sleep(12)  # 12s between coins → 96s per full cycle
        await asyncio.sleep(30)  # rest before next cycle


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
    t1 = asyncio.create_task(price_loop())    # fast: prices every 30s
    t2 = asyncio.create_task(trading_loop())  # slow: ICT analysis per coin
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
            cost_usd   = amount * entry
            current_val = amount * price
            pnl_usd    = current_val - cost_usd
            pnl_pct    = (price - entry) / entry if entry else 0
            result.append({
                "symbol": sym,
                "entry": round(entry, 4),
                "current_price": round(price, 4),
                "cost_usd": round(cost_usd, 2),
                "current_value": round(current_val, 2),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "sl": round(cs["sl"], 4),
                "tp": round(cs["tp"], 4),
                "score": cs["score"],
            })
    return result


@app.get("/coins")
def coins_endpoint():
    return [
        {
            "symbol": s,
            "price": coin_state[s]["last_price"],
            "signal": coin_state[s]["last_signal"],
            "position": coin_state[s]["position"],
            "entry": coin_state[s]["entry"],
            "sl": coin_state[s]["sl"],
            "tp": coin_state[s]["tp"],
            "bias": coin_state[s]["bias"],
            "sweep": coin_state[s]["sweep"],
            "bos": coin_state[s]["bos"],
            "score": coin_state[s]["score"],
            "fvg": coin_state[s]["fvg"] is not None,
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
