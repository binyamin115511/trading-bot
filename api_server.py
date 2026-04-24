import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

HISTORY_FILE = 'trade_history.json'
DEMO_BALANCE = 1_000.0

# CoinGecko IDs ↔ display symbols
COINS = {
    'bitcoin':    'BTC',
    'ethereum':   'ETH',
    'binancecoin':'BNB',
    'solana':     'SOL',
    'ripple':     'XRP',
    'dogecoin':   'DOGE',
    'cardano':    'ADA',
    'avalanche-2':'AVAX',
}
SYMBOLS = list(COINS.values())

coin_state: dict[str, dict] = {
    sym: {"position": None, "entry": 0.0, "amount": 0.0,
          "last_price": 0.0, "last_signal": "hold", "prices": []}
    for sym in SYMBOLS
}

bot_state = {
    "running": True,
    "mode": "demo",
    "risk": 0.05,
    "demo_balance": DEMO_BALANCE,
}


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(h: list):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(h, f, indent=2)


async def fetch_prices() -> dict[str, float]:
    ids = ','.join(COINS.keys())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        data = r.json()
    return {COINS[k]: v['usd'] for k, v in data.items() if 'usd' in v}


def compute_signal(prices_list: list) -> str:
    if len(prices_list) < 10:
        return "hold"
    arr = np.array(prices_list)
    X = np.arange(len(arr)).reshape(-1, 1)
    pred = LinearRegression().fit(X, arr).predict([[len(arr)]])[0]
    ema = float(np.convolve(arr, np.ones(min(20, len(arr))) / min(20, len(arr)), 'valid')[-1])
    current = arr[-1]
    score = (1 if pred > current else -1 if pred < current else 0) + \
            (1 if current > ema else -1 if current < ema else 0)
    return "buy" if score >= 1 else "sell" if score <= -1 else "hold"


def demo_tick(sym: str, price: float, signal: str):
    cs = coin_state[sym]
    cs["last_price"] = price
    cs["last_signal"] = signal

    if not bot_state["running"]:
        return

    if signal == "buy" and cs["position"] is None and bot_state["demo_balance"] > 5:
        amount_usd = bot_state["demo_balance"] * bot_state["risk"]
        cs["amount"] = amount_usd / price
        cs["entry"] = price
        cs["position"] = "long"
        bot_state["demo_balance"] -= amount_usd
        log.info(f"[BUY] {sym} @ ${price:.4f}")

    elif cs["position"] == "long":
        entry = cs["entry"]
        if signal == "sell" or price <= entry * 0.98 or price >= entry * 1.04:
            reason = "signal" if signal == "sell" else \
                     ("stop-loss" if price <= entry * 0.98 else "take-profit")
            _close(sym, price, reason)


def _close(sym: str, price: float, reason: str):
    cs = coin_state[sym]
    profit_pct = (price - cs["entry"]) / cs["entry"]
    bot_state["demo_balance"] += cs["amount"] * price
    h = load_history()
    h.append({"time": datetime.utcnow().isoformat(), "symbol": sym,
               "reason": reason, "entry": round(cs["entry"], 6),
               "exit": round(price, 6), "profit": round(profit_pct, 6),
               "balance": round(bot_state["demo_balance"], 2)})
    save_history(h)
    cs.update({"position": None, "entry": 0.0, "amount": 0.0})
    log.info(f"[{'✅' if profit_pct>0 else '❌'}] {sym} {reason} {profit_pct:.2%} → ${bot_state['demo_balance']:.2f}")


async def trading_loop():
    log.info("Trading loop started")
    while True:
        try:
            prices = await fetch_prices()
            for sym, price in prices.items():
                cs = coin_state[sym]
                cs["prices"].append(price)
                if len(cs["prices"]) > 100:
                    cs["prices"].pop(0)
                signal = compute_signal(cs["prices"])
                demo_tick(sym, price, signal)
            log.info(f"Tick done — BTC=${prices.get('BTC', 0):,.0f} running={bot_state['running']}")
        except Exception as e:
            log.warning(f"Tick error: {e}")
        await asyncio.sleep(30)  # CoinGecko free: max 50 req/min


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(trading_loop())
    yield
    task.cancel()


app = FastAPI(title="AI Trading Bot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/status")
def status():
    h = load_history()
    wins = [t for t in h if t.get("profit", 0) > 0]
    return {**bot_state,
            "open_positions": [s for s, cs in coin_state.items() if cs["position"]],
            "total_trades": len(h),
            "win_rate": round(len(wins) / len(h), 3) if h else 0,
            "demo_pnl": round(bot_state["demo_balance"] - DEMO_BALANCE, 2)}


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
        cs.update({"position": None, "entry": 0.0, "amount": 0.0})
    save_history([])
    return {"status": "reset"}


@app.get("/coins")
def coins_endpoint():
    return [{"symbol": s, "price": coin_state[s]["last_price"],
             "signal": coin_state[s]["last_signal"],
             "position": coin_state[s]["position"],
             "entry": coin_state[s]["entry"]} for s in SYMBOLS]


@app.get("/history")
def history_endpoint():
    return load_history()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if os.path.exists("dashboard.html"):
        return open("dashboard.html").read()
    return HTMLResponse("<h1>Bot running</h1>")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
