import asyncio
import json
import logging
import os
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sklearn.linear_model import LinearRegression

load_dotenv()
log = logging.getLogger("api")

exchange = ccxt.binance({'enableRateLimit': True})

TIMEFRAME = os.getenv('TIMEFRAME', '1m')
HISTORY_FILE = 'trade_history.json'
DEMO_BALANCE = 1_000.0

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
    'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT',
]

# per-symbol state: { symbol: {position, entry, amount, last_price, last_signal} }
coin_state: dict[str, dict] = {
    s: {"position": None, "entry": 0.0, "amount": 0.0, "last_price": 0.0, "last_signal": "hold"}
    for s in SYMBOLS
}

bot_state = {
    "running": True,
    "mode": "demo",
    "risk": 0.05,          # 5% per trade (split across coins)
    "demo_balance": DEMO_BALANCE,
}

app = FastAPI(title="AI Trading Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def trading_loop():
    """לולאת מסחר רקע — רצה תמיד, עצמאית מהדשבורד."""
    symbol_index = 0
    while True:
        symbol = SYMBOLS[symbol_index % len(SYMBOLS)]
        symbol_index += 1
        try:
            price, signal = await asyncio.get_event_loop().run_in_executor(
                None, compute_signal_for, symbol
            )
            if bot_state["mode"] == "demo":
                demo_tick(symbol, price, signal)
        except Exception as e:
            log.warning(f"loop error {symbol}: {e}")
        await asyncio.sleep(4)


@app.on_event("startup")
async def startup():
    asyncio.create_task(trading_loop())


def load_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list[dict]):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def compute_signal_for(symbol: str) -> tuple[float, str]:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
    df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    prices = df['c'].values
    X = np.arange(len(prices)).reshape(-1, 1)
    pred = LinearRegression().fit(X, prices).predict([[len(prices) + 1]])[0]
    ema = df['c'].ewm(span=20).mean().iloc[-1]
    current = prices[-1]
    score = (1 if pred > current else -1 if pred < current else 0) + \
            (1 if current > ema else -1 if current < ema else 0)
    signal = "buy" if score >= 1 else "sell" if score <= -1 else "hold"
    return float(current), signal


def demo_tick(symbol: str, price: float, signal: str):
    cs = coin_state[symbol]
    cs["last_price"] = price
    cs["last_signal"] = signal

    if not bot_state["running"]:
        return

    if signal == "buy" and cs["position"] is None and bot_state["demo_balance"] > 5:
        amount_usd = bot_state["demo_balance"] * bot_state["risk"]
        amount = amount_usd / price
        cs["amount"] = amount
        cs["entry"] = price
        cs["position"] = "long"
        bot_state["demo_balance"] -= amount_usd
        log.info(f"[DEMO] BUY {symbol} {amount:.6f} @ ${price:,.4f}")

    elif cs["position"] == "long":
        entry = cs["entry"]
        if signal == "sell" or price <= entry * 0.98 or price >= entry * 1.04:
            reason = "signal" if signal == "sell" else ("stop-loss" if price <= entry * 0.98 else "take-profit")
            _close(symbol, price, reason)


def _close(symbol: str, price: float, reason: str):
    cs = coin_state[symbol]
    entry = cs["entry"]
    amount = cs["amount"]
    proceeds = amount * price
    profit_pct = (price - entry) / entry
    bot_state["demo_balance"] += proceeds
    cs["position"] = None
    cs["entry"] = 0.0
    cs["amount"] = 0.0

    history = load_history()
    history.append({
        "time": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "reason": reason,
        "entry": round(entry, 6),
        "exit": round(price, 6),
        "profit": round(profit_pct, 6),
        "balance": round(bot_state["demo_balance"], 2),
    })
    save_history(history)
    log.info(f"[DEMO] {'✅' if profit_pct>0 else '❌'} {symbol} {reason} profit={profit_pct:.2%} balance=${bot_state['demo_balance']:,.2f}")


@app.get("/status")
def status():
    history = load_history()
    wins = [t for t in history if t.get("profit", 0) > 0]
    open_positions = [s for s, cs in coin_state.items() if cs["position"]]
    return {
        **bot_state,
        "open_positions": open_positions,
        "total_trades": len(history),
        "win_rate": round(len(wins) / len(history), 3) if history else 0,
        "demo_pnl": round(bot_state["demo_balance"] - DEMO_BALANCE, 2),
        "symbols": SYMBOLS,
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
    bot_state["demo_balance"] = DEMO_BALANCE
    bot_state["running"] = False
    for cs in coin_state.values():
        cs.update({"position": None, "entry": 0.0, "amount": 0.0})
    save_history([])
    return {"status": "reset", "balance": DEMO_BALANCE}


@app.get("/coins")
def coins_endpoint():
    return [
        {
            "symbol": s,
            "price": coin_state[s]["last_price"],
            "signal": coin_state[s]["last_signal"],
            "position": coin_state[s]["position"],
            "entry": coin_state[s]["entry"],
        }
        for s in SYMBOLS
    ]


@app.post("/mode/{mode}")
def set_mode(mode: str):
    if mode not in ("demo", "auto", "manual"):
        return {"error": "invalid mode"}
    bot_state["mode"] = mode
    return {"mode": mode}


@app.get("/history")
def history_endpoint():
    return load_history()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    symbol_index = 0
    try:
        while True:
            # rotate through symbols — one per tick to avoid rate limits
            symbol = SYMBOLS[symbol_index % len(SYMBOLS)]
            symbol_index += 1

            try:
                price, signal = compute_signal_for(symbol)
                if bot_state["mode"] == "demo":
                    demo_tick(symbol, price, signal)
            except Exception as e:
                log.warning(f"tick error {symbol}: {e}")
                await asyncio.sleep(2)
                continue

            history = load_history()
            wins = [t for t in history if t.get("profit", 0) > 0]
            open_positions = {s: cs["last_price"] for s, cs in coin_state.items() if cs["position"]}

            coins_data = [
                {
                    "symbol": s,
                    "price": coin_state[s]["last_price"],
                    "signal": coin_state[s]["last_signal"],
                    "position": coin_state[s]["position"],
                    "entry": coin_state[s]["entry"],
                }
                for s in SYMBOLS
            ]

            await websocket.send_json({
                "coins": coins_data,
                "active_symbol": symbol,
                "running": bot_state["running"],
                "mode": bot_state["mode"],
                "demo_balance": round(bot_state["demo_balance"], 2),
                "demo_pnl": round(bot_state["demo_balance"] - DEMO_BALANCE, 2),
                "win_rate": round(len(wins) / len(history), 3) if history else 0,
                "total_trades": len(history),
                "open_count": len(open_positions),
                "time": datetime.utcnow().isoformat(),
            })
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if os.path.exists("dashboard.html"):
        return open("dashboard.html").read()
    return HTMLResponse("<h1>Bot API running</h1>")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
