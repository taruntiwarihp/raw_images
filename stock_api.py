"""
╔══════════════════════════════════════════════════════════════════════╗
║   EvenStocks  ·  Live Market Dashboard  v5.0                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  PERIOD → DATA INTERVAL MAPPING                                      ║
║  1D  → 1-min bars   (today intraday)                                 ║
║  1W  → 5-min bars   (last 7 days)                                    ║
║  1M  → 15-min bars  (last 30 days)                                   ║
║  3M  → 1 trading day (EOD)                                           ║
║  6M  → 1 trading day (EOD)                                           ║
║  1Y  → every 5 trading days (EOD sampled)                            ║
║  5Y  → every 15 trading days (EOD sampled)                           ║
║  ALL → every 15 trading days (EOD sampled)                           ║
║                                                                      ║
║  THEME  : Light (white/soft-blue)                                    ║
║  AUTOSAVE: CSV every 15 min → ./stock_data/                          ║
║                                                                      ║
║  pip install websocket-client dash plotly pandas                     ║
║              dash-bootstrap-components pytz requests                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json, threading, time, logging, sys, os, requests
from collections import deque
from datetime    import datetime, timedelta
from io          import StringIO

for pkg, install in [
    ("websocket","websocket-client"),("pandas","pandas"),
    ("plotly","plotly"),("dash","dash"),("pytz","pytz"),
]:
    try: __import__(pkg)
    except ImportError: sys.exit(f"❌ pip install {install}")

import websocket
import pandas as pd
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════
TRUEDATA_USER     = "Trial198"
TRUEDATA_PASSWORD = "tarun198"

WS_HOST      = "wss://push.truedata.in"
WS_PORT_PROD = 8084
WS_PORT_SAND = 8086
USE_SANDBOX  = True
PORT         = WS_PORT_SAND if USE_SANDBOX else WS_PORT_PROD

AUTH_URL     = "https://auth.truedata.in/token"
HISTORY_URL  = "https://history.truedata.in"

DEFAULT_SYMBOL   = "RELIANCE" # INFY #TMCV
MAX_TICKS        = 10_000
DASH_PORT        = 8050
CSV_DIR          = "stock_data"
CSV_SAVE_MINUTES = 15

IST          = pytz.timezone("Asia/Kolkata")
MARKET_OPEN  = (9,  15)
MARKET_CLOSE = (15, 30)

# Period config: (days_back, rest_interval, eod_sample_step)
#   days_back      = how many calendar days to cut from full history
#   rest_interval  = TrueData REST interval string for intraday periods
#   eod_sample_step= for EOD periods: keep every Nth row (1=all, 5=every 5th trading day)
PERIOD_CFG = {
    "1D":  (1,   "1min",  1),
    "1W":  (7,   "5min",  1),
    "1M":  (30,  "15min", 1),
    "3M":  (90,  "eod",   1),
    "6M":  (180, "eod",   1),
    "1Y":  (365, "eod",   5),
    "5Y":  (1825,"eod",   15),
    "ALL": (None,"eod",   15),
}

# ════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("EvenStocks")

# ════════════════════════════════════════════════════════════════════════════
#  MARKET HOURS
# ════════════════════════════════════════════════════════════════════════════
def now_ist():
    return datetime.now(IST)

def is_market_open():
    n = now_ist()
    if n.weekday() >= 5: return False
    return MARKET_OPEN <= (n.hour, n.minute) <= MARKET_CLOSE

def market_status_str():
    n   = now_ist()
    day = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][n.weekday()]
    t   = n.strftime("%H:%M")
    if n.weekday() >= 5: return f"Weekend ({day} {t} IST)"
    tt  = (n.hour, n.minute)
    if tt < MARKET_OPEN:  return f"Pre-Market ({t} IST) — opens 09:15"
    if tt > MARKET_CLOSE: return f"Market Closed ({t} IST)"
    return f"Market LIVE ({t} IST)"

def seconds_to_next_open():
    n = now_ist()
    c = n.replace(hour=9, minute=15, second=0, microsecond=0)
    if n >= c or n.weekday() >= 5: c += timedelta(days=1)
    while c.weekday() >= 5: c += timedelta(days=1)
    return max(0, int((c - n).total_seconds()))

# ════════════════════════════════════════════════════════════════════════════
#  REST AUTH
# ════════════════════════════════════════════════════════════════════════════
_tok = {"token": None, "expires_at": None}

def get_auth_token():
    now = datetime.utcnow()
    if _tok["token"] and _tok["expires_at"] and now < _tok["expires_at"] - timedelta(seconds=60):
        return _tok["token"]
    log.info("Auth REST …")
    r = requests.post(AUTH_URL, data={"username": TRUEDATA_USER,
                                      "password": TRUEDATA_PASSWORD,
                                      "grant_type": "password"}, timeout=20)
    if r.status_code != 200: raise RuntimeError(f"Auth failed: {r.text}")
    d = r.json()
    if "access_token" not in d: raise RuntimeError(str(d))
    _tok["token"]      = d["access_token"]
    _tok["expires_at"] = datetime.utcnow() + timedelta(seconds=int(d.get("expires_in", 3600)))
    log.info("Token OK ✓")
    return _tok["token"]

# ════════════════════════════════════════════════════════════════════════════
#  REST FETCHERS
# ════════════════════════════════════════════════════════════════════════════
def _rest_get(ep: str) -> pd.DataFrame:
    token = get_auth_token()
    url   = f"{HISTORY_URL}/{ep}"
    log.info(f"REST → {url[:110]}")
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    if r.status_code == 401: raise PermissionError("REST auth denied")
    if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    text = r.text.strip()
    if not text or "No data exists" in text or "does not exist" in text.lower():
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(text), parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log.warning(f"CSV parse: {e}")
        return pd.DataFrame()


def fetch_eod_history(symbol: str) -> pd.DataFrame:
    """Full EOD history from listing to yesterday — used for 3M+ periods."""
    yesterday = (now_ist() - timedelta(days=1)).strftime("%y%m%dT23:59:59")
    df = _rest_get(f"getbars?symbol={symbol}&from=000101T00:00:00"
                   f"&to={yesterday}&response=csv&interval=eod")
    if not df.empty:
        df["source"] = "history"
        if "volume" not in df.columns: df["volume"] = 0
        log.info(f"EOD: {len(df)} rows ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    return df


# Approximate trading bars needed per period
# 1W  5-min:  5 days  × 26 bars/day ≈ 130 bars
# 1M  15-min: 22 days × 26 bars/day ≈ 572 bars  (use getbars date range)
_PERIOD_BARS = {"1W": 150, "1M": 600}


def fetch_intraday_for_period(symbol: str, period: str) -> pd.DataFrame:
    """
    Fetch intraday bars for 1W / 1M periods.

    Strategy:
      1W  → getlastnbars (last 150 5-min bars) — fast and accurate
      1M  → getbars with explicit 35-day date range of 15-min bars
            (TrueData stores ≥30 days of 15-min data via date range query)

    For 1M we add 5 extra calendar days to the request window so weekends
    don't eat into the 30-trading-day window.
    """
    days_back, interval, _ = PERIOD_CFG[period]
    if interval == "eod":
        return pd.DataFrame()

    now_t = now_ist()

    # ── 1W: getlastnbars is the simplest / most reliable ─────────────────
    if period == "1W":
        nbars = _PERIOD_BARS.get(period, 150)
        df = _rest_get(
            f"getlastnbars?symbol={symbol}&nbars={nbars}"
            f"&response=csv&interval={interval}&bidask=0"
        )
        if not df.empty:
            df["source"] = "intraday"
            if "volume" not in df.columns: df["volume"] = 0
            # Trim to the last 7 calendar days
            cutoff = pd.Timestamp(now_t.date()) - pd.Timedelta(days=days_back)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
            log.info(f"1W/5min: {len(df)} bars (last 7 days)")
        return df

    # ── 1M: getbars with a wide date range (35 calendar days back) ────────
    if period == "1M":
        # Add 5 extra days to cover weekends and ensure full 30-day window
        back     = days_back + 5          # = 35 calendar days
        from_dt  = now_t - timedelta(days=back)
        from_str = from_dt.strftime("%y%m%dT09:15:00")
        to_str   = now_t.strftime("%y%m%dT15:30:00")

        df = _rest_get(
            f"getbars?symbol={symbol}&from={from_str}&to={to_str}"
            f"&response=csv&interval={interval}"
        )
        if not df.empty:
            df["source"] = "intraday"
            if "volume" not in df.columns: df["volume"] = 0
            # Trim to exactly 30 calendar days
            cutoff = pd.Timestamp(now_t.date()) - pd.Timedelta(days=days_back)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
            log.info(f"1M/15min: {len(df)} bars ({from_dt.date()} → {now_t.date()})")
        else:
            # Fallback: use getlastnbars(200) if date range returns nothing
            log.warning("1M getbars empty — falling back to getlastnbars(200)")
            df = _rest_get(
                f"getlastnbars?symbol={symbol}&nbars=200"
                f"&response=csv&interval={interval}&bidask=0"
            )
            if not df.empty:
                df["source"] = "intraday"
                if "volume" not in df.columns: df["volume"] = 0
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    return pd.DataFrame()


def _last_trading_day() -> "datetime.date":
    """
    Return the most recent trading day date.
    Before 8 AM IST on a weekday, or any time on weekend → previous weekday.
    During / after market on a weekday → today.
    """
    n = now_ist()
    # If weekend → step back to Friday
    if n.weekday() >= 5:
        days_back = n.weekday() - 4   # Sat→1, Sun→2
        return (n - timedelta(days=days_back)).date()
    # Weekday but before 8:00 AM → use yesterday (or Friday if Monday)
    if (n.hour, n.minute) < (8, 0):
        prev = n - timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        return prev.date()
    return n.date()


def fetch_today_intraday(symbol: str) -> pd.DataFrame:
    """
    Fetch 1-min bars for the last trading day (9:15 → 15:30).
    Shows today if market is open / after close, yesterday if before 8 AM.
    """
    tday   = _last_trading_day()
    open_t = tday.strftime("%y%m%dT09:15:00")
    # If tday == today and market is still open, fetch up to now
    n      = now_ist()
    if tday == n.date() and (n.hour, n.minute) <= (15, 30):
        close_t = n.strftime("%y%m%dT%H:%M:%S")
    else:
        close_t = tday.strftime("%y%m%dT15:30:00")
    df = _rest_get(f"getbars?symbol={symbol}&from={open_t}&to={close_t}"
                   f"&response=csv&interval=1min")
    if not df.empty:
        df["source"] = "today_rest"
        if "volume" not in df.columns: df["volume"] = 0
        log.info(f"1D data: {len(df)} 1-min bars for {tday}")
    return df

# ════════════════════════════════════════════════════════════════════════════
#  PERIOD DATA HELPER — resample / filter merged data for display
# ════════════════════════════════════════════════════════════════════════════
def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV dataframe to a coarser time rule (e.g. '5min')."""
    if df.empty: return df
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).set_index("timestamp")
    agg = d["close"].resample(rule).ohlc()
    agg["volume"] = d["volume"].resample(rule).sum()
    agg.dropna(subset=["open"], inplace=True)
    agg.reset_index(inplace=True)
    agg["source"] = df["source"].iloc[0] if "source" in df.columns else "history"
    return agg


def get_display_df(merged_eod: pd.DataFrame,
                   intraday_df: pd.DataFrame,
                   live_ticks_df: pd.DataFrame,
                   period: str) -> pd.DataFrame:
    """
    Return the correctly-intervalled DataFrame for the chosen period:

    1D  → today 1-min REST  +  live ticks (1-min aggregated)
    1W  → 5-min intraday
    1M  → 15-min intraday
    3M  → EOD (all trading days, ~63 rows)
    6M  → EOD (all trading days, ~125 rows)
    1Y  → EOD every 5 trading days
    5Y  → EOD every 15 trading days
    ALL → EOD every 15 trading days
    """
    days_back, interval, step = PERIOD_CFG[period]

    # ── 1D: today's 1-min REST bars + live WS ticks ───────────────────────
    if period == "1D":
        frames = []
        if not intraday_df.empty:
            frames.append(intraday_df)
        if not live_ticks_df.empty and "ltp" in live_ticks_df.columns:
            lt = live_ticks_df.copy()
            lt["ts"] = pd.to_datetime(lt["timestamp"], errors="coerce")
            lt.dropna(subset=["ts"], inplace=True)
            if not lt.empty:
                lt.sort_values("ts", inplace=True)
                idx = lt.set_index("ts")
                agg = idx["ltp"].resample("1min").ohlc()
                vc  = idx["tick_vol"] if "tick_vol" in idx.columns else pd.Series(0, index=idx.index)
                agg["volume"] = vc.resample("1min").sum()
                agg.dropna(subset=["open"], inplace=True)
                agg.reset_index(inplace=True)
                agg.rename(columns={"ts": "timestamp"}, inplace=True)
                agg["source"] = "live"
                frames.append(agg[["timestamp","open","high","low","close","volume","source"]])
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out.dropna(subset=["timestamp"], inplace=True)
        out.sort_values("timestamp", inplace=True)
        out.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        return out.reset_index(drop=True)

    # ── 1W / 1M: intraday data already at the right interval ─────────────
    if period in ("1W", "1M"):
        if not intraday_df.empty:
            return intraday_df.reset_index(drop=True)
        return pd.DataFrame()

    # ── 3M / 6M: full EOD, filter by date ────────────────────────────────
    if period in ("3M", "6M"):
        if merged_eod.empty: return pd.DataFrame()
        cutoff = pd.Timestamp(now_ist().date()) - pd.Timedelta(days=days_back)
        out = merged_eod[merged_eod["timestamp"] >= cutoff].copy()
        return out.reset_index(drop=True)

    # ── 1Y: EOD, every 5th trading day ───────────────────────────────────
    if period == "1Y":
        if merged_eod.empty: return pd.DataFrame()
        cutoff = pd.Timestamp(now_ist().date()) - pd.Timedelta(days=days_back)
        out = merged_eod[merged_eod["timestamp"] >= cutoff].copy()
        # keep every `step`-th row to show one point per 5 trading days
        out = out.iloc[::step].reset_index(drop=True)
        return out

    # ── 5Y / ALL: EOD, every 15th trading day ────────────────────────────
    if period in ("5Y", "ALL"):
        if merged_eod.empty: return pd.DataFrame()
        if days_back:
            cutoff = pd.Timestamp(now_ist().date()) - pd.Timedelta(days=days_back)
            out = merged_eod[merged_eod["timestamp"] >= cutoff].copy()
        else:
            out = merged_eod.copy()
        out = out.iloc[::step].reset_index(drop=True)
        return out

    return pd.DataFrame()

# ════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ════════════════════════════════════════════════════════════════════════════
class LiveStore:
    def __init__(self):
        self._lock           = threading.Lock()
        self.hist_eod_df     = pd.DataFrame()    # full EOD (all history)
        self.today_rest_df   = pd.DataFrame()    # today's 1-min REST
        self.period_intraday = {}                # {period: df}  — 1W/1M cache
        self.live_ticks      = deque(maxlen=MAX_TICKS)
        self.last_tick       = {}
        self.symbol_map      = {}
        self.symbol          = DEFAULT_SYMBOL
        self.ws_connected    = False
        self.hist_loaded     = False
        self.status_msg      = "Starting …"
        self.last_csv_save   = datetime.min
        self.active_period   = "1Y"              # track what user selected

    # ── writers ──────────────────────────────────────────────────────────
    def load_history(self, eod_df, today_df):
        with self._lock:
            self.hist_eod_df   = eod_df
            self.today_rest_df = today_df
            self.hist_loaded   = True

    def cache_period_intraday(self, period: str, df: pd.DataFrame):
        with self._lock:
            self.period_intraday[period] = df

    def push_tick(self, tick):
        with self._lock:
            self.live_ticks.append(tick)
            self.last_tick = tick

    def register_symbol(self, sid, name):
        with self._lock: self.symbol_map[sid] = name

    def set_ws_status(self, conn, msg):
        with self._lock:
            self.ws_connected = conn
            self.status_msg   = msg

    def clear_live(self):
        with self._lock:
            self.live_ticks.clear()
            self.last_tick = {}

    # ── readers ──────────────────────────────────────────────────────────
    def get_last(self):
        with self._lock: return dict(self.last_tick)

    def get_status(self):
        with self._lock: return self.ws_connected, self.status_msg, self.hist_loaded

    def get_display_df_for_period(self, period: str) -> pd.DataFrame:
        """Thread-safe access to the right data for the given period."""
        with self._lock:
            eod   = self.hist_eod_df.copy()   if not self.hist_eod_df.empty   else pd.DataFrame()
            today = self.today_rest_df.copy()  if not self.today_rest_df.empty else pd.DataFrame()
            intra = self.period_intraday.get(period, pd.DataFrame()).copy()
            ticks = pd.DataFrame(list(self.live_ticks)) if self.live_ticks else pd.DataFrame()
            if not ticks.empty:
                ticks["ts"] = pd.to_datetime(ticks["timestamp"], errors="coerce")

        # 1D uses today_rest as the intraday source
        if period == "1D":
            return get_display_df(eod, today, ticks, period)
        return get_display_df(eod, intra, ticks, period)

    def get_merged_eod(self) -> pd.DataFrame:
        """Full EOD merged (used for MA calculation on ALL history)."""
        with self._lock:
            return self.hist_eod_df.copy() if not self.hist_eod_df.empty else pd.DataFrame()

    def get_today_ticks_df(self):
        with self._lock:
            if not self.live_ticks: return pd.DataFrame()
            df = pd.DataFrame(list(self.live_ticks))
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    def get_metrics_from_history(self):
        """Fallback metrics when no live tick is present."""
        with self._lock:
            today = now_ist().date()
            for df in [self.today_rest_df, self.hist_eod_df]:
                if df.empty: continue
                d = df.copy()
                d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
                tr = d[d["timestamp"].dt.date == today]
                if not tr.empty:
                    prev_close = float(self.hist_eod_df["close"].iloc[-1]) \
                                 if not self.hist_eod_df.empty else 0
                    return {"ltp": float(tr["close"].iloc[-1]),
                            "open": float(tr["open"].iloc[0]),
                            "high": float(tr["high"].max()),
                            "low":  float(tr["low"].min()),
                            "volume": float(tr["volume"].sum()),
                            "prev_close": prev_close}
            if not self.hist_eod_df.empty:
                row = self.hist_eod_df.iloc[-1]
                pc  = float(self.hist_eod_df["close"].iloc[-2]) \
                      if len(self.hist_eod_df) >= 2 else 0
                return {"ltp": float(row["close"]), "open": float(row["open"]),
                        "high": float(row["high"]),  "low":  float(row["low"]),
                        "volume": float(row.get("volume",0)), "prev_close": pc}
            return {}

    def needs_csv_save(self):
        with self._lock:
            return (datetime.utcnow() - self.last_csv_save).total_seconds() >= CSV_SAVE_MINUTES*60

    def mark_csv_saved(self):
        with self._lock: self.last_csv_save = datetime.utcnow()


STORE = LiveStore()

# ════════════════════════════════════════════════════════════════════════════
#  BACKGROUND THREADS
# ════════════════════════════════════════════════════════════════════════════
def history_load_loop(symbol: str):
    """Load EOD + today intraday at startup, refresh every 5 min."""
    while True:
        try:
            STORE.set_ws_status(STORE.ws_connected, f"Loading {symbol} …")
            eod   = fetch_eod_history(symbol)
            today = fetch_today_intraday(symbol)
            STORE.load_history(eod, today)
            # Pre-fetch intraday for default period if it's a short one
            ap = STORE.active_period
            if ap in ("1W", "1M"):
                _fetch_and_cache_intraday(symbol, ap)
        except Exception as e:
            log.error(f"History error: {e}")
        time.sleep(300)


def _fetch_and_cache_intraday(symbol: str, period: str):
    """Fetch intraday bars for the given period and cache in STORE."""
    try:
        df = fetch_intraday_for_period(symbol, period)
        STORE.cache_period_intraday(period, df)
        log.info(f"Period cache updated: {period} → {len(df)} rows")
    except Exception as e:
        log.error(f"Intraday fetch {period}: {e}")


def csv_autosave_loop():
    os.makedirs(CSV_DIR, exist_ok=True)
    while True:
        time.sleep(30)
        if not (STORE.needs_csv_save() and STORE.hist_loaded): continue
        try:
            m = STORE.get_merged_eod()
            if m.empty: continue
            sym  = STORE.symbol.replace("-","_").replace(" ","_")
            ts   = now_ist().strftime("%Y%m%d_%H%M")
            m.to_csv(os.path.join(CSV_DIR, f"{sym}_merged_{ts}.csv"),  index=False)
            m.to_csv(os.path.join(CSV_DIR, f"{sym}_latest.csv"),        index=False)
            STORE.mark_csv_saved()
            log.info(f"💾 CSV saved ({len(m)} rows)")
        except Exception as e:
            log.error(f"CSV error: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET
# ════════════════════════════════════════════════════════════════════════════
class EvenStocksWS:
    def __init__(self, symbol):
        self.symbol  = symbol.upper()
        self._ws     = None
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self): self._thread.start()

    def stop(self):
        self._stop.set()
        if self._ws:
            try: self._ws.close()
            except: pass

    def change_symbol(self, new):
        old = self.symbol; self.symbol = new.upper()
        if self._ws:
            self._send({"method":"removesymbol","symbols":[old]})
            time.sleep(0.3)
            self._send({"method":"addsymbol","symbols":[self.symbol]})

    def _send(self, p):
        try:
            if self._ws: self._ws.send(json.dumps(p))
        except Exception as e: log.warning(f"WS send: {e}")

    def _run_loop(self):
        while not self._stop.is_set():
            if not is_market_open():
                STORE.set_ws_status(False, market_status_str())
                time.sleep(min(seconds_to_next_open(), 60))
                continue
            try: self._connect()
            except Exception as e: log.error(f"WS: {e}")
            if not self._stop.is_set() and is_market_open():
                STORE.set_ws_status(False, "Reconnecting …"); time.sleep(5)

    def _connect(self):
        url = f"{WS_HOST}:{PORT}?user={TRUEDATA_USER}&password={TRUEDATA_PASSWORD}"
        STORE.set_ws_status(False, "Connecting …")
        ws = websocket.WebSocketApp(url,
            on_open=self._on_open, on_message=self._on_message,
            on_error=self._on_error, on_close=self._on_close)
        self._ws = ws
        ws.run_forever(ping_interval=20, ping_timeout=10)

    def _on_open(self, ws): STORE.set_ws_status(False, "Authenticating …")

    def _on_message(self, ws, raw):
        if not is_market_open(): ws.close(); return
        try: msg = json.loads(raw)
        except: return

        if msg.get("message") == "TrueData Real Time Data Service":
            if msg.get("success"):
                STORE.set_ws_status(True, f"Live · {self.symbol}")
                self._send({"method":"addsymbol","symbols":[self.symbol]})
            else:
                STORE.set_ws_status(False, "Auth failed"); ws.close()
            return

        if msg.get("message") in ("HeartBeat","marketstatus"): return

        if msg.get("message") in ("symbols added","touchline"):
            for entry in msg.get("symbollist",[]):
                if len(entry) >= 2:
                    name, sid = entry[0], str(entry[1])
                    STORE.register_symbol(sid, name)
                    if len(entry) >= 11:
                        try:
                            STORE.push_tick({
                                "timestamp": entry[2],
                                "ltp":       float(entry[3]  or 0),
                                "tick_vol":  float(entry[4]  or 0),
                                "atp":       float(entry[5]  or 0),
                                "tot_vol":   float(entry[6]  or 0),
                                "open":      float(entry[7]  or 0),
                                "high":      float(entry[8]  or 0),
                                "low":       float(entry[9]  or 0),
                                "prev_close":float(entry[10] or 0),
                                "symbol": name,
                            })
                        except: pass
            return

        if "trade" in msg:
            t = msg["trade"]
            try:
                STORE.push_tick({
                    "timestamp": t[1],
                    "ltp":       float(t[2]  or 0),
                    "tick_vol":  float(t[3]  or 0),
                    "atp":       float(t[4]  or 0),
                    "tot_vol":   float(t[5]  or 0),
                    "open":      float(t[6]  or 0),
                    "high":      float(t[7]  or 0),
                    "low":       float(t[8]  or 0),
                    "prev_close":float(t[9]  or 0),
                    "oi":        float(t[10] or 0),
                    "symbol":    STORE.symbol_map.get(str(t[0]), str(t[0])),
                })
            except: pass

    def _on_error(self, ws, e): STORE.set_ws_status(False,"WS Error")
    def _on_close(self, ws, c, r): STORE.set_ws_status(False, market_status_str())

# ════════════════════════════════════════════════════════════════════════════
#  LIGHT THEME COLOURS
# ════════════════════════════════════════════════════════════════════════════
C = {
    # ── Layout ─────────────────────────────────────────────────────────────
    "bg":       "#eef2f7",   # soft blue-gray page background
    "card":     "#ffffff",   # white cards
    "card2":    "#f8fafd",   # slightly tinted card (strip, toolbar)
    "border":   "#d0dae8",   # light gray-blue border
    # ── Typography ─────────────────────────────────────────────────────────
    "text":     "#1a2b45",   # dark navy text
    "text_dim": "#6b80a0",   # muted gray-blue
    "muted":    "#9baec8",   # light muted
    # ── Accent & signals ────────────────────────────────────────────────────
    "accent":   "#2563eb",   # EvenStocks blue
    "green":    "#16a34a",   # profit green
    "red":      "#dc2626",   # loss red
    "amber":    "#d97706",   # warning amber
    # ── Chart source colours ────────────────────────────────────────────────
    "hist":     "#3b82f6",   # medium blue  — EOD history
    "intra":    "#0891b2",   # cyan         — intraday REST (1W/1M)
    "today":    "#7c3aed",   # purple       — today REST 1-min
    "live":     "#16a34a",   # green        — live WS
    # ── Moving averages ─────────────────────────────────────────────────────
    "ma50":     "#f59e0b",   # amber/gold
    "ma150":    "#8b5cf6",   # violet
    "ma200":    "#64748b",   # slate
}

# ── Source → (display label, up-colour, down-colour)
SRC_STYLE = {
    "history":    ("EOD History", C["hist"],  "#ef4444"),
    "intraday":   ("Intraday",    C["intra"], "#ef4444"),
    "today_rest": ("Today REST",  C["today"], "#ef4444"),
    "live":       ("Live WS",     C["live"],  C["red"]),
}

BTN = {
    "border":"none","borderRadius":"5px","padding":"5px 13px",
    "fontSize":"11px","fontWeight":"700","cursor":"pointer",
    "fontFamily":"'Courier New',monospace","letterSpacing":"1px",
    "transition":"all 0.15s",
}

def cs(extra=None):
    b = {"background":C["card"],"border":f"1px solid {C['border']}",
         "borderRadius":"8px","padding":"14px 18px",
         "boxShadow":"0 1px 4px rgba(0,0,0,0.06)"}
    if extra: b.update(extra)
    return b

def pbtn(pid, active=False):
    return html.Button(pid, id=f"btn-{pid}", n_clicks=0, style={
        **BTN,
        "background": C["accent"] if active else "transparent",
        "color":      "#fff"       if active else C["text_dim"],
        "border":     f"1px solid {C['accent']}" if active else "1px solid transparent",
    })

def cbtn(label, bid, active=False):
    return html.Button(label, id=bid, n_clicks=0, style={
        **BTN,
        "background": C["green"] if active else "transparent",
        "color":      "#fff"     if active else C["text_dim"],
        "border":     f"1px solid {C['green']}" if active else f"1px solid {C['border']}",
    })

def mabtn(label, bid, col, active=True):
    return html.Button(label, id=bid, n_clicks=0, style={
        **BTN, "fontSize":"10px", "padding":"4px 10px",
        "background": col   if active else "transparent",
        "color":      "#fff" if active else C["text_dim"],
        "border":     f"1px solid {col}",
    })

def mcard(cid, label, col):
    return html.Div(style={**cs(), "flex":"1","minWidth":"118px","textAlign":"center","padding":"12px 8px"}, children=[
        html.Div(label, style={"fontSize":"9px","letterSpacing":"2px","color":C["text_dim"],
                               "textTransform":"uppercase","marginBottom":"5px",
                               "fontFamily":"'Courier New',monospace"}),
        html.Div(id=cid, children="—",
                 style={"fontSize":"20px","fontWeight":"700","color":col,
                        "fontFamily":"'Courier New',monospace","letterSpacing":"1px"}),
    ])

# ════════════════════════════════════════════════════════════════════════════
#  DASH APP LAYOUT
# ════════════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="EvenStocks", update_title=None,
                suppress_callback_exceptions=True)

app.layout = html.Div(
    style={"background":C["bg"],"minHeight":"100vh",
           "fontFamily":"'Courier New',monospace","color":C["text"]},
    children=[

        # ── HEADER ─────────────────────────────────────────────────────────
        html.Div(style={
            "background":C["card"],"borderBottom":f"2px solid {C['accent']}",
            "padding":"10px 24px","display":"flex","alignItems":"center","gap":"14px",
            "boxShadow":"0 2px 8px rgba(37,99,235,0.08)"},
        children=[
            html.Div([
                html.Span("Even",  style={"color":C["accent"],"fontWeight":"900",
                                          "fontSize":"22px","letterSpacing":"-0.5px"}),
                html.Span("Stocks",style={"color":C["text"],"fontWeight":"400",
                                          "fontSize":"22px"}),
                html.Span(" ●",    style={"color":C["green"],"fontSize":"11px","marginLeft":"3px"}),
            ]),
            html.Div(style={"width":"1px","height":"26px","background":C["border"],"margin":"0 4px"}),
            dcc.Input(id="symbol-input", type="text", debounce=True,
                      placeholder="Symbol …", value=DEFAULT_SYMBOL,
                      style={"background":"#f0f4ff","border":f"1.5px solid {C['accent']}",
                             "borderRadius":"5px","color":C["accent"],"padding":"6px 12px",
                             "fontSize":"14px","fontFamily":"'Courier New',monospace",
                             "fontWeight":"700","width":"145px","outline":"none"}),
            html.Button("▶ LOAD", id="stream-btn", n_clicks=0, style={
                **BTN,"background":C["accent"],"color":"#fff",
                "padding":"7px 18px","fontSize":"12px","borderRadius":"5px"}),
            html.Div(style={"flex":"1"}),
            html.Div(id="mkt-badge",
                     style={"fontSize":"11px","color":C["amber"],"letterSpacing":"1px",
                            "marginRight":"12px","fontWeight":"600"}),
            html.Div(style={"display":"flex","alignItems":"center","gap":"7px"}, children=[
                html.Div(id="status-dot",
                         style={"width":"9px","height":"9px","borderRadius":"50%",
                                "background":C["red"]}),
                html.Span(id="status-text", children="Initialising …",
                          style={"fontSize":"11px","color":C["text_dim"]}),
            ]),
            html.Div(id="clock",
                     style={"fontSize":"12px","color":C["muted"],
                            "minWidth":"90px","textAlign":"right"}),
        ]),

        # ── DATA STRIP ─────────────────────────────────────────────────────
        html.Div(id="data-strip",
                 style={"background":C["card2"],"borderBottom":f"1px solid {C['border']}",
                        "padding":"5px 24px","fontSize":"10px","color":C["text_dim"],
                        "display":"flex","gap":"18px","flexWrap":"wrap"}),

        # ── METRICS ────────────────────────────────────────────────────────
        html.Div(style={"display":"flex","gap":"8px","padding":"10px 24px","flexWrap":"wrap"},
            children=[
                mcard("m-ltp",  "LTP ₹",       C["accent"]),
                mcard("m-chg",  "Change",        C["text"]),
                mcard("m-open", "Open ₹",        C["text_dim"]),
                mcard("m-high", "Day High ₹",    C["green"]),
                mcard("m-low",  "Day Low ₹",     C["red"]),
                mcard("m-vol",  "Volume",          C["amber"]),
                mcard("m-prev", "Prev Close ₹",  C["muted"]),
            ]),

        # ── TOOLBAR ────────────────────────────────────────────────────────
        html.Div(style={
            "display":"flex","alignItems":"center","gap":"8px",
            "padding":"0 24px 10px","flexWrap":"wrap"},
        children=[
            # Period selector
            html.Div(style={"display":"flex","gap":"2px","background":C["card2"],
                            "border":f"1px solid {C['border']}","borderRadius":"7px",
                            "padding":"3px"},
                children=[pbtn(p, p=="1Y") for p in PERIOD_CFG]),

            html.Div(style={"width":"1px","height":"28px","background":C["border"]}),

            # Chart type
            html.Div(style={"display":"flex","gap":"2px","background":C["card2"],
                            "border":f"1px solid {C['border']}","borderRadius":"7px",
                            "padding":"3px"},
                children=[
                    cbtn("🕯 Candle","btn-candle", active=True),
                    cbtn("📈 Line",  "btn-line",   active=False),
                ]),

            html.Div(style={"width":"1px","height":"28px","background":C["border"]}),

            # MA toggles
            html.Div(style={"display":"flex","gap":"4px","alignItems":"center"},
                children=[
                    html.Span("MA:", style={"fontSize":"10px","color":C["text_dim"],
                                            "fontWeight":"600","letterSpacing":"1px"}),
                    mabtn("50 DMA",  "btn-ma50",  C["ma50"],  True),
                    mabtn("150 DMA", "btn-ma150", C["ma150"], False),
                    mabtn("200 DMA", "btn-ma200", C["ma200"], True),
                ]),

            html.Div(style={"flex":"1"}),

            # Interval label (updated dynamically)
            html.Div(id="interval-label",
                     style={"fontSize":"10px","color":C["text_dim"],"letterSpacing":"1px",
                            "background":C["card2"],"border":f"1px solid {C['border']}",
                            "borderRadius":"5px","padding":"4px 10px"}),
        ]),

        # ── CHARTS ROW ─────────────────────────────────────────────────────
        html.Div(style={"display":"flex","gap":"10px","padding":"0 24px 14px"}, children=[

            # Main chart
            html.Div(style={**cs(),"flex":"3"}, children=[
                html.Div(style={"display":"flex","justifyContent":"space-between",
                                "alignItems":"center","marginBottom":"8px"}, children=[
                    html.Div(id="chart-title",
                             style={"fontSize":"10px","letterSpacing":"2px",
                                    "color":C["text_dim"],"fontWeight":"600"}),
                    html.Div(style={"display":"flex","gap":"12px","fontSize":"10px"}, children=[
                        html.Span("▬ History", style={"color":C["hist"]}),
                        html.Span("▬ Intraday",style={"color":C["intra"]}),
                        html.Span("▬ Live",    style={"color":C["live"]}),
                        html.Span("╌ 50D",     style={"color":C["ma50"]}),
                        html.Span("╌ 150D",    style={"color":C["ma150"]}),
                        html.Span("╌ 200D",    style={"color":C["ma200"]}),
                    ]),
                ]),
                dcc.Graph(id="main-chart",
                          config={"displayModeBar":True,
                                  "modeBarButtonsToRemove":["lasso2d","select2d"]},
                          style={"height":"460px"}),
            ]),

            # Right: tick stream
            html.Div(style={**cs(),"flex":"1.1"}, children=[
                html.Div("TODAY  ·  TICK  STREAM",
                         style={"fontSize":"10px","letterSpacing":"3px",
                                "color":C["text_dim"],"marginBottom":"8px","fontWeight":"600"}),
                dcc.Graph(id="tick-chart", config={"displayModeBar":False},
                          style={"height":"220px"}),
                html.Div(style={"marginTop":"10px","borderTop":f"1px solid {C['border']}",
                                "paddingTop":"10px"}, children=[
                    html.Div("RECENT TICKS",
                             style={"fontSize":"9px","letterSpacing":"3px",
                                    "color":C["text_dim"],"marginBottom":"6px","fontWeight":"600"}),
                    html.Div(id="tick-table",
                             style={"fontSize":"11px","lineHeight":"2","color":C["text_dim"]}),
                    html.Div(id="csv-info",
                             style={"marginTop":"10px","fontSize":"10px","color":C["muted"]}),
                ]),
            ]),
        ]),

        dcc.Interval(id="interval", interval=1_000, n_intervals=0),
        dcc.Store(id="active-symbol",  data=DEFAULT_SYMBOL),
        dcc.Store(id="active-period",  data="1Y"),
        dcc.Store(id="active-chart",   data="candle"),
        dcc.Store(id="active-ma50",    data=True),
        dcc.Store(id="active-ma150",   data=False),
        dcc.Store(id="active-ma200",   data=True),
    ],
)

# ════════════════════════════════════════════════════════════════════════════
#  WS SINGLETON
# ════════════════════════════════════════════════════════════════════════════
_ws_client = None

def get_or_start_ws(symbol):
    global _ws_client
    if _ws_client is None:
        _ws_client = EvenStocksWS(symbol)
        _ws_client.start()
    return _ws_client

# ════════════════════════════════════════════════════════════════════════════
#  STORE + PERIOD BUTTON CALLBACKS
# ════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("active-period","data"),
    [Input(f"btn-{p}","n_clicks") for p in PERIOD_CFG],
    prevent_initial_call=True,
)
def set_period(*_):
    from dash import callback_context as ctx
    if not ctx.triggered: return "1Y"
    pid = ctx.triggered[0]["prop_id"].split(".")[0].replace("btn-","")
    STORE.active_period = pid
    # Trigger background fetch for short intraday periods
    if pid in ("1W","1M"):
        threading.Thread(target=_fetch_and_cache_intraday,
                         args=(STORE.symbol, pid), daemon=True).start()
    return pid

@app.callback(
    Output("active-chart","data"),
    Input("btn-candle","n_clicks"), Input("btn-line","n_clicks"),
    prevent_initial_call=True,
)
def set_chart(*_):
    from dash import callback_context as ctx
    if not ctx.triggered: return "candle"
    return "candle" if "candle" in ctx.triggered[0]["prop_id"] else "line"

@app.callback(Output("active-ma50","data"),  Input("btn-ma50","n_clicks"),
              State("active-ma50","data"),   prevent_initial_call=True)
def tog_ma50(n, v):  return not v

@app.callback(Output("active-ma150","data"), Input("btn-ma150","n_clicks"),
              State("active-ma150","data"),  prevent_initial_call=True)
def tog_ma150(n, v): return not v

@app.callback(Output("active-ma200","data"), Input("btn-ma200","n_clicks"),
              State("active-ma200","data"),  prevent_initial_call=True)
def tog_ma200(n, v): return not v

# Period button styles
@app.callback([Output(f"btn-{p}","style") for p in PERIOD_CFG],
              Input("active-period","data"))
def sync_period_btns(active):
    return [{**BTN,
             "background": C["accent"] if p==active else "transparent",
             "color":      "#fff"       if p==active else C["text_dim"],
             "border":     f"1px solid {C['accent']}" if p==active else "1px solid transparent",
             } for p in PERIOD_CFG]

@app.callback(Output("btn-candle","style"), Output("btn-line","style"),
              Input("active-chart","data"))
def sync_chart_btns(ct):
    on  = {**BTN,"background":C["green"],"color":"#fff","border":f"1px solid {C['green']}"}
    off = {**BTN,"background":"transparent","color":C["text_dim"],"border":f"1px solid {C['border']}"}
    return (on,off) if ct=="candle" else (off,on)

@app.callback(Output("btn-ma50","style"),  Input("active-ma50","data"))
def s50(a):  return {**BTN,"fontSize":"10px","padding":"4px 10px",
                     "background":C["ma50"] if a else "transparent",
                     "color":"#fff" if a else C["text_dim"],"border":f"1px solid {C['ma50']}"}

@app.callback(Output("btn-ma150","style"), Input("active-ma150","data"))
def s150(a): return {**BTN,"fontSize":"10px","padding":"4px 10px",
                     "background":C["ma150"] if a else "transparent",
                     "color":"#fff" if a else C["text_dim"],"border":f"1px solid {C['ma150']}"}

@app.callback(Output("btn-ma200","style"), Input("active-ma200","data"))
def s200(a): return {**BTN,"fontSize":"10px","padding":"4px 10px",
                     "background":C["ma200"] if a else "transparent",
                     "color":"#fff" if a else C["text_dim"],"border":f"1px solid {C['ma200']}"}

@app.callback(Output("active-symbol","data"),
              Input("stream-btn","n_clicks"),
              State("symbol-input","value"),
              prevent_initial_call=False)
def on_stream(n, symbol):
    sym = (symbol or DEFAULT_SYMBOL).strip().upper()
    ws  = get_or_start_ws(sym)
    if n and n > 0:
        STORE.clear_live()
        STORE.hist_loaded = False
        STORE.symbol = sym
        ws.change_symbol(sym)
        threading.Thread(target=history_load_loop, args=(sym,), daemon=True).start()
    return sym

# ════════════════════════════════════════════════════════════════════════════
#  MAIN LIVE UPDATE CALLBACK
# ════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("status-dot",    "style"),
    Output("status-text",   "children"),
    Output("clock",         "children"),
    Output("mkt-badge",     "children"),
    Output("data-strip",    "children"),
    Output("chart-title",   "children"),
    Output("interval-label","children"),
    Output("m-ltp",  "children"), Output("m-ltp",  "style"),
    Output("m-chg",  "children"), Output("m-chg",  "style"),
    Output("m-open", "children"),
    Output("m-high", "children"),
    Output("m-low",  "children"),
    Output("m-vol",  "children"),
    Output("m-prev", "children"),
    Output("main-chart", "figure"),
    Output("tick-chart", "figure"),
    Output("tick-table", "children"),
    Output("csv-info",   "children"),

    Input("interval",       "n_intervals"),
    State("active-symbol",  "data"),
    State("active-period",  "data"),
    State("active-chart",   "data"),
    State("active-ma50",    "data"),
    State("active-ma150",   "data"),
    State("active-ma200",   "data"),
)
def live_update(n, symbol, period, chart_type, ma50, ma150, ma200):
    sym = symbol or DEFAULT_SYMBOL
    ws_conn, ws_msg, hist_loaded = STORE.get_status()

    # ── Status ────────────────────────────────────────────────────────────
    dot_col = C["green"] if ws_conn else (C["amber"] if is_market_open() else C["muted"])
    dot_sty = {"width":"9px","height":"9px","borderRadius":"50%","background":dot_col,
               "boxShadow":f"0 0 7px {dot_col}" if ws_conn else "none"}
    clock   = now_ist().strftime("%H:%M:%S IST")

    # ── Interval label ────────────────────────────────────────────────────
    _, interval_str, step = PERIOD_CFG.get(period, (None,"eod",1))
    if step > 1:
        ilabel = f"Interval: every {step} trading days"
    else:
        ilabel = f"Interval: {interval_str}"

    # ── Data strip ────────────────────────────────────────────────────────
    n_eod  = len(STORE.hist_eod_df)
    n_live = len(STORE.live_ticks)
    disp_df = STORE.get_display_df_for_period(period)
    strip = [
        html.Span(f"📅 EOD: {n_eod:,}d",   style={"color":C["hist"]}),    html.Span("  |  "),
        html.Span(f"⚡ Live: {n_live:,}",   style={"color":C["live"]}),    html.Span("  |  "),
        html.Span(f"📊 Shown: {len(disp_df):,} pts", style={"color":C["accent"]}),html.Span("  |  "),
        html.Span("✔ Ready" if hist_loaded else "⟳ Loading …",
                  style={"color":C["green"] if hist_loaded else C["amber"],"fontWeight":"600"}),
    ]

    chart_title = f"{sym}  ·  {period}  ·  {'CANDLE' if chart_type=='candle' else 'LINE'}  ·  {ilabel}"

    # ── Metrics ───────────────────────────────────────────────────────────
    msty = {"fontSize":"20px","fontWeight":"700",
            "fontFamily":"'Courier New',monospace","letterSpacing":"1px"}

    def fn(v, p=2):
        try:    return f"{float(v):,.{p}f}" if v and float(v)!=0 else "—"
        except: return "—"

    def fvol(v):
        try:
            v = float(v)
            if v >= 1e7: return f"{v/1e7:.2f}Cr"
            if v >= 1e5: return f"{v/1e5:.2f}L"
            return f"{v:,.0f}"
        except: return "—"

    last    = STORE.get_last()
    ltp     = float(last.get("ltp",       0) or 0)
    prev_cl = float(last.get("prev_close",0) or 0)
    op      = float(last.get("open",      0) or 0)
    high    = float(last.get("high",      0) or 0)
    low     = float(last.get("low",       0) or 0)
    tot_vol = float(last.get("tot_vol",   0) or 0)

    # Fallback: populate from history when market is closed
    if not ltp and hist_loaded:
        fb = STORE.get_metrics_from_history()
        if fb:
            ltp     = fb.get("ltp",       ltp)
            op      = fb.get("open",      op)
            high    = fb.get("high",      high)
            low     = fb.get("low",       low)
            tot_vol = fb.get("volume",    tot_vol)
            prev_cl = fb.get("prev_close",prev_cl)

    if not prev_cl and not STORE.hist_eod_df.empty:
        eod = STORE.hist_eod_df
        prev_cl = float(eod.iloc[-1]["close"] or 0)

    chg     = ltp - prev_cl if prev_cl else 0
    chg_pct = chg / prev_cl * 100 if prev_cl else 0
    chg_str = f"{chg:+,.2f}  ({chg_pct:+.2f}%)" if ltp else "—"
    chg_col = C["green"] if chg >= 0 else C["red"]
    ltp_col = C["accent"] if ltp else C["text_dim"]

    # ── Auto-trigger intraday cache fetch if needed ─────────────────────
    if period in ("1W","1M") and STORE.period_intraday.get(period, pd.DataFrame()).empty:
        threading.Thread(target=_fetch_and_cache_intraday,
                         args=(sym, period), daemon=True).start()

    # ── Charts ────────────────────────────────────────────────────────────
    ma_cfg = {"ma50":ma50,"ma150":ma150,"ma200":ma200}
    main_fig = _make_main_fig(disp_df, STORE.get_merged_eod(), sym, period, chart_type, ma_cfg)
    tick_fig = _make_tick_fig(STORE.get_today_ticks_df())

    # ── Tick table ────────────────────────────────────────────────────────
    tick_rows = []
    tdf = STORE.get_today_ticks_df()
    if not tdf.empty:
        for _, row in tdf.tail(8).iloc[::-1].iterrows():
            p  = row.get("ltp",0); ts = str(row.get("timestamp",""))[-8:]
            cl = C["green"] if p >= (row.get("prev_close") or p) else C["red"]
            tick_rows.append(html.Div([
                html.Span(ts,    style={"color":C["muted"],"marginRight":"10px"}),
                html.Span(fn(p), style={"color":cl,"fontWeight":"700"}),
                html.Span(f"  v:{fn(row.get('tick_vol',0),0)}",
                          style={"color":C["muted"],"fontSize":"10px"}),
            ]))

    # ── CSV info ──────────────────────────────────────────────────────────
    el  = (datetime.utcnow() - STORE.last_csv_save).total_seconds()
    nxt = max(0, int(CSV_SAVE_MINUTES*60 - el))
    csv_txt = (f"⏳ First save in {nxt}s" if STORE.last_csv_save == datetime.min
               else f"💾 Saved {int(el)}s ago · next {nxt}s")

    return (
        dot_sty, ws_msg, clock, market_status_str(), strip, chart_title, ilabel,
        fn(ltp),  {**msty,"color":ltp_col},
        chg_str,  {**msty,"color":chg_col,"fontSize":"14px"},
        fn(op), fn(high), fn(low), fvol(tot_vol), fn(prev_cl),
        main_fig, tick_fig,
        tick_rows or [html.Div("Waiting for live ticks …",
                               style={"color":C["muted"],"fontSize":"12px"})],
        csv_txt,
    )

# ════════════════════════════════════════════════════════════════════════════
#  CHART LAYOUT BASE (light theme)
# ════════════════════════════════════════════════════════════════════════════
_BASE = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "#f8fbff",       # clean light blue-white plot area
    font = dict(family="'Courier New',monospace", color=C["text_dim"], size=10),
    margin = dict(l=60, r=16, t=22, b=38),
    xaxis = dict(showgrid=True, gridcolor="#e4eaf4", linecolor=C["border"],
                 tickcolor=C["border"], zeroline=False, tickfont=dict(color=C["text_dim"])),
    yaxis = dict(showgrid=True, gridcolor="#e4eaf4", linecolor=C["border"],
                 tickcolor=C["border"], zeroline=False, tickprefix="₹",
                 tickfont=dict(color=C["text_dim"])),
    legend = dict(bgcolor="rgba(255,255,255,0.85)",
                  bordercolor=C["border"], borderwidth=1,
                  orientation="h", x=0, y=1.06, font=dict(size=10, color=C["text"])),
    hovermode = "x unified",
    hoverlabel = dict(bgcolor="#ffffff", font_color=C["text"], bordercolor=C["border"],
                      font_family="'Courier New',monospace"),
)

# ════════════════════════════════════════════════════════════════════════════
#  MA TRACE HELPER
# ════════════════════════════════════════════════════════════════════════════
def _add_ma(fig, full_eod: pd.DataFrame, view_df: pd.DataFrame,
            window: int, col: str, label: str):
    """
    Compute MA on the FULL EOD history (so the window is always complete),
    then display only the portion that overlaps the view date range.
    """
    if full_eod.empty or "close" not in full_eod.columns: return

    fdf = full_eod.copy().sort_values("timestamp").reset_index(drop=True)
    fdf["_ma"] = fdf["close"].rolling(window, min_periods=1).mean()

    if not view_df.empty and "timestamp" in view_df.columns:
        cutoff = pd.to_datetime(view_df["timestamp"].min())
        plot   = fdf[fdf["timestamp"] >= cutoff].dropna(subset=["_ma"])
    else:
        plot = fdf.dropna(subset=["_ma"])

    if plot.empty: return

    fig.add_trace(go.Scatter(
        x=plot["timestamp"], y=plot["_ma"],
        mode="lines", name=label,
        line=dict(color=col, width=1.8, dash="dot"),
        opacity=0.9,
    ), row=1, col=1)


def _insert_gaps(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Insert NaN sentinel rows at overnight/weekend time gaps so Plotly
    line charts draw real breaks instead of connecting across sessions.

    Threshold by interval:
      1D  (1-min)  → gap > 5 min
      1W  (5-min)  → gap > 75 min
      1M  (15-min) → gap > 4 hours
      EOD          → no gap insertion
    """
    if df.empty or "timestamp" not in df.columns:
        return df
    _, interval_str, _ = PERIOD_CFG.get(period, (None, "eod", 1))
    if interval_str == "eod":
        return df
    threshold_map = {
        "1min":  pd.Timedelta(minutes=5),
        "5min":  pd.Timedelta(minutes=75),
        "15min": pd.Timedelta(hours=4),
    }
    threshold = threshold_map.get(interval_str, pd.Timedelta(hours=4))
    d = df.copy().reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    rows = []
    for i in range(len(d)):
        rows.append(d.iloc[i].to_dict())
        if i < len(d) - 1:
            if d.iloc[i + 1]["timestamp"] - d.iloc[i]["timestamp"] > threshold:
                gap_row = {col: None for col in d.columns}
                gap_row["timestamp"] = d.iloc[i]["timestamp"] + pd.Timedelta(seconds=1)
                rows.append(gap_row)
    result = pd.DataFrame(rows)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result


# ════════════════════════════════════════════════════════════════════════════
#  MAIN CHART BUILDER
# ════════════════════════════════════════════════════════════════════════════
def _make_main_fig(disp_df: pd.DataFrame, full_eod: pd.DataFrame,
                   symbol: str, period: str, chart_type: str,
                   ma_cfg: dict) -> go.Figure:

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.75, 0.25])

    if disp_df.empty:
        fig.add_annotation(text="⟳  Loading data …", xref="paper", yref="paper",
                           x=0.5, y=0.55, showarrow=False,
                           font=dict(size=15, color=C["muted"]))
    else:
        # Intraday periods need rangebreaks applied after all traces
        is_intraday = period in ("1D", "1W", "1M")

        for src, (lbl, up, dn) in SRC_STYLE.items():
            seg = disp_df[disp_df["source"] == src] if "source" in disp_df.columns else disp_df
            if seg.empty: continue

            seg = seg.dropna(subset=["open"]).copy()
            seg["timestamp"] = pd.to_datetime(seg["timestamp"], errors="coerce")
            seg = seg.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            if chart_type == "candle":
                fig.add_trace(go.Candlestick(
                    x=seg["timestamp"],
                    open=seg["open"], high=seg["high"],
                    low=seg["low"],   close=seg["close"],
                    name=lbl,
                    increasing_line_color=up, decreasing_line_color=dn,
                    increasing_fillcolor=up,  decreasing_fillcolor=dn,
                    line=dict(width=1),
                ), row=1, col=1)
            else:
                # connectgaps=True: Plotly joins every point directly.
                # rangebreaks collapses overnight/weekend on the x-axis,
                # so the last bar of Monday and first bar of Tuesday sit
                # right next to each other — producing the "vertical join"
                # the user expects between sessions.
                fig.add_trace(go.Scatter(
                    x=seg["timestamp"], y=seg["close"],
                    mode="lines", name=lbl,
                    line=dict(color=up, width=2.2, shape="linear"),
                    connectgaps=True,
                    fill="tozeroy" if src=="live" else None,
                    fillcolor="rgba(22,163,74,0.07)" if src=="live" else None,
                ), row=1, col=1)

            # Volume bars — clamp to >= 0
            seg_v = seg.copy()
            seg_v["volume"] = pd.to_numeric(seg_v["volume"], errors="coerce").fillna(0).clip(lower=0)
            bc = [up if float(c or 0) >= float(o or 0) else dn
                  for c, o in zip(seg_v["close"], seg_v["open"])]
            fig.add_trace(go.Bar(
                x=seg_v["timestamp"], y=seg_v["volume"],
                name=lbl, marker_color=bc, opacity=0.75, showlegend=False,
            ), row=2, col=1)

        # ── Moving averages (computed on full EOD, clipped to view) ──────
        if ma_cfg.get("ma50"):
            _add_ma(fig, full_eod, disp_df, 50,  C["ma50"],  "50 DMA")
        if ma_cfg.get("ma150"):
            _add_ma(fig, full_eod, disp_df, 150, C["ma150"], "150 DMA")
        if ma_cfg.get("ma200"):
            _add_ma(fig, full_eod, disp_df, 200, C["ma200"], "200 DMA")

        # ── x-axis range + rangebreaks for intraday periods ────────────
        disp_ts = pd.to_datetime(disp_df["timestamp"], errors="coerce").dropna()
        xmin = disp_ts.min()
        _xbuf = pd.Timedelta(minutes=30) if is_intraday else pd.Timedelta(days=3)
        xmax = disp_ts.max() + _xbuf

        _TICK_FMT = {
            "1D":  "%H:%M",
            "1W":  "%d %b",
            "1M":  "%d %b",
            "3M":  "%d %b",
            "6M":  "%b \'%y",
            "1Y":  "%b \'%y",
            "5Y":  "%b %Y",
            "ALL": "%Y",
        }
        tfmt = _TICK_FMT.get(period, "%d %b \'%y")

        if is_intraday:
            # rangebreaks physically remove non-trading hours and weekends from
            # the x-axis so all candles appear as a continuous sequence
            # NSE trading hours: 09:15 – 15:30 IST
            fig.update_xaxes(
                range=[xmin.isoformat(), xmax.isoformat()],
                tickformat=tfmt,
                rangebreaks=[
                    # Remove weekends (Saturday → Monday)
                    dict(bounds=["sat", "mon"]),
                    # Remove non-trading hours: after 15:30 until 09:15 next day
                    # bounds=[closingHour, openingHour] removes that night block
                    dict(bounds=[15.5, 9.25], pattern="hour"),
                ],
            )
        else:
            fig.update_xaxes(
                range=[xmin.isoformat(), xmax.isoformat()],
                tickformat=tfmt,
            )

        # ── Live price line ───────────────────────────────────────────────
        last_price = float(disp_df["close"].iloc[-1])
        fig.add_hline(y=last_price, line_dash="dot",
                      line_color=C["accent"], line_width=1.5,
                      annotation_text=f"  ₹{last_price:,.2f}",
                      annotation_font_color=C["accent"],
                      annotation_font_size=10, row=1, col=1)

        # ── Today separator ───────────────────────────────────────────────
        today_iso = now_ist().date().isoformat()
        fig.add_shape(type="line",
                      x0=today_iso, x1=today_iso, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color=C["amber"], width=1.2, dash="dash"))
        fig.add_annotation(x=today_iso, y=1, xref="x", yref="paper",
                           text="Today", showarrow=False,
                           font=dict(color=C["amber"], size=9),
                           xanchor="left", yanchor="bottom")

        # ── Interval label on chart ──────────────────────────────────────
        _ILABEL = {
            "1D":"Each candle = 1 min","1W":"Each candle = 5 min",
            "1M":"Each candle = 15 min","3M":"Each bar = 1 trading day",
            "6M":"Each bar = 1 trading day","1Y":"Each bar = 5 trading days",
            "5Y":"Each bar = 15 trading days","ALL":"Each bar = 15 trading days",
        }
        fig.add_annotation(
            text=_ILABEL.get(period,""),
            xref="paper", yref="paper", x=1.0, y=-0.07,
            showarrow=False, xanchor="right",
            font=dict(size=9, color=C["muted"]),
        )

        # ── Stats banner ──────────────────────────────────────────────────
        ath   = float(disp_df["high"].max())
        atl   = float(disp_df["low"].min())
        first = float(disp_df["close"].iloc[0])
        ret   = (last_price - first) / first * 100 if first else 0
        fig.add_annotation(
            text=(f"<b>{symbol}</b>  ·  ATH ₹{ath:,.2f}  ·  "
                  f"ATL ₹{atl:,.2f}  ·  Return {ret:+.1f}%"),
            xref="paper", yref="paper", x=0.5, y=1.075,
            showarrow=False, font=dict(size=11, color=C["text_dim"]))

    layout = {k: v for k, v in _BASE.items()}
    layout["height"] = 460
    layout["xaxis_rangeslider_visible"] = False
    # Volume y-axis: force baseline at zero, no negative values
    fig.update_yaxes(rangemode="tozero", row=2, col=1)
    fig.update_layout(**layout)
    return fig

# ════════════════════════════════════════════════════════════════════════════
#  TICK CHART BUILDER
# ════════════════════════════════════════════════════════════════════════════
def _make_tick_fig(tick_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not tick_df.empty and "ltp" in tick_df.columns:
        recent = tick_df.tail(400)
        ts  = pd.to_datetime(recent.get("ts", recent["timestamp"]), errors="coerce")
        ltp = recent["ltp"].values
        fig.add_trace(go.Scatter(
            x=ts, y=ltp, mode="lines", name="LTP",
            line=dict(color=C["live"], width=2.0, shape="linear"),
            fill="tozeroy", fillcolor="rgba(22,163,74,0.08)",
        ))
        if len(ltp) > 0:
            fig.add_trace(go.Scatter(
                x=[ts.iloc[-1]], y=[ltp[-1]], mode="markers", showlegend=False,
                marker=dict(size=8, color=C["live"],
                            line=dict(width=2, color="#fff")),
            ))

    tl = {k: v for k, v in _BASE.items() if k not in ("margin","yaxis")}
    tl["height"] = 220
    tl["margin"] = dict(l=56, r=10, t=6, b=28)
    tl["yaxis"]  = dict(showgrid=True, gridcolor="#e4eaf4",
                        linecolor=C["border"], zeroline=False,
                        tickprefix="₹", tickfont=dict(color=C["text_dim"]))
    fig.update_layout(**tl)
    return fig

# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="EvenStocks Dashboard v5")
    p.add_argument("user",     nargs="?", default=TRUEDATA_USER)
    p.add_argument("password", nargs="?", default=TRUEDATA_PASSWORD)
    p.add_argument("symbol",   nargs="?", default=DEFAULT_SYMBOL)
    p.add_argument("--prod",   action="store_true")
    args = p.parse_args()

    TRUEDATA_USER     = args.user
    TRUEDATA_PASSWORD = args.password
    DEFAULT_SYMBOL    = args.symbol.upper()
    PORT              = WS_PORT_PROD if args.prod else WS_PORT_SAND
    STORE.symbol      = DEFAULT_SYMBOL

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  EvenStocks Dashboard  v5.0  (Light Theme)                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Symbol  : {DEFAULT_SYMBOL:<51}║
║  Port    : {PORT} ({'PROD' if args.prod else 'SANDBOX'})                                      ║
║  CSV     : ./{CSV_DIR}/  (every {CSV_SAVE_MINUTES} min)                       ║
║  WS gate : Mon–Fri  09:15 – 15:30 IST                           ║
╠══════════════════════════════════════════════════════════════════╣
║  PERIOD → INTERVAL                                               ║
║  1D=1min  1W=5min  1M=15min  3M/6M=EOD  1Y=5d  5Y/ALL=15d       ║
╠══════════════════════════════════════════════════════════════════╣
║  → http://localhost:{DASH_PORT}                                      ║
╚══════════════════════════════════════════════════════════════════╝
""")

    os.makedirs(CSV_DIR, exist_ok=True)
    threading.Thread(target=history_load_loop, args=(DEFAULT_SYMBOL,), daemon=True).start()
    threading.Thread(target=csv_autosave_loop, daemon=True).start()
    get_or_start_ws(DEFAULT_SYMBOL)
    app.run(debug=False, port=DASH_PORT, use_reloader=False, dev_tools_hot_reload=False)
