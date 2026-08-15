import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
from datetime import datetime, date, timedelta

# ============================================================
# BULLISH RSI DIVERGENCE SCREENER
# NSE Daily Bhavcopy + 2,463-stock user universe
#
# Features:
# - DD-MMM-YYYY date entry
# - Automatic NSE Bhavcopy download when selected date/history
#   is not already cached
# - Local cache to avoid repeated downloads
# - RSI(14)
# - Pivot Low(5,5)
# - Price Lower Low + RSI Higher Low
# - Signal date = pivot confirmation date
# ============================================================

st.set_page_config(
    page_title="Bullish RSI Divergence Screener",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = Path("data")
STOCKLIST_FILE = Path("stocklist.txt")

DEFAULT_RSI_LEN = 14
DEFAULT_SWING_LEN = 5
HISTORY_TRADING_DAYS = 80

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


# ------------------------------------------------------------
# RSI
# ------------------------------------------------------------
def rsi_wilder(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(
        ~((avg_loss == 0) & (avg_gain > 0)),
        100
    )
    rsi = rsi.where(
        ~((avg_gain == 0) & (avg_loss > 0)),
        0
    )

    return rsi


# ------------------------------------------------------------
# Stocklist
# ------------------------------------------------------------
@st.cache_data
def load_stocklist():
    if not STOCKLIST_FILE.exists():
        return set()

    symbols = set()

    with open(STOCKLIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            symbol = line.strip().upper()
            if symbol and set(symbol) != {"-"}:
                symbols.add(symbol)

    return symbols


# ------------------------------------------------------------
# NSE download
# ------------------------------------------------------------
def nse_url(dt):
    date_str = dt.strftime("%Y%m%d")
    return (
        "https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{date_str}_F_0000.csv.zip"
    )


def download_nse_day(dt, session=None):
    DATA_DIR.mkdir(exist_ok=True)

    target_csv = DATA_DIR / f"{dt.strftime('%Y%m%d')}.csv"

    if target_csv.exists():
        return True, "cached"

    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)

    try:
        response = session.get(
            nse_url(dt),
            timeout=30
        )

        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_names = [
                name for name in z.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_names:
                return False, "CSV not found in ZIP"

            csv_data = z.read(csv_names[0])

        target_csv.write_bytes(csv_data)

        return True, "downloaded"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def ensure_nse_history(target_date, required_trading_days=80):
    """
    Ensure target date + sufficient preceding NSE trading-day files
    exist locally. Existing files are never downloaded again.
    """
    DATA_DIR.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    # First walk backwards and download/check until enough trading days
    # are available. We allow a generous calendar window.
    found = 0
    checked = 0
    downloaded = 0

    dt = target_date
    max_calendar_days = required_trading_days * 4

    status_box = st.empty()
    progress = st.progress(0)

    while found < required_trading_days and checked < max_calendar_days:
        checked += 1

        csv_path = DATA_DIR / f"{dt.strftime('%Y%m%d')}.csv"

        if csv_path.exists():
            found += 1
        else:
            ok, status = download_nse_day(dt, session)

            if ok:
                found += 1
                if status == "downloaded":
                    downloaded += 1

        progress.progress(
            min(found / required_trading_days, 1.0)
        )

        status_box.write(
            f"NSE history: {found}/{required_trading_days} "
            f"trading days ready • checking {dt.strftime('%d-%b-%Y')}"
        )

        dt -= timedelta(days=1)

    status_box.empty()
    progress.empty()

    return {
        "ready_days": found,
        "checked_days": checked,
        "downloaded_days": downloaded,
        "target_file_exists": (
            DATA_DIR / f"{target_date.strftime('%Y%m%d')}.csv"
        ).exists(),
    }


# ------------------------------------------------------------
# Historical NSE data
# ------------------------------------------------------------
@st.cache_data
def load_nse_data():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        return pd.DataFrame()

    frames = []

    for file in files:
        try:
            df = pd.read_csv(file, low_memory=False)

            required = {
                "TradDt",
                "TckrSymb",
                "FinInstrmTp",
                "SctySrs",
                "HghPric",
                "LwPric",
                "ClsPric",
                "PrvsClsgPric",
                "TtlTradgVol",
            }

            if not required.issubset(df.columns):
                continue

            # Equity securities only.
            df = df[
                (df["FinInstrmTp"] == "STK") &
                (
                    df["SctySrs"].isin(
                        ["EQ", "BE", "BZ", "SM", "ST", "SZ"]
                    )
                )
            ].copy()

            df["TckrSymb"] = (
                df["TckrSymb"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            frames.append(
                df[
                    [
                        "TradDt",
                        "TckrSymb",
                        "HghPric",
                        "LwPric",
                        "ClsPric",
                        "PrvsClsgPric",
                        "TtlTradgVol",
                    ]
                ]
            )

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    data["TradDt"] = pd.to_datetime(
        data["TradDt"],
        errors="coerce"
    )

    for col in [
        "HghPric",
        "LwPric",
        "ClsPric",
        "PrvsClsgPric",
        "TtlTradgVol",
    ]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=[
            "TradDt",
            "TckrSymb",
            "LwPric",
            "ClsPric",
        ]
    )

    data = data.sort_values(
        ["TckrSymb", "TradDt"]
    )

    data = data.drop_duplicates(
        subset=["TckrSymb", "TradDt"],
        keep="last"
    )

    return data


# ------------------------------------------------------------
# Bullish divergence
# ------------------------------------------------------------
def find_bullish_divergence(
    group,
    target_date,
    swing_len=5,
    rsi_len=14
):
    group = (
        group
        .sort_values("TradDt")
        .reset_index(drop=True)
        .copy()
    )

    group["RSI"] = rsi_wilder(
        group["ClsPric"],
        rsi_len
    )

    low = group["LwPric"].to_numpy(dtype=float)
    rsi = group["RSI"].to_numpy(dtype=float)

    n = len(group)

    prev_low = np.nan
    prev_rsi_low = np.nan

    for i in range(swing_len, n):

        pivot_i = i - swing_len

        left = pivot_i - swing_len
        right = pivot_i + swing_len

        if left < 0 or right >= n:
            continue

        pivot_low = low[pivot_i]
        pivot_rsi = rsi[pivot_i]

        if not np.isfinite(pivot_low):
            continue

        if not np.isfinite(pivot_rsi):
            continue

        window = low[left:right + 1]

        if pivot_low != np.nanmin(window):
            continue

        bullish_div = (
            np.isfinite(prev_low)
            and np.isfinite(prev_rsi_low)
            and pivot_low < prev_low
            and pivot_rsi > prev_rsi_low
        )

        if bullish_div:

            confirmation_date = group.loc[i, "TradDt"]

            if confirmation_date.date() == target_date:

                row = group.loc[i]

                close = float(row["ClsPric"])
                prev_close = float(row["PrvsClsgPric"])
                volume = float(row["TtlTradgVol"])

                pct_change = np.nan

                if (
                    np.isfinite(prev_close)
                    and prev_close != 0
                ):
                    pct_change = (
                        (close / prev_close) - 1
                    ) * 100

                return {
                    "Stock Symbol": str(
                        row["TckrSymb"]
                    ),
                    "Closing Price": close,
                    "% Change": pct_change,
                    "Volume": volume,
                    "Pivot Date": group.loc[
                        pivot_i, "TradDt"
                    ].strftime("%d-%b-%Y"),
                    "Pivot Low": pivot_low,
                    "Pivot RSI": pivot_rsi,
                    "Previous Pivot Low": prev_low,
                    "Previous Pivot RSI": prev_rsi_low,
                    "RSI": float(row["RSI"]),
                }

        # Same order as Pine:
        # test divergence first, then update previous pivot.
        prev_low = pivot_low
        prev_rsi_low = pivot_rsi

    return None


# ------------------------------------------------------------
# Scanner
# ------------------------------------------------------------
def scan_watchlist(target_date, swing_len, rsi_len):

    stocklist = load_stocklist()
    data = load_nse_data()

    if not stocklist:
        raise RuntimeError(
            "stocklist.txt not found or empty."
        )

    if data.empty:
        raise RuntimeError(
            "No NSE CSV files found in data folder."
        )

    data = data[
        data["TckrSymb"].isin(stocklist)
    ].copy()

    data = data[
        data["TradDt"] <= pd.Timestamp(target_date)
    ].copy()

    if data.empty:
        return pd.DataFrame()

    results = []

    grouped = data.groupby(
        "TckrSymb",
        sort=False
    )

    progress = st.progress(0)
    status = st.empty()

    total = len(grouped)

    for number, (symbol, group) in enumerate(
        grouped,
        start=1
    ):

        result = find_bullish_divergence(
            group,
            target_date,
            swing_len,
            rsi_len
        )

        if result:
            results.append(result)

        if number == total or number % 100 == 0:
            progress.progress(number / total)
            status.write(
                f"Scanning {number:,} / {total:,} stocks..."
            )

    status.empty()
    progress.empty()

    return pd.DataFrame(results)


# ============================================================
# UI
# ============================================================

st.title("📈 Bullish RSI Divergence Screener")

st.caption(
    "NSE Daily Bhavcopy • Bullish RSI Divergence only • "
    "TradingView-style Pivot Low confirmation"
)

st.divider()

with st.sidebar:

    st.header("⚙️ Settings")

    rsi_len = st.number_input(
        "RSI Length",
        min_value=2,
        max_value=50,
        value=DEFAULT_RSI_LEN,
        step=1
    )

    swing_len = st.number_input(
        "Swing Length",
        min_value=2,
        max_value=20,
        value=DEFAULT_SWING_LEN,
        step=1
    )

    st.divider()

    symbols = load_stocklist()

    st.metric(
        "Stocklist",
        f"{len(symbols):,}"
    )

    cached_files = list(DATA_DIR.glob("*.csv"))

    if cached_files:
        cached_dates = []

        for f in cached_files:
            try:
                cached_dates.append(
                    datetime.strptime(
                        f.stem,
                        "%Y%m%d"
                    ).date()
                )
            except Exception:
                pass

        if cached_dates:
            st.write(
                f"Cached data: "
                f"**{min(cached_dates).strftime('%d-%b-%Y')}** "
                f"to "
                f"**{max(cached_dates).strftime('%d-%b-%Y')}**"
            )

    st.caption(
        "NSE data is automatically cached locally after download."
    )


# ------------------------------------------------------------
# Date input + trading-day navigation
# ------------------------------------------------------------
today = date.today()

# Navigation state is intentionally kept separate from the text_input widget.
# Streamlit does not allow assigning to a widget's session_state key after
# that widget has been instantiated. Using a separate state variable avoids
# the StreamlitAPIException seen on Previous/Next Trading Day clicks.
if "selected_date_text" not in st.session_state:
    st.session_state.selected_date_text = today.strftime("%d-%b-%Y")


def cached_trading_dates():
    """Return all locally cached NSE trading dates from YYYYMMDD.csv files."""
    dates = []

    for f in DATA_DIR.glob("*.csv"):
        try:
            dates.append(
                datetime.strptime(
                    f.stem,
                    "%Y%m%d"
                ).date()
            )
        except ValueError:
            # Ignore CSVs that are not named as YYYYMMDD.
            continue

    return sorted(set(dates))


date_text = st.text_input(
    "📅 DD-MMM-YYYY",
    value=st.session_state.selected_date_text,
    help="Example: 14-Aug-2026"
).strip()

# Keep the latest text input in our own state variable. This is NOT the
# widget's key, so it can safely be changed by navigation buttons below.
st.session_state.selected_date_text = date_text

try:
    typed_date = datetime.strptime(
        date_text,
        "%d-%b-%Y"
    ).date()
    valid_date = True
except ValueError:
    typed_date = None
    valid_date = False

# Build known/cached trading dates from local CSV files.
trading_dates = cached_trading_dates()


def previous_trading_day(d):
    candidates = [x for x in trading_dates if x < d]
    return max(candidates) if candidates else None


def next_trading_day(d):
    candidates = [x for x in trading_dates if x > d]
    return min(candidates) if candidates else None


# For a manually typed weekend/holiday, show the previous cached trading
# day, but DO NOT overwrite the widget state automatically. This prevents
# Streamlit widget-state conflicts and keeps the typed date visible.
selected_date = typed_date

if valid_date and trading_dates and typed_date not in trading_dates:
    prev_day = previous_trading_day(typed_date)

    if prev_day is not None:
        selected_date = prev_day

        st.info(
            f"📅 {typed_date.strftime('%d-%b-%Y')} was not a trading day. "
            f"Using previous trading day: "
            f"**{selected_date.strftime('%d-%b-%Y')}**"
        )

if valid_date:

    st.write(
        f"Selected Trading Date: "
        f"**{selected_date.strftime('%d-%b-%Y')}**"
    )

    col_prev, col_get, col_next = st.columns(
        [1, 2, 1]
    )

    prev_cached = previous_trading_day(
        selected_date
    )

    next_cached = next_trading_day(
        selected_date
    )

    with col_prev:
        prev_clicked = st.button(
            "◀ Previous Trading Day",
            use_container_width=True,
            disabled=(prev_cached is None)
        )

    with col_get:
        get_watchlist = st.button(
            "🔎 GET WATCHLIST",
            type="primary",
            use_container_width=True
        )

    with col_next:
        next_clicked = st.button(
            "Next Trading Day ▶",
            use_container_width=True,
            disabled=(next_cached is None)
        )

    # Navigation is handled after the widgets are evaluated. We update the
    # separate selected_date_text state, then rerun so the text box displays
    # the new date. No widget-owned session_state key is modified.
    if prev_clicked and prev_cached is not None:
        st.session_state.selected_date_text = (
            prev_cached.strftime("%d-%b-%Y")
        )
        st.rerun()

    if next_clicked and next_cached is not None:
        st.session_state.selected_date_text = (
            next_cached.strftime("%d-%b-%Y")
        )
        st.rerun()

else:
    get_watchlist = False

    st.error(
        "Invalid date. Please use DD-MMM-YYYY, "
        "for example 14-Aug-2026."
    )


# ------------------------------------------------------------
# Scan button
# ------------------------------------------------------------
if get_watchlist and valid_date:

    if selected_date > today:

        st.error(
            "Future date is not allowed."
        )
        st.stop()

    with st.spinner(
        "Checking NSE cache and downloading missing history..."
    ):

        cache_info = ensure_nse_history(
            selected_date,
            HISTORY_TRADING_DAYS
        )

    if not cache_info["target_file_exists"]:

        st.error(
            f"No NSE trading-day Bhavcopy is available for "
            f"{selected_date.strftime('%d-%b-%Y')}. "
            f"Please select an NSE trading day."
        )
        st.stop()

    # New files may have been added; clear the cached dataframe.
    load_nse_data.clear()

    with st.spinner(
        "Calculating Bullish RSI Divergence..."
    ):

        try:

            result = scan_watchlist(
                selected_date,
                int(swing_len),
                int(rsi_len)
            )

        except Exception as e:

            st.error(str(e))
            st.stop()

    st.divider()

    if result.empty:

        st.warning(
            f"No Bullish RSI Divergence found on "
            f"{selected_date.strftime('%d-%b-%Y')}."
        )

    else:

        result = result.sort_values(
            "% Change",
            ascending=False
        ).reset_index(drop=True)

        st.success(
            f"🟢 {len(result)} Bullish RSI Divergence "
            f"stock(s) found"
        )

        display = result[
            [
                "Stock Symbol",
                "Closing Price",
                "% Change",
                "Volume",
            ]
        ].copy()

        display["Closing Price"] = display[
            "Closing Price"
        ].map(
            lambda x: f"{x:,.2f}"
        )

        display["% Change"] = display[
            "% Change"
        ].map(
            lambda x: (
                f"{x:+.2f}%"
                if pd.notna(x)
                else "-"
            )
        )

        def format_volume(x):
            if x >= 1_000_000:
                return f"{x / 1_000_000:.2f}M"
            elif x >= 100_000:
                return f"{x / 100_000:.2f}L"
            elif x >= 1_000:
                return f"{x / 1_000:.2f}K"
            return f"{x:,.0f}"

        display["Volume"] = display[
            "Volume"
        ].map(format_volume)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=600
        )

        csv = result[
            [
                "Stock Symbol",
                "Closing Price",
                "% Change",
                "Volume",
            ]
        ].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Watchlist CSV",
            data=csv,
            file_name=(
                "bullish_rsi_watchlist_"
                f"{selected_date.strftime('%Y-%m-%d')}.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

        with st.expander(
            "🔍 Show Divergence Details"
        ):

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

st.divider()

st.caption(
    "Logic: Price Lower Low + RSI Higher Low, "
    "using RSI(14) and Pivot Low(5,5). "
    "Signal is reported on the pivot confirmation day."
)
