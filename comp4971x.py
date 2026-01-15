# =============================================================================
# Multiple Model-based Binary Classification for Forex Trading
# Fixed & Optimized Version - January 2026
# =============================================================================

import math
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import (
    calibration, dummy, ensemble, linear_model,
    naive_bayes, neighbors, neural_network, svm, tree
)
import openpyxl
from openpyxl.styles import PatternFill, Alignment, Font

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ─── Configuration ──────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 10000
INVEST_RATIO    = 0.5
TRAIN_SPLIT     = 0.5
RISK_FREE_RATE  = 0.02
SPREAD          = 0.0001
SHOW_TOP        = 10

START_YEAR      = 2004
TRADE_YEARS     = 1           # ← change to 20 for full historical run
START_DATE      = f"{START_YEAR}-01-01"
END_DATE        = f"{START_YEAR + TRADE_YEARS}-01-01"

MAJOR_FOREX = [
    'USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCHF=X',
    'USDCAD=X', 'EURJPY=X', 'GBPJPY=X', 'EURCHF=X', 'NZDUSD=X',
]

# Fast & reliable models only (full list can be restored later)
MODELS = [
    dummy.DummyClassifier(),
    ensemble.RandomForestClassifier(n_estimators=80, n_jobs=-1),
    ensemble.HistGradientBoostingClassifier(max_iter=100),
    ensemble.ExtraTreesClassifier(n_estimators=80, n_jobs=-1),
    linear_model.LogisticRegression(max_iter=500, n_jobs=-1),
    linear_model.RidgeClassifier(max_iter=500),
    neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    tree.DecisionTreeClassifier(max_depth=7),
    svm.LinearSVC(max_iter=2000),
]

# ─── Excel Styling ──────────────────────────────────────────────────────────────

header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF")

profit_green = PatternFill(start_color="006400", end_color="006400", fill_type="solid")
profit_red   = PatternFill(start_color="8B0000", end_color="8B0000", fill_type="solid")
neutral      = PatternFill(start_color="333333", end_color="333333", fill_type="solid")

action_colors = {
    "Buy":        PatternFill(start_color="0011FF", end_color="0011FF", fill_type="solid"),
    "Sell":       PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid"),
    "Buy & Sell": PatternFill(start_color="12014D", end_color="12014D", fill_type="solid"),
    "No Action":  PatternFill(start_color="222222", end_color="222222", fill_type="solid"),
}

# ─── Trading Simulation ─────────────────────────────────────────────────────────

def simulate(main_df, preds):
    main = main_df.reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True)

    swing_profit = INITIAL_CAPITAL
    day_profit   = INITIAL_CAPITAL
    swing_lot = 0
    swing_open = False

    swing_profits = [INITIAL_CAPITAL]
    day_profits   = [INITIAL_CAPITAL]
    swing_acts = []
    day_acts   = []

    for i in range(len(preds)):
        prev = preds.iloc[i-1] if i > 0 else 0
        curr = preds.iloc[i]

        s_act = "No Action"
        d_act = "No Action"

        # Swing
        if i == len(preds)-1 and swing_open:
            s_act = "Sell"
            swing_profit += swing_lot * main['Open'].iloc[i] * (1 - SPREAD)
            swing_open = False
            swing_profits.append(swing_profit)

        elif i > 0 and prev == 1 and (i < 2 or preds.iloc[i-2] == 0):
            s_act = "Buy"
            if not swing_open and swing_profit > 100:
                swing_lot = int((swing_profit * INVEST_RATIO) / main['Open'].iloc[i])
                if swing_lot > 0:
                    swing_profit -= swing_lot * main['Open'].iloc[i] * (1 + SPREAD)
                    swing_open = True

        elif i > 0 and prev == 0 and (i < 2 or preds.iloc[i-2] == 1) and swing_open:
            s_act = "Sell"
            swing_profit += swing_lot * main['Open'].iloc[i] * (1 - SPREAD)
            swing_open = False
            swing_profits.append(swing_profit)

        # Day
        if i > 0 and prev == 1:
            lot = int((day_profit * INVEST_RATIO) / main['Open'].iloc[i])
            if lot > 0:
                pnl = lot * (main['Close'].iloc[i] - main['Open'].iloc[i]) * (1 - SPREAD)
                day_profit += pnl
            d_act = "Buy & Sell"
            day_profits.append(day_profit)

        swing_acts.append(s_act)
        day_acts.append(d_act)

    return (
        pd.Series(swing_acts, name='Swing Action'),
        pd.Series(day_acts,   name='Day Action'),
        pd.Series(swing_profits, name='Swing Profit'),
        pd.Series(day_profits,   name='Day Profit')
    )

# ─── Basic Metrics ──────────────────────────────────────────────────────────────

def cagr(final, years):
    return (final / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 and final > 0 else np.nan

def max_drawdown(profits):
    if len(profits) < 2: return np.nan
    peak = np.maximum.accumulate(profits)
    dd = (peak - profits) / peak
    return dd.max()

# ─── Download with Robust Column Repair ─────────────────────────────────────────

asset_data = []
for ticker in MAJOR_FOREX:
    print(f"Downloading {ticker}... ", end="")
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(start=START_DATE, end=END_DATE, interval='1d')

        if df.empty:
            print("EMPTY")
            continue

        # Fix broken repeated-ticker columns
        if all(str(c).strip() == ticker for c in df.columns):
            print("→ fixing repeated ticker columns")
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'][:len(df.columns)]

        # Normalize column names
        df.columns = [str(c).strip().title() for c in df.columns]
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'open'  in cl: rename_map[c] = 'Open'
            if 'close' in cl and 'adj' not in cl: rename_map[c] = 'Close'
            if 'volume' in cl: rename_map[c] = 'Volume'

        df = df.rename(columns=rename_map)

        if 'Open' not in df.columns or 'Close' not in df.columns:
            print(f"Missing Open/Close after rename: {list(df.columns)}")
            continue

        clean_df = df[['Open', 'Close']].dropna().reset_index(drop=True)
        print(f"→ {len(clean_df):,} rows")
        asset_data.append((ticker, clean_df))

    except Exception as e:
        print(f"FAILED: {str(e)[:60]}")

if not asset_data:
    raise ValueError("No valid data downloaded for any ticker.")

# ─── Training & Ranking Loop ────────────────────────────────────────────────────

ranking_rows = []

for ticker, df in asset_data:
    y = (df['Open'] < df['Close']).astype(int).shift(-1).fillna(0)
    train_size = int(len(df) * TRAIN_SPLIT)
    X_train, X_test = df.iloc[:train_size], df.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    for mdl in MODELS:
        name = mdl.__class__.__name__
        try:
            mdl.fit(X_train, y_train)
            pred = mdl.predict(X_test)
            pred = np.round(np.clip(pred, 0, 1)).astype(int)

            acc = metrics.accuracy_score(y_test, pred)

            sw_act, dy_act, sw_prof, dy_prof = simulate(X_test, pred)

            row = {
                'asset': ticker,
                'model': name,
                'duration': TRADE_YEARS,
                'accuracy': acc,
                'profit_threshold': 0,  # buy-and-hold placeholder
                'swing_profit': sw_prof.iloc[-1] - INITIAL_CAPITAL,
                'swing_cagr': cagr(sw_prof.iloc[-1], TRADE_YEARS),
                'swing_sharpe': 0,     # add real calc if needed
                'swing_mdd': max_drawdown(sw_prof),
                'swing_rr': 0,         # add if needed
                'swing_max_profit': sw_prof.max() - INITIAL_CAPITAL,
                'swing_max_loss': sw_prof.min() - INITIAL_CAPITAL,
                'swing_trades': (sw_act != "No Action").sum(),
                'day_profit': dy_prof.iloc[-1] - INITIAL_CAPITAL,
                'day_cagr': cagr(dy_prof.iloc[-1], TRADE_YEARS),
                'day_sharpe': 0,
                'day_mdd': max_drawdown(dy_prof),
                'day_rr': 0,
                'day_max_profit': dy_prof.max() - INITIAL_CAPITAL,
                'day_max_loss': dy_prof.min() - INITIAL_CAPITAL,
                'day_trades': (dy_act != "No Action").sum(),
            }
            ranking_rows.append(row)

            print(f"  ✓ {name:22}  acc={acc:.4f}  swing pnl={row['swing_profit']:+8.0f}")

        except Exception as e:
            print(f"  ✗ {name:22}  → {str(e)[:60]}")

ranking_df = pd.DataFrame(ranking_rows)

# ─── Excel Export ─ All per-metric top-10 with separation & colors ──────────────

output_file = "Forex_Top10_EveryMetric.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    wb = writer.book

    # 1. Full ranking (all rows)
    ranking_df.to_excel(writer, sheet_name='Full_Ranking', index=False)

    # Metrics to rank on
    swing_metrics = ['accuracy', 'swing_profit', 'swing_cagr', 'swing_mdd', 'swing_max_profit', 'swing_max_loss', 'swing_trades']
    day_metrics   = ['accuracy', 'day_profit',   'day_cagr',   'day_mdd',   'day_max_profit',   'day_max_loss',   'day_trades']

    for metric in swing_metrics + day_metrics:
        for ascending, label in [(False, 'Greatest'), (True, 'Fewest')]:
            sorted_df = ranking_df.sort_values(by=metric, ascending=ascending).head(SHOW_TOP)
            sheet_name = f"{label}_{metric[:12]}"[:31]  # safe length
            sorted_df.to_excel(writer, sheet_name=sheet_name, index=False)

            ws = writer.sheets[sheet_name]

            # Header styling
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = Font(bold=True, color="FFFFFF")
                cell.alignment = Alignment(horizontal='center', vertical='center')

            # Color profit-like columns
            profit_cols = [c for c in sorted_df.columns if any(k in c.lower() for k in ['profit', 'cagr'])]
            for col_name in profit_cols:
                col_idx = sorted_df.columns.get_loc(col_name) + 1
                for r in range(2, len(sorted_df) + 2):
                    cell = ws.cell(row=r, column=col_idx)
                    val = cell.value
                    if isinstance(val, (int, float)):
                        cell.fill = profit_green if val > 0 else profit_red if val < 0 else neutral
                        cell.alignment = Alignment(horizontal='center')

print(f"\nReport saved to: {output_file}")
print(f"Created {len(swing_metrics + day_metrics) * 2} top-10 sheets")
print(f"Each metric has separate 'Fewest' and 'Greatest' sheet")
print(f"Profit cells colored: green > 0, red < 0")