# =============================================================================
# Multiple Model-based Binary Classification for Forex Trading
# Fixed & Optimized Version - January 2026
# =============================================================================
# Changes:
# - Stores profit histories for plotting
# - Generates capital flow graphs (swing & day) for top-10 per metric
# - Embeds PNG graphs directly into each top-10 Excel sheet
# - Works outside Colab (using matplotlib + openpyxl)
# - Added basic buy-and-hold calculation
# - Improved styling and positioning
# =============================================================================
import math
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import (
    calibration, dummy, ensemble, linear_model,
    naive_bayes, neighbors, neural_network, svm, tree
)
import openpyxl
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.drawing.image import Image
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ─── Configuration ──────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10000
INVEST_RATIO   = 0.5
TRAIN_SPLIT    = 0.5
RISK_FREE_RATE = 0.02
SPREAD         = 0.0001          # Forex typical spread
SHOW_TOP       = 10
START_YEAR     = 2004
TRADE_YEARS    = 20
START_DATE     = f"{START_YEAR}-01-01"
END_DATE       = f"{START_YEAR + TRADE_YEARS}-01-01"

MAJOR_FOREX = [
    'USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCHF=X',
    'USDCAD=X', 'EURJPY=X', 'GBPJPY=X', 'EURCHF=X', 'NZDUSD=X',
]

# Reliable & fast models (expand as needed)
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

# ─── Trading Simulation ─────────────────────────────────────────────────────────
def simulate(main_df, preds):
    main = main_df.reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True)
    swing_profit = INITIAL_CAPITAL
    day_profit   = INITIAL_CAPITAL
    swing_lot    = 0
    swing_open   = False
    swing_profits = [INITIAL_CAPITAL]
    day_profits   = [INITIAL_CAPITAL]
    swing_acts = []
    day_acts   = []

    for i in range(len(preds)):
        prev = preds.iloc[i-1] if i > 0 else 0
        curr = preds.iloc[i]
        s_act = "No Action"
        d_act = "No Action"

        # Swing Trading
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

        # Day Trading
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

# ─── Download Data ──────────────────────────────────────────────────────────────
asset_data = []
for ticker in MAJOR_FOREX:
    print(f"Downloading {ticker}... ", end="")
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(start=START_DATE, end=END_DATE, interval='1d')
        if df.empty:
            print("EMPTY")
            continue
        # Fix repeated ticker columns (common yfinance bug)
        if all(str(c).strip() == ticker for c in df.columns):
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'][:len(df.columns)]
        df.columns = [str(c).strip().title() for c in df.columns]
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'open' in cl: rename_map[c] = 'Open'
            if 'close' in cl and 'adj' not in cl: rename_map[c] = 'Close'
            if 'volume' in cl: rename_map[c] = 'Volume'
        df = df.rename(columns=rename_map)
        if 'Open' not in df.columns or 'Close' not in df.columns:
            print(f"Missing Open/Close: {list(df.columns)}")
            continue
        clean_df = df[['Open', 'Close']].dropna().reset_index(drop=True)
        print(f"→ {len(clean_df):,} rows")
        asset_data.append((ticker, clean_df))
    except Exception as e:
        print(f"FAILED: {str(e)[:60]}")

if not asset_data:
    raise ValueError("No valid data downloaded.")

# ─── Training & Ranking ─────────────────────────────────────────────────────────
ranking_rows = []
for ticker, df in asset_data:
    y = (df['Open'] < df['Close']).astype(int).shift(-1).fillna(0)
    train_size = int(len(df) * TRAIN_SPLIT)
    X_train, X_test = df.iloc[:train_size], df.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Buy-and-Hold for reference
    bh_units = INITIAL_CAPITAL / X_test['Open'].iloc[0]
    bh_profit = (X_test['Close'].iloc[-1] - X_test['Open'].iloc[0]) * bh_units

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
                'duration_years': TRADE_YEARS,
                'accuracy': acc,
                'buy_hold_profit': bh_profit,
                'swing_profit': sw_prof.iloc[-1] - INITIAL_CAPITAL,
                'swing_cagr': cagr(sw_prof.iloc[-1], TRADE_YEARS),
                'swing_mdd': max_drawdown(sw_prof),
                'swing_max_profit': sw_prof.max() - INITIAL_CAPITAL,
                'swing_max_loss': sw_prof.min() - INITIAL_CAPITAL,
                'swing_trades': (sw_act != "No Action").sum(),
                'day_profit': dy_prof.iloc[-1] - INITIAL_CAPITAL,
                'day_cagr': cagr(dy_prof.iloc[-1], TRADE_YEARS),
                'day_mdd': max_drawdown(dy_prof),
                'day_max_profit': dy_prof.max() - INITIAL_CAPITAL,
                'day_max_loss': dy_prof.min() - INITIAL_CAPITAL,
                'day_trades': (dy_act != "No Action").sum(),
                # Histories for plotting
                'swing_profit_history': sw_prof.tolist(),
                'day_profit_history': dy_prof.tolist(),
            }
            ranking_rows.append(row)
            print(f" ✓ {name:22} acc={acc:.4f} swing pnl={row['swing_profit']:+8.0f}")
        except Exception as e:
            print(f" ✗ {name:22} → {str(e)[:60]}")

ranking_df = pd.DataFrame(ranking_rows)

# ─── Excel Export with Embedded Graphs ──────────────────────────────────────────
output_file = "Forex_Top10_EveryMetric_WithGraphs.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    wb = writer.book

    # 1. Full ranking
    ranking_df.drop(columns=['swing_profit_history', 'day_profit_history']).to_excel(writer, sheet_name='Full_Ranking', index=False)

    # Metrics to rank on
    swing_metrics = ['accuracy', 'swing_profit', 'swing_cagr', 'swing_mdd', 'swing_max_profit', 'swing_max_loss', 'swing_trades']
    day_metrics   = ['accuracy', 'day_profit',   'day_cagr',   'day_mdd',   'day_max_profit',   'day_max_loss',   'day_trades']

    for metric in swing_metrics + day_metrics:
        for ascending, label in [(False, 'Greatest'), (True, 'Fewest')]:
            sorted_df = ranking_df.sort_values(by=metric, ascending=ascending).head(SHOW_TOP)
            sheet_name = f"{label}_{metric}"[:31]
            sorted_df.drop(columns=['swing_profit_history', 'day_profit_history']).to_excel(writer, sheet_name=sheet_name, index=False)

            ws = writer.sheets[sheet_name]

            # Header styling
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')

            # Color profit columns
            profit_cols = [c for c in sorted_df.columns if any(k in c.lower() for k in ['profit', 'cagr'])]
            for col_name in profit_cols:
                col_idx = sorted_df.columns.get_loc(col_name) + 1
                for r in range(2, len(sorted_df) + 2):
                    cell = ws.cell(row=r, column=col_idx)
                    val = cell.value
                    if isinstance(val, (int, float)):
                        cell.fill = profit_green if val > 0 else profit_red if val < 0 else neutral

            # ─── Embed Swing Profit History Graph ───────────────────────────────────
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f"{label} Top {SHOW_TOP} - Swing Profit History ({metric})", fontsize=14)
            axes = axes.flat
            for i, (_, row) in enumerate(sorted_df.iterrows()):
                ax = axes[i]
                ax.plot(row['swing_profit_history'], color='green' if row['swing_profit'] > 0 else 'red')
                ax.set_title(f"{row['asset']}\n{row['model']}\nProfit: {row['swing_profit']:.0f}", fontsize=9)
                ax.set_xlabel('Step')
                ax.set_ylabel('Capital')
                ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image(buf)
            ws.add_image(img, 'A20')  # Swing graph starts at A20
            plt.close(fig)

            # ─── Embed Day Profit History Graph ─────────────────────────────────────
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f"{label} Top {SHOW_TOP} - Day Profit History ({metric})", fontsize=14)
            axes = axes.flat
            for i, (_, row) in enumerate(sorted_df.iterrows()):
                ax = axes[i]
                ax.plot(row['day_profit_history'], color='green' if row['day_profit'] > 0 else 'red')
                ax.set_title(f"{row['asset']}\n{row['model']}\nProfit: {row['day_profit']:.0f}", fontsize=9)
                ax.set_xlabel('Step')
                ax.set_ylabel('Capital')
                ax.tick_params(axis='both', labelsize=8)
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image(buf)
            ws.add_image(img, 'J20')  # Day graph starts next to Swing (column J)
            plt.close(fig)

print(f"\nReport saved to: {output_file}")
print(f"Created {len(swing_metrics + day_metrics) * 2} top-10 sheets")
print("Each sheet now includes:")
print(" - Ranked table")
print(" - Embedded capital flow graphs for Swing & Day (2x5 subplots)")
print(" - Profit cells colored: green (>0), red (<0)")