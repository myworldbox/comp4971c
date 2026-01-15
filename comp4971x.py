# =============================================================================
# Multiple Model-based Binary Classification for Forex Trading
# Enhanced & Refactored Version - January 2026
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
CONFIG = {
    "initial_capital": 10000,
    "invest_ratio": 0.5,
    "train_split": 0.5,
    "risk_free_rate": 0.02,
    "spread": 0.0001,
    "show_top": 10,
    "start_year": 2004,
    "trade_years": 20,
}
CONFIG["start_date"] = f"{CONFIG['start_year']}-01-01"
CONFIG["end_date"]   = f"{CONFIG['start_year'] + CONFIG['trade_years']}-01-01"

MAJOR_FOREX = [
    'USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCHF=X',
    'USDCAD=X', 'EURJPY=X', 'GBPJPY=X', 'EURCHF=X', 'NZDUSD=X',
]

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

# ─── Helper Functions ───────────────────────────────────────────────────────────
def trade_lot(capital, price, ratio):
    return int((capital * ratio) / price)

class Metrics:
    @staticmethod
    def cagr(final, years, initial=CONFIG["initial_capital"]):
        return (final / initial) ** (1 / years) - 1 if years > 0 and final > 0 else np.nan

    @staticmethod
    def max_drawdown(profits):
        if len(profits) < 2: return np.nan
        peak = np.maximum.accumulate(profits)
        return ((peak - profits) / peak).max()

def clean_yf_data(ticker, start, end):
    df = yf.Ticker(ticker).history(start=start, end=end, interval='1d')
    if df.empty: return None
    df.columns = [c.strip().title() for c in df.columns]
    rename_map = {c: 'Open' if 'open' in c.lower() else 'Close' if 'close' in c.lower() and 'adj' not in c.lower() else c for c in df.columns}
    df = df.rename(columns=rename_map)
    return df[['Open', 'Close']].dropna().reset_index(drop=True) if {'Open','Close'} <= set(df.columns) else None

# ─── Trading Simulation ─────────────────────────────────────────────────────────
def simulate(main_df, preds):
    main = main_df.reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True)
    swing_profit = CONFIG["initial_capital"]
    day_profit   = CONFIG["initial_capital"]
    swing_lot    = 0
    swing_open   = False
    swing_profits = [CONFIG["initial_capital"]]
    day_profits   = [CONFIG["initial_capital"]]
    swing_acts, day_acts = [], []

    for i in range(len(preds)):
        prev = preds.iloc[i-1] if i > 0 else 0
        curr = preds.iloc[i]
        s_act, d_act = "No Action", "No Action"

        # Swing Trading
        if i == len(preds)-1 and swing_open:
            s_act = "Sell"
            swing_profit += swing_lot * main['Open'].iloc[i] * (1 - CONFIG["spread"])
            swing_open = False
            swing_profits.append(swing_profit)
        elif i > 0 and prev == 1 and (i < 2 or preds.iloc[i-2] == 0):
            s_act = "Buy"
            if not swing_open and swing_profit > 100:
                swing_lot = trade_lot(swing_profit, main['Open'].iloc[i], CONFIG["invest_ratio"])
                if swing_lot > 0:
                    swing_profit -= swing_lot * main['Open'].iloc[i] * (1 + CONFIG["spread"])
                    swing_open = True
        elif i > 0 and prev == 0 and (i < 2 or preds.iloc[i-2] == 1) and swing_open:
            s_act = "Sell"
            swing_profit += swing_lot * main['Open'].iloc[i] * (1 - CONFIG["spread"])
            swing_open = False
            swing_profits.append(swing_profit)

        # Day Trading
        if i > 0 and prev == 1:
            lot = trade_lot(day_profit, main['Open'].iloc[i], CONFIG["invest_ratio"])
            if lot > 0:
                pnl = lot * (main['Close'].iloc[i] - main['Open'].iloc[i]) * (1 - CONFIG["spread"])
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

# ─── Excel Helpers ──────────────────────────────────────────────────────────────
def color_profit_cells(ws, sorted_df):
    profit_cols = [c for c in sorted_df.columns if any(k in c.lower() for k in ['profit','cagr'])]
    for col_name in profit_cols:
        col_idx = sorted_df.columns.get_loc(col_name) + 1
        for r in range(2, len(sorted_df) + 2):
            cell = ws.cell(row=r, column=col_idx)
            if isinstance(cell.value, (int, float)):
                cell.fill = profit_green if cell.value > 0 else profit_red if cell.value < 0 else neutral

def embed_profit_graph(ws, sorted_df, metric, label, history_col, profit_col, start_cell):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"{label} Top {CONFIG['show_top']} - {history_col} ({metric})", fontsize=14)
    axes = axes.flat
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax = axes[i]
        ax.plot(row[history_col], color='green' if row[profit_col] > 0 else 'red')
        ax.set_title(f"{row['asset']}\n{row['model']}\nProfit: {row[profit_col]:.0f}", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=100); buf.seek(0)
    ws.add_image(Image(buf), start_cell)
    plt.close(fig)

# ─── Download Data ──────────────────────────────────────────────────────────────
asset_data = []
for ticker in MAJOR_FOREX:
    print(f"Downloading {ticker}... ", end="")
    df = clean_yf_data(ticker, CONFIG["start_date"], CONFIG["end_date"])
    if df is not None:
        print(f"→ {len(df):,} rows")
        asset_data.append((ticker, df))
    else:
        print("EMPTY or invalid")

if not asset_data:
    raise ValueError("No valid data downloaded.")

# ─── Training & Ranking ─────────────────────────────────────────────────────────
ranking_rows = []
for ticker, df in asset_data:
    y = (df['Open'] < df['Close']).astype(int).shift(-1).fillna(0)
    train_size = int(len(df) * CONFIG["train_split"])
    X_train, X_test = df.iloc[:train_size], df.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Buy-and-Hold benchmark
    bh_units = CONFIG["initial_capital"] / X_test['Open'].iloc[0]
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
                'duration_years': CONFIG["trade_years"],
                'accuracy': acc,
                'buy_hold_profit': bh_profit,
                'swing_profit': sw_prof.iloc[-1] - CONFIG["initial_capital"],
                'swing_cagr': Metrics.cagr(sw_prof.iloc[-1], CONFIG["trade_years"]),
                'swing_mdd': Metrics.max_drawdown(sw_prof),
                'swing_max_profit': sw_prof.max() - CONFIG["initial_capital"],
                'swing_max_loss': sw_prof.min() - CONFIG["initial_capital"],
                'swing_trades': (sw_act != "No Action").sum(),
                'day_profit': dy_prof.iloc[-1] - CONFIG["initial_capital"],
                'day_cagr': Metrics.cagr(dy_prof.iloc[-1], CONFIG["trade_years"]),
                'day_mdd': Metrics.max_drawdown(dy_prof),
                'day_max_profit': dy_prof.max() - CONFIG["initial_capital"],
                'day_max_loss': dy_prof.min() - CONFIG["initial_capital"],
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

    # 1. Full ranking (without histories)
    ranking_df.drop(columns=['swing_profit_history', 'day_profit_history']).to_excel(
        writer, sheet_name='Full_Ranking', index=False
    )

    # Metrics to rank on
    swing_metrics = ['accuracy', 'swing_profit', 'swing_cagr', 'swing_mdd',
                     'swing_max_profit', 'swing_max_loss', 'swing_trades']
    day_metrics   = ['accuracy', 'day_profit', 'day_cagr', 'day_mdd',
                     'day_max_profit', 'day_max_loss', 'day_trades']

    for metric in swing_metrics + day_metrics:
        for ascending, label in [(False, 'Greatest'), (True, 'Fewest')]:
            sorted_df = ranking_df.sort_values(by=metric, ascending=ascending).head(CONFIG["show_top"])
            sheet_name = f"{label}_{metric}"[:31]
            sorted_df.drop(columns=['swing_profit_history', 'day_profit_history']).to_excel(
                writer, sheet_name=sheet_name, index=False
            )

            ws = writer.sheets[sheet_name]

            # Header styling
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')

            # Profit cell coloring
            color_profit_cells(ws, sorted_df)

            # Embed Swing & Day Profit Graphs
            embed_profit_graph(ws, sorted_df, metric, label,
                               'swing_profit_history', 'swing_profit', 'A20')
            embed_profit_graph(ws, sorted_df, metric, label,
                               'day_profit_history', 'day_profit', 'J20')

print(f"\nReport saved to: {output_file}")
print(f"Created {len(swing_metrics + day_metrics) * 2} top-10 sheets")
print("Each sheet includes:")
print(" - Ranked table")
print(" - Embedded capital flow graphs for Swing & Day (2x5 subplots)")
print(" - Profit cells colored: green (>0), red (<0)")
