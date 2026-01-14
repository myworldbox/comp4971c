import math
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import (
    calibration, dummy, ensemble, gaussian_process,
    linear_model, naive_bayes, neighbors, neural_network,
    svm, tree
)
import openpyxl
from openpyxl.styles import PatternFill, Alignment

warnings.filterwarnings('ignore')

# ─── Config ─────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 10_000
INVEST_RATIO    = 0.5
TRAIN_SPLIT     = 0.5
RISK_FREE_RATE  = 0.02
SPREAD          = 0.0001
SHOW_TOP        = 10

START_YEAR      = 2004
TRADE_YEARS     = 1
START_DATE      = f"{START_YEAR}-01-01"
END_DATE        = f"{START_YEAR + TRADE_YEARS}-01-01"

MAJOR_FOREX = [
    'USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCHF=X',
    'USDCAD=X', 'EURJPY=X', 'GBPJPY=X', 'EURCHF=X', 'NZDUSD=X',
]

# Large but reasonable list of classifiers (skipped regressors, isolation, semi-supervised, etc.)
MODELS = [
    dummy.DummyClassifier(),
    calibration.CalibratedClassifierCV(),
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(n_estimators=100),
    ensemble.GradientBoostingClassifier(),
    ensemble.HistGradientBoostingClassifier(),
    ensemble.RandomForestClassifier(n_estimators=100),
    gaussian_process.GaussianProcessClassifier(),
    linear_model.LogisticRegression(max_iter=2000),
    linear_model.LogisticRegressionCV(max_iter=2000),
    linear_model.PassiveAggressiveClassifier(max_iter=2000),
    linear_model.Perceptron(max_iter=2000),
    linear_model.RidgeClassifier(max_iter=2000),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(max_iter=2000),
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    neighbors.KNeighborsClassifier(),
    neighbors.NearestCentroid(),
    neural_network.MLPClassifier(max_iter=800),
    svm.LinearSVC(max_iter=4000),
    svm.SVC(),
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    # You can uncomment more if you want — but these are the most likely to run
    # ensemble.IsolationForest(),               # anomaly detection
    # semi_supervised.LabelPropagation(),       # needs special handling
    # semi_supervised.LabelSpreading(),
]

# ─── Styling ────────────────────────────────────────────────────────────────────

def action_fill(action):
    if action == "Buy":        return PatternFill(start_color="0011FF", end_color="0011FF", fill_type="solid")
    if action == "Sell":       return PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    if action == "Buy & Sell": return PatternFill(start_color="12014D", end_color="12014D", fill_type="solid")
    return PatternFill(start_color="222222", end_color="222222", fill_type="solid")

profit_green = PatternFill(start_color="006400", end_color="006400", fill_type="solid")
profit_red   = PatternFill(start_color="8B0000", end_color="8B0000", fill_type="solid")
neutral      = PatternFill(start_color="333333", end_color="333333", fill_type="solid")

# ─── Trading Simulation ─────────────────────────────────────────────────────────

def simulate(main_df, preds):
    main = main_df.reset_index(drop=True)
    preds = preds.reset_index(drop=True)

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

        # Day (intraday assume open → close)
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

# ─── Metrics ────────────────────────────────────────────────────────────────────

def cagr(final, initial, years):
    if years <= 0 or final <= 0: return np.nan
    return (final / initial) ** (1 / years) - 1

def max_dd(profits):
    if len(profits) < 2: return np.nan
    peak = np.maximum.accumulate(profits)
    dd = (peak - profits) / peak
    return dd.max()

# ─── Main ───────────────────────────────────────────────────────────────────────

results = []

for ticker in MAJOR_FOREX:
    print(f"\n=== {ticker} ===")
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(start=START_DATE, end=END_DATE, interval='1d')

        if df.empty:
            print("No data")
            continue

        # Clean columns
        df.columns = [c.title().replace(' ', '') for c in df.columns.str.strip()]
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'open'  in cl: rename_map[c] = 'Open'
            if 'close' in cl and 'adj' not in cl: rename_map[c] = 'Close'
            if 'volume'in cl: rename_map[c] = 'Volume'
        df = df.rename(columns=rename_map)

        if not {'Open', 'Close'}.issubset(df.columns):
            print("Missing OHLC")
            continue

        df = df[['Open', 'Close']].dropna()

        # Target: tomorrow open < close ? 1 : 0
        y = (df.Open < df.Close).astype(int).shift(-1).fillna(0)

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

                sw_act, dy_act, sw_prof, dy_prof = simulate(X_test, pd.Series(pred))

                sw_final = sw_prof.iloc[-1]
                dy_final = dy_prof.iloc[-1]

                results.append({
                    'Asset': ticker,
                    'Model': name,
                    'Accuracy': acc,
                    'Swing Profit': sw_final - INITIAL_CAPITAL,
                    'Day Profit':   dy_final - INITIAL_CAPITAL,
                    'Swing CAGR':   cagr(sw_final, INITIAL_CAPITAL, TRADE_YEARS),
                    'Day CAGR':     cagr(dy_final, INITIAL_CAPITAL, TRADE_YEARS),
                    'Swing Max DD': max_dd(sw_prof),
                    'Day Max DD':   max_dd(dy_prof),
                    # For detailed sheets if needed
                    'Swing Actions': sw_act.tolist(),
                    'Day Actions':   dy_act.tolist(),
                    'Swing Profit Series': sw_prof.tolist(),
                })

                print(f"  ✓ {name:22}  acc={acc:.4f}  swing pnl={sw_final-INITIAL_CAPITAL:8.0f}")

            except Exception as e:
                print(f"  ✗ {name:22}  → {str(e)[:60]}")

    except Exception as e:
        print(f"Download failed: {str(e)[:80]}")

# ─── Report & Excel with colors ─────────────────────────────────────────────────

if not results:
    print("\nNo results generated.")
else:
    df = pd.DataFrame(results)

    output = "Forex_Model_Ranking.xlsx"

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        wb = writer.book

        # Main ranking
        df.sort_values('Swing Profit', ascending=False).drop(
            columns=['Swing Actions','Day Actions','Swing Profit Series']
        ).to_excel(writer, sheet_name='Ranking_Swing', index=False)

        df.sort_values('Day Profit', ascending=False).drop(
            columns=['Swing Actions','Day Actions','Swing Profit Series']
        ).to_excel(writer, sheet_name='Ranking_Day', index=False)

        # Top detailed (with actions)
        for col, sheet in [('Swing Profit', 'Top_Swing'), ('Day Profit', 'Top_Day')]:
            top = df.sort_values(col, ascending=False).head(20)
            top[['Asset','Model','Accuracy',col,'Swing CAGR','Day CAGR','Swing Max DD','Day Max DD']]\
               .to_excel(writer, sheet_name=sheet, index=False)

            ws = writer.sheets[sheet]

            # Color profit cells
            profit_col_idx = top.columns.get_loc(col) + 1
            for r in range(2, len(top)+2):
                cell = ws.cell(row=r, column=profit_col_idx)
                val = cell.value
                cell.fill = profit_green if (val or 0) > 0 else profit_red if (val or 0) < 0 else neutral
                cell.alignment = Alignment(horizontal='center')

    print(f"\nReport saved → {output}")
    print(f"Evaluated {len(df)} model × asset combinations")
    print(f"Best swing: {df['Swing Profit'].max():+.0f} ({df.loc[df['Swing Profit'].idxmax(), 'Model']})")
    print(f"Best day  : {df['Day Profit'].max():+.0f} ({df.loc[df['Day Profit'].idxmax(), 'Model']})")