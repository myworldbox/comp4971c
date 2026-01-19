# =============================================================================
# Optimized Multiple Model-based Binary Classification for Forex Trading
# Refactored + Rich Classification Metrics - January 2026
# =============================================================================

import math
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
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

# ─── Data Classes for Configuration ──────────────────────────────────────────
@dataclass
class TradingConfig:
    """Centralized configuration for trading parameters"""
    initial_capital: float = 10000
    invest_ratio: float = 0.5
    train_split: float = 0.5
    risk_free_rate: float = 0.02
    spread: float = 0.0001
    show_top: int = 10
    start_year: int = 2004
    trade_years: int = 20
   
    def __post_init__(self):
        self.start_date = f"{self.start_year}-01-01"
        self.end_date = f"{self.start_year + self.trade_years}-01-01"

@dataclass
class ModelConfig:
    """Model configuration with factory methods"""
    name: str
    model_class: Any
    params: Dict[str, Any] = field(default_factory=dict)
   
    def create(self):
        return self.model_class(**self.params)

@dataclass
class TradeResult:
    """Container for trade simulation results"""
    swing_profit: float
    day_profit: float
    swing_cagr: float
    day_cagr: float
    swing_mdd: float
    day_mdd: float
    swing_max_profit: float
    day_max_profit: float
    swing_max_loss: float
    day_max_loss: float
    swing_trades: int
    day_trades: int
    swing_profit_history: List[float]
    day_profit_history: List[float]
    swing_actions: pd.Series
    day_actions: pd.Series

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG = TradingConfig()

MAJOR_FOREX = [
    'USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCHF=X',
    'USDCAD=X', 'EURJPY=X', 'GBPJPY=X', 'EURCHF=X', 'NZDUSD=X',
]

# Consolidated model configurations (same as original)
MODEL_CONFIGS = [
    ModelConfig("Dummy_MostFrequent", dummy.DummyClassifier, {"strategy": "most_frequent"}),
    ModelConfig("Dummy_Stratified", dummy.DummyClassifier, {"strategy": "stratified"}),
    ModelConfig("Dummy_Uniform", dummy.DummyClassifier, {"strategy": "uniform"}),
    ModelConfig("RF_50_5", ensemble.RandomForestClassifier, {"n_estimators": 50, "max_depth": 5, "n_jobs": -1}),
    ModelConfig("RF_100_10", ensemble.RandomForestClassifier, {"n_estimators": 100, "max_depth": 10, "n_jobs": -1}),
    ModelConfig("RF_200_None", ensemble.RandomForestClassifier, {"n_estimators": 200, "max_depth": None, "n_jobs": -1}),
    ModelConfig("ET_50_Gini", ensemble.ExtraTreesClassifier, {"n_estimators": 50, "criterion": "gini", "n_jobs": -1}),
    ModelConfig("ET_100_Entropy", ensemble.ExtraTreesClassifier, {"n_estimators": 100, "criterion": "entropy", "n_jobs": -1}),
    ModelConfig("GB_200_0.05", ensemble.HistGradientBoostingClassifier, {"max_iter": 200, "learning_rate": 0.05}),
    ModelConfig("GB_300_0.1", ensemble.HistGradientBoostingClassifier, {"max_iter": 300, "learning_rate": 0.1}),
    ModelConfig("LR_LBFGS", linear_model.LogisticRegression, {"max_iter": 500, "solver": "lbfgs", "n_jobs": -1}),
    ModelConfig("LR_SAGA_L1", linear_model.LogisticRegression, {"max_iter": 500, "solver": "saga", "penalty": "l1"}),
    ModelConfig("Ridge", linear_model.RidgeClassifier, {"max_iter": 500, "solver": "sparse_cg"}),
    ModelConfig("KNN_5_Euclidean", neighbors.KNeighborsClassifier, {"n_neighbors": 5, "metric": "euclidean", "n_jobs": -1}),
    ModelConfig("KNN_15_Manhattan", neighbors.KNeighborsClassifier, {"n_neighbors": 15, "metric": "manhattan", "n_jobs": -1}),
    ModelConfig("KNN_25_Minkowski", neighbors.KNeighborsClassifier, {"n_neighbors": 25, "metric": "minkowski", "n_jobs": -1}),
    ModelConfig("GaussianNB", naive_bayes.GaussianNB, {}),
    ModelConfig("BernoulliNB", naive_bayes.BernoulliNB, {}),
    ModelConfig("MultinomialNB", naive_bayes.MultinomialNB, {}),
    ModelConfig("MLP_50", neural_network.MLPClassifier, {"hidden_layer_sizes": (50,), "max_iter": 500}),
    ModelConfig("MLP_100", neural_network.MLPClassifier, {"hidden_layer_sizes": (100,), "max_iter": 500}),
    ModelConfig("MLP_50_50", neural_network.MLPClassifier, {"hidden_layer_sizes": (50, 50), "max_iter": 500}),
    ModelConfig("SVM_Linear", svm.SVC, {"kernel": "linear", "max_iter": 2000, "probability": True}),
    ModelConfig("SVM_RBF", svm.SVC, {"kernel": "rbf", "gamma": "scale", "max_iter": 2000, "probability": True}),
    ModelConfig("SVM_Poly", svm.SVC, {"kernel": "poly", "degree": 3, "max_iter": 2000, "probability": True}),
    ModelConfig("DT_5_Gini", tree.DecisionTreeClassifier, {"max_depth": 5, "criterion": "gini"}),
    ModelConfig("DT_10_Entropy", tree.DecisionTreeClassifier, {"max_depth": 10, "criterion": "entropy"}),
    ModelConfig("DT_None_Gini", tree.DecisionTreeClassifier, {"max_depth": None, "criterion": "gini"})
]

# ─── Rich Classification Metrics ─────────────────────────────────────────────
def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, model=None, X_test=None) -> dict:
    """Compute a broad set (20+) of binary classification metrics"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    if len(y_true) == 0:
        return {k: np.nan for k in [
            'n_samples', 'pos_ratio_true', 'pos_ratio_pred', 'accuracy', 'balanced_accuracy',
            'cohen_kappa', 'matthews_cc', 'precision', 'recall', 'f1', 'f2', 'f05',
            'precision_neg', 'recall_neg', 'roc_auc', 'average_precision', 'log_loss',
            'brier_score', 'hamming_loss', 'zero_one_loss', 'jaccard', 'support_pos', 'support_neg'
        ]}
    
    n_pos_true = np.sum(y_true == 1)
    n_neg_true = np.sum(y_true == 0)
    n_pos_pred = np.sum(y_pred == 1)
    
    metrics_dict = {
        'n_samples': len(y_true),
        'support_pos': n_pos_true,
        'support_neg': n_neg_true,
        'pos_ratio_true': n_pos_true / len(y_true) if len(y_true) > 0 else np.nan,
        'pos_ratio_pred': n_pos_pred / len(y_true) if len(y_true) > 0 else np.nan,
        
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'cohen_kappa': metrics.cohen_kappa_score(y_true, y_pred),
        'matthews_cc': metrics.matthews_corrcoef(y_true, y_pred),
        
        'precision': metrics.precision_score(y_true, y_pred, zero_division=0),
        'recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'f1': metrics.f1_score(y_true, y_pred, zero_division=0),
        'f2': metrics.fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        'f05': metrics.fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        
        'precision_neg': metrics.precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'recall_neg': metrics.recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        
        'hamming_loss': metrics.hamming_loss(y_true, y_pred),
        'zero_one_loss': metrics.zero_one_loss(y_true, y_pred),
        'jaccard': metrics.jaccard_score(y_true, y_pred, zero_division=0),
        
        'roc_auc': np.nan,
        'average_precision': np.nan,
        'log_loss': np.nan,
        'brier_score': np.nan,
    }
    
    # Probabilistic metrics when available
    if model is not None and hasattr(model, "predict_proba") and X_test is not None:
        try:
            proba_pos = model.predict_proba(X_test)[:, 1]
            metrics_dict.update({
                'roc_auc': metrics.roc_auc_score(y_true, proba_pos),
                'average_precision': metrics.average_precision_score(y_true, proba_pos),
                'log_loss': metrics.log_loss(y_true, proba_pos),
                'brier_score': metrics.brier_score_loss(y_true, proba_pos),
            })
        except:
            pass
    
    return metrics_dict

# ─── Excel Styling (unchanged) ───────────────────────────────────────────────
class ExcelStyler:
    @staticmethod
    def get_styles():
        return {
            'header_fill': PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid"),
            'header_font': Font(bold=True, color="FFFFFF"),
            'profit_green': PatternFill(start_color="006400", end_color="006400", fill_type="solid"),
            'profit_red': PatternFill(start_color="8B0000", end_color="8B0000", fill_type="solid"),
            'neutral': PatternFill(start_color="333333", end_color="333333", fill_type="solid")
        }
   
    @staticmethod
    def apply_header_style(ws):
        styles = ExcelStyler.get_styles()
        for cell in ws[1]:
            cell.fill = styles['header_fill']
            cell.font = styles['header_font']
            cell.alignment = Alignment(horizontal='center', vertical='center')
   
    @staticmethod
    def color_profit_cells(ws, df):
        styles = ExcelStyler.get_styles()
        profit_cols = [c for c in df.columns if any(k in c.lower() for k in ['profit', 'cagr', 'mdd'])]
        for col_name in profit_cols:
            col_idx = df.columns.get_loc(col_name) + 1
            for r in range(2, len(df) + 2):
                cell = ws.cell(row=r, column=col_idx)
                if isinstance(cell.value, (int, float)):
                    if cell.value > 0:
                        cell.fill = styles['profit_green']
                    elif cell.value < 0:
                        cell.fill = styles['profit_red']
                    else:
                        cell.fill = styles['neutral']

# ─── Helper Functions (mostly unchanged) ─────────────────────────────────────
class TradingMetrics:
    @staticmethod
    def cagr(final: float, years: int, initial: float = None) -> float:
        initial = initial or CONFIG.initial_capital
        if years <= 0 or final <= 0:
            return np.nan
        return (final / initial) ** (1 / years) - 1
   
    @staticmethod
    def max_drawdown(profits: np.ndarray) -> float:
        if len(profits) < 2:
            return np.nan
        peak = np.maximum.accumulate(profits)
        trough = (peak - profits) / peak
        return np.max(trough) if len(trough) > 0 else np.nan
   
    @staticmethod
    def calculate_buy_hold_profit(df_test: pd.DataFrame) -> float:
        units = CONFIG.initial_capital / df_test['Open'].iloc[0]
        return (df_test['Close'].iloc[-1] - df_test['Open'].iloc[0]) * units

class DataHandler:
    @staticmethod
    def clean_yf_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, interval='1d')
            if df.empty:
                return None
           
            df.columns = [col.strip().title() for col in df.columns]
            column_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    column_map[col] = 'Open'
                elif 'close' in col_lower and 'adj' not in col_lower:
                    column_map[col] = 'Close'
           
            df = df.rename(columns=column_map)
           
            if {'Open', 'Close'}.issubset(df.columns):
                return df[['Open', 'Close']].dropna()
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
        return None
   
    @staticmethod
    def prepare_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        y = (df['Open'] < df['Close']).astype(int).shift(-1).fillna(0)
        return df, y

class TradingSimulator:
    @staticmethod
    def calculate_lot_size(capital: float, price: float, ratio: float = None) -> int:
        ratio = ratio or CONFIG.invest_ratio
        return max(0, int((capital * ratio) / price))
   
    @staticmethod
    def simulate_swing_trading(df: pd.DataFrame, preds: pd.Series) -> Tuple[pd.Series, pd.Series]:
        capital = CONFIG.initial_capital
        position_open = False
        position_size = 0
        capital_history = [capital]
        actions = []
       
        for i in range(1, len(preds)):
            prev_signal = preds.iloc[i-1]
            curr_signal = preds.iloc[i]
            action = "No Action"
           
            if not position_open and prev_signal == 1 and (i < 2 or preds.iloc[i-2] == 0):
                action = "Buy"
                position_size = TradingSimulator.calculate_lot_size(capital, df['Open'].iloc[i])
                if position_size > 0:
                    capital -= position_size * df['Open'].iloc[i] * (1 + CONFIG.spread)
                    position_open = True
           
            elif position_open and prev_signal == 0 and (i < 2 or preds.iloc[i-2] == 1):
                action = "Sell"
                capital += position_size * df['Open'].iloc[i] * (1 - CONFIG.spread)
                position_open = False
                position_size = 0
                capital_history.append(capital)
           
            actions.append(action)
       
        if position_open:
            capital += position_size * df['Open'].iloc[-1] * (1 - CONFIG.spread)
            capital_history.append(capital)
            if actions:
                actions[-1] = "Sell"
       
        return pd.Series(actions, name='Swing Action'), pd.Series(capital_history, name='Swing Profit')
   
    @staticmethod
    def simulate_day_trading(df: pd.DataFrame, preds: pd.Series) -> Tuple[pd.Series, pd.Series]:
        capital = CONFIG.initial_capital
        capital_history = [capital]
        actions = []
       
        for i in range(1, len(preds)):
            prev_signal = preds.iloc[i-1]
            action = "No Action"
           
            if prev_signal == 1:
                action = "Buy & Sell"
                lot_size = TradingSimulator.calculate_lot_size(capital, df['Open'].iloc[i])
                if lot_size > 0:
                    pnl = lot_size * (df['Close'].iloc[i] - df['Open'].iloc[i]) * (1 - CONFIG.spread)
                    capital += pnl
                capital_history.append(capital)
           
            actions.append(action)
       
        return pd.Series(actions, name='Day Action'), pd.Series(capital_history, name='Day Profit')
   
    @staticmethod
    def simulate_trading(df: pd.DataFrame, preds: pd.Series) -> TradeResult:
        swing_actions, swing_profits = TradingSimulator.simulate_swing_trading(df, preds)
        day_actions, day_profits = TradingSimulator.simulate_day_trading(df, preds)
       
        return TradeResult(
            swing_profit=swing_profits.iloc[-1] - CONFIG.initial_capital if len(swing_profits) > 0 else 0,
            day_profit=day_profits.iloc[-1] - CONFIG.initial_capital if len(day_profits) > 0 else 0,
            swing_cagr=TradingMetrics.cagr(swing_profits.iloc[-1], CONFIG.trade_years) if len(swing_profits) > 0 else np.nan,
            day_cagr=TradingMetrics.cagr(day_profits.iloc[-1], CONFIG.trade_years) if len(day_profits) > 0 else np.nan,
            swing_mdd=TradingMetrics.max_drawdown(swing_profits.values),
            day_mdd=TradingMetrics.max_drawdown(day_profits.values),
            swing_max_profit=swing_profits.max() - CONFIG.initial_capital if len(swing_profits) > 0 else 0,
            day_max_profit=day_profits.max() - CONFIG.initial_capital if len(day_profits) > 0 else 0,
            swing_max_loss=swing_profits.min() - CONFIG.initial_capital if len(swing_profits) > 0 else 0,
            day_max_loss=day_profits.min() - CONFIG.initial_capital if len(day_profits) > 0 else 0,
            swing_trades=(swing_actions != "No Action").sum(),
            day_trades=(day_actions != "No Action").sum(),
            swing_profit_history=swing_profits.tolist(),
            day_profit_history=day_profits.tolist(),
            swing_actions=swing_actions,
            day_actions=day_actions
        )

# ─── Excel Exporter (slightly extended to handle more columns) ────────────────
class ExcelExporter:
    TRADE_STYLES = {
        "swing": {"bg": "#f0f8ff", "pos": "darkgreen", "neg": "red"},
        "day": {"bg": "#fffacd", "pos": "blue", "neg": "purple"},
    }
   
    @staticmethod
    def create_profit_graph(sorted_df: pd.DataFrame, metric: str, label: str, trade_type: str) -> BytesIO:
        history_col = f"{trade_type}_profit_history"
        profit_col = f"{trade_type}_profit"
       
        if sorted_df.empty or history_col not in sorted_df.columns:
            return None
       
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"{label} Top {CONFIG.show_top} - {metric}", fontsize=14)
        axes = axes.flat
       
        style = ExcelExporter.TRADE_STYLES[trade_type]
       
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            if i >= len(axes):
                break
            ax = axes[i]
            ax.set_facecolor(style["bg"])
           
            profit_history = row[history_col]
            ax.plot(profit_history, color=style["pos"] if row[profit_col] > 0 else style["neg"])
           
            title = f"{row['asset']}\n{row['model']}\nProfit: {row[profit_col]:.0f}"
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
       
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return buf
   
    @staticmethod
    def export_to_excel(ranking_df: pd.DataFrame, filename: str = "Forex_Top10_EveryMetric_Rich_2026.xlsx"):
        TRADE_METRICS = {
            "swing": ['accuracy', 'matthews_cc', 'f1', 'f2', 'swing_profit', 'swing_cagr', 'swing_mdd',
                      'swing_max_profit', 'swing_max_loss', 'swing_trades'],
            "day": ['accuracy', 'matthews_cc', 'f1', 'f2', 'day_profit', 'day_cagr', 'day_mdd',
                    'day_max_profit', 'day_max_loss', 'day_trades']
        }
       
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            wb = writer.book
           
            # 1. Full ranking (without history columns to keep file size reasonable)
            drop_cols = ['swing_profit_history', 'day_profit_history', 'swing_actions', 'day_actions']
            full_df = ranking_df.drop(columns=[c for c in drop_cols if c in ranking_df.columns])
            full_df.to_excel(writer, sheet_name='Full_Ranking', index=False)
           
            full_ws = writer.sheets['Full_Ranking']
            ExcelStyler.apply_header_style(full_ws)
            ExcelStyler.color_profit_cells(full_ws, full_df)
           
            # 2. Top-10 sheets by selected metrics
            for trade_type, metrics_list in TRADE_METRICS.items():
                for metric in metrics_list:
                    for ascending, label in [(False, 'Greatest'), (True, 'Fewest')]:
                        sorted_df = ranking_df.sort_values(by=metric, ascending=ascending).head(CONFIG.show_top)
                        sheet_name = f"{label}_{metric}"[:31]
                       
                        sorted_df.drop(columns=[c for c in drop_cols if c in sorted_df.columns]).to_excel(
                            writer, sheet_name=sheet_name, index=False)
                       
                        ws = writer.sheets[sheet_name]
                        ExcelStyler.apply_header_style(ws)
                        ExcelStyler.color_profit_cells(ws, sorted_df)
                       
                        graph_buf = ExcelExporter.create_profit_graph(sorted_df, metric, label, trade_type)
                        if graph_buf:
                            start_cell = 'A' + ('15' if trade_type == "swing" else '55')
                            ws.add_image(Image(graph_buf), start_cell)
       
        print(f"\nReport saved to: {filename}")
        print(f"Created {sum(len(m) for m in TRADE_METRICS.values()) * 2} top-10 sheets + full ranking")

# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    print("Downloading Forex data...")
    asset_data = []
    for ticker in MAJOR_FOREX:
        print(f" {ticker}...", end="")
        df = DataHandler.clean_yf_data(ticker, CONFIG.start_date, CONFIG.end_date)
        if df is not None:
            print(f" ✓ ({len(df):,} rows)")
            asset_data.append((ticker, df))
        else:
            print(" ✗")
   
    if not asset_data:
        raise ValueError("No valid data downloaded.")
   
    print("\nTraining models and simulating trades...")
    ranking_rows = []
   
    for ticker, df in asset_data:
        print(f"\n{ticker}:")
       
        X, y = DataHandler.prepare_features_targets(df)
        train_size = int(len(X) * CONFIG.train_split)
       
        X_train = X.iloc[:train_size]
        X_test  = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test  = y.iloc[train_size:]
       
        bh_profit = TradingMetrics.calculate_buy_hold_profit(X_test)
       
        for model_config in MODEL_CONFIGS:
            try:
                model = model_config.create()
                model.fit(X_train, y_train)
                
                # Hard class predictions
                pred = model.predict(X_test)
                pred = np.clip(np.round(pred), 0, 1).astype(int)
                
                # Rich metrics
                class_metrics = compute_classification_metrics(
                    y_test.values, pred, model=model, X_test=X_test
                )
                
                # Trading simulation
                trade_result = TradingSimulator.simulate_trading(X_test, pd.Series(pred, index=X_test.index))
                
                result_row = {
                    'asset': ticker,
                    'model': model_config.name,
                    'duration_years': CONFIG.trade_years,
                    'buy_hold_profit': bh_profit,
                    **class_metrics,
                    **trade_result.__dict__
                }
                
                ranking_rows.append(result_row)
                
                print(f" ✓ {model_config.name:25} "
                      f"acc={class_metrics['accuracy']:.4f} "
                      f"MCC={class_metrics.get('matthews_cc', np.nan):.4f} "
                      f"F1={class_metrics.get('f1', np.nan):.4f} "
                      f"swing ${trade_result.swing_profit:,.0f} "
                      f"day ${trade_result.day_profit:,.0f}")
               
            except Exception as e:
                print(f" ✗ {model_config.name:25} → {str(e)[:60]}...")
   
    ranking_df = pd.DataFrame(ranking_rows)
   
    ExcelExporter.export_to_excel(ranking_df)
   
    print("\n" + "="*70)
    print("SUMMARY – BEST MODELS")
    print("="*70)
   
    if not ranking_df.empty:
        best_swing = ranking_df.loc[ranking_df['swing_profit'].idxmax()]
        best_day   = ranking_df.loc[ranking_df['day_profit'].idxmax()]
        
        print(f"Best Swing Trading:")
        print(f"  Model: {best_swing['model']} @ {best_swing['asset']}")
        print(f"  Profit: ${best_swing['swing_profit']:,.0f}")
        print(f"  CAGR:   {best_swing['swing_cagr']:.2%}")
        print(f"  MCC:    {best_swing.get('matthews_cc', 'N/A'):.4f}")
        print(f"  F1:     {best_swing.get('f1', 'N/A'):.4f}")
        
        print(f"\nBest Day Trading:")
        print(f"  Model: {best_day['model']} @ {best_day['asset']}")
        print(f"  Profit: ${best_day['day_profit']:,.0f}")
        print(f"  CAGR:   {best_day['day_cagr']:.2%}")
        print(f"  MCC:    {best_day.get('matthews_cc', 'N/A'):.4f}")
        print(f"  F1:     {best_day.get('f1', 'N/A'):.4f}")
        
        print(f"\nBuy-and-Hold Average: ${ranking_df['buy_hold_profit'].mean():,.0f}")

if __name__ == "__main__":
    main()