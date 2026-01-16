# =============================================================================
# Optimized Multiple Model-based Binary Classification for Forex Trading
# Refactored Version - January 2026
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

# Consolidated model configurations
MODEL_CONFIGS = [
    ModelConfig("Dummy_MostFrequent", dummy.DummyClassifier, {"strategy": "most_frequent"}),
    ModelConfig("Dummy_Stratified", dummy.DummyClassifier, {"strategy": "stratified"}),
    ModelConfig("Dummy_Uniform", dummy.DummyClassifier, {"strategy": "uniform"}),
    ModelConfig("RF_50_5", ensemble.RandomForestClassifier, 
                {"n_estimators": 50, "max_depth": 5, "n_jobs": -1}),
    ModelConfig("RF_100_10", ensemble.RandomForestClassifier,
                {"n_estimators": 100, "max_depth": 10, "n_jobs": -1}),
    ModelConfig("RF_200_None", ensemble.RandomForestClassifier,
                {"n_estimators": 200, "max_depth": None, "n_jobs": -1}),
    ModelConfig("ET_50_Gini", ensemble.ExtraTreesClassifier,
                {"n_estimators": 50, "criterion": "gini", "n_jobs": -1}),
    ModelConfig("ET_100_Entropy", ensemble.ExtraTreesClassifier,
                {"n_estimators": 100, "criterion": "entropy", "n_jobs": -1}),
    ModelConfig("GB_200_0.05", ensemble.HistGradientBoostingClassifier,
                {"max_iter": 200, "learning_rate": 0.05}),
    ModelConfig("GB_300_0.1", ensemble.HistGradientBoostingClassifier,
                {"max_iter": 300, "learning_rate": 0.1}),
    ModelConfig("LR_LBFGS", linear_model.LogisticRegression,
                {"max_iter": 500, "solver": "lbfgs", "n_jobs": -1}),
    ModelConfig("LR_SAGA_L1", linear_model.LogisticRegression,
                {"max_iter": 500, "solver": "saga", "penalty": "l1"}),
    ModelConfig("Ridge", linear_model.RidgeClassifier,
                {"max_iter": 500, "solver": "sparse_cg"}),
    ModelConfig("KNN_5_Euclidean", neighbors.KNeighborsClassifier,
                {"n_neighbors": 5, "metric": "euclidean", "n_jobs": -1}),
    ModelConfig("KNN_15_Manhattan", neighbors.KNeighborsClassifier,
                {"n_neighbors": 15, "metric": "manhattan", "n_jobs": -1}),
    ModelConfig("KNN_25_Minkowski", neighbors.KNeighborsClassifier,
                {"n_neighbors": 25, "metric": "minkowski", "n_jobs": -1}),
    ModelConfig("GaussianNB", naive_bayes.GaussianNB, {}),
    ModelConfig("BernoulliNB", naive_bayes.BernoulliNB, {}),
    ModelConfig("MultinomialNB", naive_bayes.MultinomialNB, {}),
    ModelConfig("MLP_50", neural_network.MLPClassifier,
                {"hidden_layer_sizes": (50,), "max_iter": 500}),
    ModelConfig("MLP_100", neural_network.MLPClassifier,
                {"hidden_layer_sizes": (100,), "max_iter": 500}),
    ModelConfig("MLP_50_50", neural_network.MLPClassifier,
                {"hidden_layer_sizes": (50, 50), "max_iter": 500}),
    ModelConfig("SVM_Linear", svm.SVC,
                {"kernel": "linear", "max_iter": 2000}),
    ModelConfig("SVM_RBF", svm.SVC,
                {"kernel": "rbf", "gamma": "scale", "max_iter": 2000}),
    ModelConfig("SVM_Poly", svm.SVC,
                {"kernel": "poly", "degree": 3, "max_iter": 2000}),
    ModelConfig("DT_5_Gini", tree.DecisionTreeClassifier,
                {"max_depth": 5, "criterion": "gini"}),
    ModelConfig("DT_10_Entropy", tree.DecisionTreeClassifier,
                {"max_depth": 10, "criterion": "entropy"}),
    ModelConfig("DT_None_Gini", tree.DecisionTreeClassifier,
                {"max_depth": None, "criterion": "gini"})
]

# ─── Excel Styling ──────────────────────────────────────────────────────────
class ExcelStyler:
    """Centralized Excel styling utilities"""
    
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
        """Apply header styling to entire first row"""
        styles = ExcelStyler.get_styles()
        for cell in ws[1]:
            cell.fill = styles['header_fill']
            cell.font = styles['header_font']
            cell.alignment = Alignment(horizontal='center', vertical='center')
    
    @staticmethod
    def color_profit_cells(ws, df):
        """Color profit-related cells based on value"""
        styles = ExcelStyler.get_styles()
        profit_cols = [c for c in df.columns if any(k in c.lower() for k in ['profit', 'cagr'])]
        
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

# ─── Helper Functions ───────────────────────────────────────────────────────
class TradingMetrics:
    """Centralized metric calculations"""
    
    @staticmethod
    def cagr(final: float, years: int, initial: float = None) -> float:
        """Calculate Compound Annual Growth Rate"""
        initial = initial or CONFIG.initial_capital
        if years <= 0 or final <= 0:
            return np.nan
        return (final / initial) ** (1 / years) - 1
    
    @staticmethod
    def max_drawdown(profits: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(profits) < 2:
            return np.nan
        peak = np.maximum.accumulate(profits)
        trough = (peak - profits) / peak
        return np.max(trough) if len(trough) > 0 else np.nan
    
    @staticmethod
    def calculate_buy_hold_profit(df_test: pd.DataFrame) -> float:
        """Calculate buy-and-hold benchmark profit"""
        units = CONFIG.initial_capital / df_test['Open'].iloc[0]
        return (df_test['Close'].iloc[-1] - df_test['Open'].iloc[0]) * units

class DataHandler:
    """Handle data downloading and preprocessing"""
    
    @staticmethod
    def clean_yf_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Download and clean Yahoo Finance data"""
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, interval='1d')
            if df.empty:
                return None
            
            # Clean column names
            df.columns = [col.strip().title() for col in df.columns]
            
            # Standardize column names
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
        """Prepare features and targets for modeling"""
        # Use price difference as target (1 if price increases, 0 otherwise)
        y = (df['Open'] < df['Close']).astype(int).shift(-1).fillna(0)
        return df, y

class TradingSimulator:
    """Handle trade simulations"""
    
    @staticmethod
    def calculate_lot_size(capital: float, price: float, ratio: float = None) -> int:
        """Calculate lot size for trading"""
        ratio = ratio or CONFIG.invest_ratio
        return max(0, int((capital * ratio) / price))
    
    @staticmethod
    def simulate_swing_trading(df: pd.DataFrame, preds: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Simulate swing trading strategy"""
        capital = CONFIG.initial_capital
        position_open = False
        position_size = 0
        capital_history = [capital]
        actions = []
        
        for i in range(1, len(preds)):
            prev_signal = preds.iloc[i-1]
            curr_signal = preds.iloc[i]
            action = "No Action"
            
            # Entry signal
            if not position_open and prev_signal == 1 and (i < 2 or preds.iloc[i-2] == 0):
                action = "Buy"
                position_size = TradingSimulator.calculate_lot_size(
                    capital, df['Open'].iloc[i])
                if position_size > 0:
                    capital -= position_size * df['Open'].iloc[i] * (1 + CONFIG.spread)
                    position_open = True
            
            # Exit signal
            elif position_open and prev_signal == 0 and (i < 2 or preds.iloc[i-2] == 1):
                action = "Sell"
                capital += position_size * df['Open'].iloc[i] * (1 - CONFIG.spread)
                position_open = False
                position_size = 0
                capital_history.append(capital)
            
            actions.append(action)
        
        # Close any open position at the end
        if position_open:
            capital += position_size * df['Open'].iloc[-1] * (1 - CONFIG.spread)
            capital_history.append(capital)
            actions[-1] = "Sell" if len(actions) > 0 else "No Action"
        
        return pd.Series(actions, name='Swing Action'), pd.Series(capital_history, name='Swing Profit')
    
    @staticmethod
    def simulate_day_trading(df: pd.DataFrame, preds: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Simulate day trading strategy"""
        capital = CONFIG.initial_capital
        capital_history = [capital]
        actions = []
        
        for i in range(1, len(preds)):
            prev_signal = preds.iloc[i-1]
            action = "No Action"
            
            if prev_signal == 1:
                action = "Buy & Sell"
                lot_size = TradingSimulator.calculate_lot_size(
                    capital, df['Open'].iloc[i])
                if lot_size > 0:
                    pnl = lot_size * (df['Close'].iloc[i] - df['Open'].iloc[i]) * (1 - CONFIG.spread)
                    capital += pnl
                capital_history.append(capital)
            
            actions.append(action)
        
        return pd.Series(actions, name='Day Action'), pd.Series(capital_history, name='Day Profit')
    
    @staticmethod
    def simulate_trading(df: pd.DataFrame, preds: pd.Series) -> TradeResult:
        """Run both trading simulations"""
        swing_actions, swing_profits = TradingSimulator.simulate_swing_trading(df, preds)
        day_actions, day_profits = TradingSimulator.simulate_day_trading(df, preds)
        
        return TradeResult(
            swing_profit=swing_profits.iloc[-1] - CONFIG.initial_capital,
            day_profit=day_profits.iloc[-1] - CONFIG.initial_capital,
            swing_cagr=TradingMetrics.cagr(swing_profits.iloc[-1], CONFIG.trade_years),
            day_cagr=TradingMetrics.cagr(day_profits.iloc[-1], CONFIG.trade_years),
            swing_mdd=TradingMetrics.max_drawdown(swing_profits.values),
            day_mdd=TradingMetrics.max_drawdown(day_profits.values),
            swing_max_profit=swing_profits.max() - CONFIG.initial_capital,
            day_max_profit=day_profits.max() - CONFIG.initial_capital,
            swing_max_loss=swing_profits.min() - CONFIG.initial_capital,
            day_max_loss=day_profits.min() - CONFIG.initial_capital,
            swing_trades=(swing_actions != "No Action").sum(),
            day_trades=(day_actions != "No Action").sum(),
            swing_profit_history=swing_profits.tolist(),
            day_profit_history=day_profits.tolist(),
            swing_actions=swing_actions,
            day_actions=day_actions
        )

class ExcelExporter:
    """Handle Excel export operations"""
    
    TRADE_STYLES = {
        "swing": {"bg": "#f0f8ff", "pos": "darkgreen", "neg": "red"},
        "day": {"bg": "#fffacd", "pos": "blue", "neg": "purple"},
    }
    
    @staticmethod
    def create_profit_graph(sorted_df: pd.DataFrame, metric: str, 
                          label: str, trade_type: str) -> BytesIO:
        """Create profit graph for top performers"""
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
            ax.plot(profit_history, 
                   color=style["pos"] if row[profit_col] > 0 else style["neg"])
            
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
    def export_to_excel(ranking_df: pd.DataFrame, filename: str = "Forex_Top10_EveryMetric_WithGraphs.xlsx"):
        """Export results to Excel with graphs"""
        
        # Define metrics for each trade type
        TRADE_METRICS = {
            "swing": ['accuracy', 'swing_profit', 'swing_cagr', 'swing_mdd',
                     'swing_max_profit', 'swing_max_loss', 'swing_trades'],
            "day": ['accuracy', 'day_profit', 'day_cagr', 'day_mdd',
                   'day_max_profit', 'day_max_loss', 'day_trades']
        }
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            wb = writer.book
            
            # 1. Full ranking (without histories)
            full_df = ranking_df.drop(columns=['swing_profit_history', 'day_profit_history'])
            full_df.to_excel(writer, sheet_name='Full_Ranking', index=False)
            
            # Apply styling to full ranking
            full_ws = writer.sheets['Full_Ranking']
            ExcelStyler.apply_header_style(full_ws)
            ExcelStyler.color_profit_cells(full_ws, full_df)
            
            # 2. Top-10 rankings by metric with graphs
            for trade_type, metrics in TRADE_METRICS.items():
                for metric in metrics:
                    for ascending, label in [(False, 'Greatest'), (True, 'Fewest')]:
                        sorted_df = ranking_df.sort_values(
                            by=metric, ascending=ascending
                        ).head(CONFIG.show_top)
                        
                        sheet_name = f"{label}_{metric}"[:31]
                        # Export without histories for the table
                        sorted_df.drop(
                            columns=['swing_profit_history', 'day_profit_history']
                        ).to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        ws = writer.sheets[sheet_name]
                        ExcelStyler.apply_header_style(ws)
                        ExcelStyler.color_profit_cells(ws, sorted_df)
                        
                        # Add graph
                        graph_buf = ExcelExporter.create_profit_graph(
                            sorted_df, metric, label, trade_type
                        )
                        if graph_buf:
                            start_cell = 'A15' if trade_type == "swing" else 'A55'
                            ws.add_image(Image(graph_buf), start_cell)
        
        print(f"\nReport saved to: {filename}")
        print(f"Created {sum(len(m) for m in TRADE_METRICS.values()) * 2} top-10 sheets")

# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    """Main execution function"""
    
    # Download data
    print("Downloading Forex data...")
    asset_data = []
    for ticker in MAJOR_FOREX:
        print(f"  {ticker}...", end="")
        df = DataHandler.clean_yf_data(ticker, CONFIG.start_date, CONFIG.end_date)
        if df is not None:
            print(f" ✓ ({len(df):,} rows)")
            asset_data.append((ticker, df))
        else:
            print(" ✗")
    
    if not asset_data:
        raise ValueError("No valid data downloaded.")
    
    # Training and ranking
    print("\nTraining models and simulating trades...")
    ranking_rows = []
    
    for ticker, df in asset_data:
        print(f"\n{ticker}:")
        
        # Prepare data
        X, y = DataHandler.prepare_features_targets(df)
        train_size = int(len(X) * CONFIG.train_split)
        
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Calculate buy-and-hold benchmark
        bh_profit = TradingMetrics.calculate_buy_hold_profit(X_test)
        
        # Train and evaluate each model
        for model_config in MODEL_CONFIGS:
            try:
                model = model_config.create()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                pred = np.round(np.clip(pred, 0, 1)).astype(int)
                accuracy = metrics.accuracy_score(y_test, pred)
                
                # Simulate trading
                trade_result = TradingSimulator.simulate_trading(X_test, pd.Series(pred))
                
                # Compile results
                result_row = {
                    'asset': ticker,
                    'model': model_config.name,
                    'duration_years': CONFIG.trade_years,
                    'accuracy': accuracy,
                    'buy_hold_profit': bh_profit,
                    **trade_result.__dict__
                }
                
                ranking_rows.append(result_row)
                print(f"  ✓ {model_config.name:25} "
                      f"acc={accuracy:.4f} "
                      f"swing={trade_result.swing_profit:+8.0f} "
                      f"day={trade_result.day_profit:+8.0f}")
                
            except Exception as e:
                print(f"  ✗ {model_config.name:25} → {str(e)[:50]}")
    
    # Create DataFrame and export
    ranking_df = pd.DataFrame(ranking_rows)
    
    # Export to Excel
    ExcelExporter.export_to_excel(ranking_df)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS:")
    print("="*60)
    
    best_swing = ranking_df.loc[ranking_df['swing_profit'].idxmax()]
    best_day = ranking_df.loc[ranking_df['day_profit'].idxmax()]
    
    print(f"\nBest Swing Trading:")
    print(f"  Model: {best_swing['model']} on {best_swing['asset']}")
    print(f"  Profit: ${best_swing['swing_profit']:,.0f}")
    print(f"  CAGR: {best_swing['swing_cagr']:.2%}")
    
    print(f"\nBest Day Trading:")
    print(f"  Model: {best_day['model']} on {best_day['asset']}")
    print(f"  Profit: ${best_day['day_profit']:,.0f}")
    print(f"  CAGR: {best_day['day_cagr']:.2%}")
    
    print(f"\nBuy-and-Hold Average Profit: "
          f"${ranking_df['buy_hold_profit'].mean():,.0f}")

if __name__ == "__main__":
    main()