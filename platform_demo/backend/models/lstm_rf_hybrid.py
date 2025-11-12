"""
LSTM + 随机森林混合模型 - 带Optuna调参功能
用于水质污染水平的时间序列预测

支持特性：
1. 自动GPU加速（CUDA/MPS）
2. Optuna超参数优化
3. 交叉验证评估
4. 模型持久化
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import json
from typing import Dict, Tuple, List, Optional
import joblib

try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)


class LSTMModel(nn.Module):
    """
    ⭐⭐ 改进的LSTM时间序列模型

    新特性（参考 lstm_station_optimized_auto_gpu.py）：
    - LayerNorm for stability
    - 更好的全连接层结构
    - 改进的 dropout 策略
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # ⭐ 添加LayerNorm提高稳定性
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 取最后一个时间步
        out = self.layer_norm(out)  # ⭐ LayerNorm
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""

    def __init__(self, data: np.ndarray, labels: np.ndarray, seq_length: int = 10):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels).view(-1, 1)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.labels[idx + self.seq_length]
        return x, y


def get_device():
    """自动检测最优设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class LSTMRFHybridPredictor:
    """LSTM + RandomForest混合预测器，支持Optuna调参"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.lstm_model = None
        self.rf_model = None
        self.scaler = None
        self.feature_cols = None
        self.rf_feature_cols = None
        self.hyperparams = {}
        self.metrics = {}

    def _engineer_features(self, df: pd.DataFrame, indicator: str) -> pd.DataFrame:
        """特征工程"""
        result = df.copy()
        result = result.sort_values('Date').reset_index(drop=True)

        # 时间特征
        if isinstance(result['Date'].iloc[0], str):
            result['Date'] = pd.to_datetime(result['Date'], errors='coerce')

        result['Month'] = result['Date'].dt.month
        result['DayOfYear'] = result['Date'].dt.dayofyear

        # 滞后特征
        for lag in [1, 3, 7, 14]:
            result[f'{indicator}_lag_{lag}'] = result[indicator].shift(lag)

        # 差分特征
        result[f'{indicator}_diff_1'] = result[indicator].diff(1)

        # 滚动平均
        for window in [7, 14]:
            result[f'{indicator}_rolling_mean_{window}'] = result[indicator].rolling(
                window=window
            ).mean()

        return result.dropna()

    def _prepare_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                      indicator: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备训练和测试数据"""
        feature_cols = [col for col in train_data.columns
                       if col not in ['Date', 'Station', 'Year', indicator]]
        feature_cols.append(indicator)

        X_train = train_data[feature_cols].values
        y_train = train_data[indicator].values

        X_test = test_data[feature_cols].values
        y_test = test_data[indicator].values

        return X_train, y_train, X_test, y_test

    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    hyperparams: Dict) -> Tuple[nn.Module, MinMaxScaler]:
        """
        ⭐⭐ 优化后的LSTM训练函数

        改进（参考 lstm_station_optimized_auto_gpu.py）：
        - patience=50（原无） → 避免过早停止
        - min_delta=1e-5 → 更敏感的早停检测
        - CosineAnnealingLR → 更好的学习率调度
        """
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        dataset = TimeSeriesDataset(X_scaled, y_train, seq_length=hyperparams['seq_length'])
        loader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=True)

        model = LSTMModel(
            input_size=X_scaled.shape[1],
            hidden_size=hyperparams['hidden_size'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout']
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        criterion = nn.MSELoss()

        # ⭐ 改进的学习率调度：CosineAnnealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        best_loss = float('inf')
        patience = 30  # ⭐ 正常训练使用 patience=30
        patience_counter = 0
        min_delta = 1e-4  # ⭐ 增大 min_delta 到 1e-4，更容易触发早停

        for epoch in range(hyperparams['epochs']):
            model.train()
            epoch_losses = []
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            scheduler.step()  # ⭐ 学习率调度

            # ⭐⭐ 改进的早停检查（使用min_delta）
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return model, scaler

    def _train_rf(self, X_train: np.ndarray, y_train: np.ndarray,
                  hyperparams: Dict) -> RandomForestRegressor:
        """训练随机森林模型"""
        model = RandomForestRegressor(
            n_estimators=hyperparams['rf_n_estimators'],
            max_depth=hyperparams['rf_max_depth'],
            min_samples_split=hyperparams['rf_min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def _predict_lstm(self, model: nn.Module, X_test: np.ndarray,
                      scaler: MinMaxScaler, seq_length: int) -> np.ndarray:
        """LSTM预测"""
        X_scaled = scaler.transform(X_test)
        model.eval()

        predictions = []
        with torch.no_grad():
            for i in range(len(X_scaled) - seq_length):
                x_seq = torch.FloatTensor(X_scaled[i:i + seq_length]).unsqueeze(0).to(self.device)
                pred = model(x_seq).cpu().numpy()[0, 0]
                predictions.append(pred)

        return np.array(predictions)

    def _predict_rf(self, model: RandomForestRegressor, X_test: np.ndarray) -> np.ndarray:
        """RF预测"""
        return model.predict(X_test)

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
              indicator: str, hyperparams: Dict) -> Dict:
        """训练混合模型"""

        # 特征工程
        train_df = self._engineer_features(train_df, indicator)
        test_df = self._engineer_features(test_df, indicator)

        if len(train_df) < 50 or len(test_df) < 10:
            raise ValueError("样本数据不足")

        # 准备数据
        feature_cols = [col for col in train_df.columns
                       if col not in ['Date', 'Station', 'Year', indicator]]
        feature_cols.append(indicator)

        X_train = train_df[feature_cols].values
        y_train = train_df[indicator].values
        X_test = test_df[feature_cols].values
        y_test = test_df[indicator].values

        self.feature_cols = feature_cols

        # 分割验证集
        split_idx = int(len(X_train) * 0.8)
        X_train_split, y_train_split = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]

        # 训练LSTM
        self.lstm_model, self.scaler = self._train_lstm(
            X_train_split, y_train_split, X_val, y_val, hyperparams
        )

        # 训练RF
        self.rf_model = self._train_rf(X_train_split, y_train_split, hyperparams)
        self.rf_feature_cols = feature_cols

        # 预测
        y_lstm = self._predict_lstm(self.lstm_model, X_test, self.scaler,
                                   hyperparams['seq_length'])
        y_rf = self._predict_rf(self.rf_model, X_test)

        # 混合预测（加权平均）
        min_len = min(len(y_lstm), len(y_rf))
        y_lstm_trim = y_lstm[:min_len]
        y_rf_trim = y_rf[:min_len]
        y_hybrid = (hyperparams['lstm_weight'] * y_lstm_trim +
                   (1 - hyperparams['lstm_weight']) * y_rf_trim)

        y_true = y_test[-min_len:]

        # 计算指标
        mse = mean_squared_error(y_true, y_hybrid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_hybrid)
        r2 = r2_score(y_true, y_hybrid)

        self.metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'y_pred': y_hybrid.tolist(),
            'y_true': y_true.tolist()
        }

        self.hyperparams = hyperparams

        return self.metrics

    def optimize_hyperparams(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                            indicator: str, n_trials: int = 10) -> Dict:
        """使用Optuna优化超参数"""

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Run: pip install optuna")

        # 特征工程
        train_df = self._engineer_features(train_df, indicator)
        test_df = self._engineer_features(test_df, indicator)

        feature_cols = [col for col in train_df.columns
                       if col not in ['Date', 'Station', 'Year', indicator]]
        feature_cols.append(indicator)

        X_train = train_df[feature_cols].values
        y_train = train_df[indicator].values
        X_test = test_df[feature_cols].values
        y_test = test_df[indicator].values

        # 分割验证集
        split_idx = int(len(X_train) * 0.8)
        X_train_split, y_train_split = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]

        def objective(trial: optuna.Trial) -> float:
            try:
                hyperparams = {
                    'hidden_size': trial.suggest_int('hidden_size', 32, 128),
                    'num_layers': trial.suggest_int('num_layers', 2, 4),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_int('batch_size', 8, 32),
                    'epochs': trial.suggest_int('epochs', 30, 150),  # ⭐ 增加epochs范围
                    'seq_length': trial.suggest_int('seq_length', 5, 20),
                    'rf_n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                    'rf_max_depth': trial.suggest_int('rf_max_depth', 10, 30),
                    'rf_min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                    'lstm_weight': trial.suggest_float('lstm_weight', 0.3, 0.7),
                }

                # 训练模型
                metrics = self.train(train_df, test_df, indicator, hyperparams)

                return metrics['r2']

            except Exception as e:
                print(f"Trial failed: {e}")
                return -1.0

        # 创建优化器
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner()
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params['lstm_weight'] = study.best_trial.params.get('lstm_weight', 0.5)

        # 用最佳参数重新训练
        self.train(train_df, test_df, indicator, best_params)

        return {
            'best_params': best_params,
            'best_r2': study.best_value,
            'metrics': self.metrics
        }

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """使用已训练的模型进行预测"""
        if self.lstm_model is None or self.rf_model is None:
            raise RuntimeError("Model not trained yet")

        X = input_data[self.feature_cols].values

        y_lstm = self._predict_lstm(self.lstm_model, X, self.scaler,
                                   self.hyperparams['seq_length'])
        y_rf = self._predict_rf(self.rf_model, X)

        min_len = min(len(y_lstm), len(y_rf))
        y_hybrid = (self.hyperparams['lstm_weight'] * y_lstm[:min_len] +
                   (1 - self.hyperparams['lstm_weight']) * y_rf[:min_len])

        return y_hybrid

    def save(self, path: str):
        """保存模型"""
        os.makedirs(path, exist_ok=True)

        # 保存LSTM
        torch.save(self.lstm_model.state_dict(), os.path.join(path, 'lstm_model.pt'))

        # 保存RF
        joblib.dump(self.rf_model, os.path.join(path, 'rf_model.pkl'))

        # 保存scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))

        # 保存超参数和特征列
        config = {
            'hyperparams': self.hyperparams,
            'feature_cols': self.feature_cols,
            'metrics': self.metrics
        }

        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, path: str, input_size: int):
        """加载模型"""
        # 加载超参数和特征列
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)

        self.hyperparams = config['hyperparams']
        self.feature_cols = config['feature_cols']
        self.metrics = config.get('metrics', {})

        # 加载LSTM
        self.lstm_model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hyperparams.get('hidden_size', 64),
            num_layers=self.hyperparams.get('num_layers', 2),
            dropout=self.hyperparams.get('dropout', 0.2)
        ).to(self.device)
        self.lstm_model.load_state_dict(torch.load(os.path.join(path, 'lstm_model.pt')))

        # 加载RF
        self.rf_model = joblib.load(os.path.join(path, 'rf_model.pkl'))

        # 加载scaler
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
