"""
分类器基类
所有模型（传统和深度学习）都应继承此基类，确保接口统一
"""

import os
import json
from abc import ABC, abstractmethod


class BaseWaterQualityClassifier(ABC):
    """
    水质分类器基类

    所有模型应继承此类，实现以下方法：
    - build_model(): 构建模型
    - train(): 训练模型
    - predict(): 预测
    - predict_proba(): 预测概率（可选）
    """

    def __init__(self, model_name, output_dir=None):
        """
        初始化

        参数:
        - model_name: 模型名称（如 'XGBoost', 'LSTM' 等）
        - output_dir: 输出目录（默认为项目根/Outputs/Pollution_Classification/{model_name}）
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_history = {}

        if output_dir is None:
            project_root = self._get_project_root()
            self.output_dir = os.path.join(
                project_root, 'Outputs', 'Pollution_Classification', model_name
            )
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_project_root(self):
        """获取项目根目录（从core模块向上4层）"""
        current_file = os.path.abspath(__file__)
        # core/base_classifier.py -> core -> pollution_level_classification -> models -> 项目根
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

    @abstractmethod
    def build_model(self, **kwargs):
        """
        构建模型

        子类必须实现此方法来构建模型架构
        """
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        训练模型

        参数:
        - X_train, y_train: 训练数据
        - X_val, y_val: 验证数据（可选）
        - **kwargs: 其他超参数

        返回:
        - 训练历史（字典）
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        预测

        参数:
        - X: 输入特征

        返回:
        - 预测标签数组
        """
        pass

    def predict_proba(self, X):
        """
        预测概率（可选实现）

        参数:
        - X: 输入特征

        返回:
        - 各类别的概率

        默认实现：抛出NotImplementedError
        """
        raise NotImplementedError(f"{self.model_name} 不支持概率预测")

    def save_model(self, filename='model.pkl'):
        """
        保存模型

        参数:
        - filename: 保存文件名
        """
        if self.model is None:
            print(f"[错误] 模型未创建，无法保存")
            return False

        import joblib
        path = os.path.join(self.output_dir, filename)
        joblib.dump(self.model, path)
        print(f"[保存] 模型已保存至: {path}")
        return True

    def load_model(self, filename='model.pkl'):
        """
        加载模型

        参数:
        - filename: 加载文件名
        """
        import joblib
        path = os.path.join(self.output_dir, filename)

        if not os.path.exists(path):
            print(f"[错误] 模型文件不存在: {path}")
            return False

        self.model = joblib.load(path)
        self.is_trained = True
        print(f"[加载] 模型已加载: {path}")
        return True

    def save_training_history(self, filename='training_history.json'):
        """
        保存训练历史

        参数:
        - filename: 保存文件名
        """
        path = os.path.join(self.output_dir, filename)

        # 将numpy数组转换为列表以便JSON序列化
        history_json = {}
        for key, value in self.training_history.items():
            if hasattr(value, 'tolist'):
                history_json[key] = value.tolist()
            else:
                history_json[key] = value

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history_json, f, ensure_ascii=False, indent=2)

        print(f"[保存] 训练历史已保存: {path}")

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'output_dir': self.output_dir,
            'model_type': self.__class__.__name__
        }

    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print("\n" + "="*60)
        print(f"模型信息: {info['model_name']}")
        print("="*60)
        for key, value in info.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")


class TraditionalMLClassifier(BaseWaterQualityClassifier):
    """
    传统机器学习分类器基类

    适用于：RandomForest, XGBoost, SVM, LogisticRegression 等
    """

    def predict_proba(self, X):
        """传统ML通常支持概率预测"""
        if not hasattr(self.model, 'predict_proba'):
            return super().predict_proba(X)

        return self.model.predict_proba(X)


class DeepLearningClassifier(BaseWaterQualityClassifier):
    """
    深度学习分类器基类

    适用于：LSTM, CNN, MLP 等
    """

    def __init__(self, model_name, output_dir=None, device='auto'):
        """
        初始化

        参数:
        - device: 计算设备 ('cpu', 'cuda', 'mps', 'auto')
        """
        super().__init__(model_name, output_dir)
        self.device = device

    def set_device(self, device):
        """设置计算设备"""
        self.device = device
        print(f"[设备] 计算设备设置为: {device}")

    def get_device(self):
        """获取当前计算设备"""
        return self.device
