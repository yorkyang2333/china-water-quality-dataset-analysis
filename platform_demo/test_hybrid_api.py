#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM + RandomForest 混合模型 API 测试脚本

使用方式：
  python test_hybrid_api.py
"""

import requests
import json
import time
from typing import Dict, Any

# 配置
BASE_URL = "http://localhost:8000"
USERNAME = "testuser"
PASSWORD = "testpass123"


class HybridModelTester:
    """混合模型API测试类"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()

    def login(self, username: str, password: str) -> bool:
        """登录获取token"""
        try:
            print("🔐 正在登录...")
            response = self.session.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": password}
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.token = data['data']['token']
                    print(f"✅ 登录成功")
                    self.session.headers.update(
                        {"Authorization": f"Bearer {self.token}"}
                    )
                    return True
                else:
                    print(f"❌ 登录失败: {data.get('error')}")
                    return False
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return False

        except Exception as e:
            print(f"❌ 登录异常: {e}")
            return False

    def register_if_needed(self, username: str, password: str) -> bool:
        """如果用户不存在则注册"""
        try:
            print("📝 尝试注册用户...")
            response = self.session.post(
                f"{self.base_url}/api/auth/register",
                json={"username": username, "password": password, "email": f"{username}@test.com"}
            )

            if response.status_code == 201:
                data = response.json()
                if data.get('success'):
                    self.token = data['data']['token']
                    print(f"✅ 用户注册成功")
                    self.session.headers.update(
                        {"Authorization": f"Bearer {self.token}"}
                    )
                    return True
            return False
        except:
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        try:
            print("\n📊 获取混合模型状态...")
            response = self.session.get(
                f"{self.base_url}/api/model/hybrid/status"
            )

            data = response.json()
            if data.get('success'):
                status = data['data']
                print(f"✅ 获取状态成功")
                print(f"   已训练: {status.get('trained', False)}")

                if status.get('trained'):
                    print(f"   R²值: {status['metrics'].get('r2', 'N/A'):.4f}")
                    print(f"   RMSE: {status['metrics'].get('rmse', 'N/A'):.4f}")
                else:
                    print(f"   消息: {status.get('message')}")

                return status
            else:
                print(f"❌ 获取失败: {data.get('error')}")
                return None

        except Exception as e:
            print(f"❌ 异常: {e}")
            return None

    def train_model(self, indicator: str = "COD_mg_L",
                   province: str = None, station: str = None) -> Dict[str, Any]:
        """训练模型"""
        try:
            print(f"\n🚀 训练混合模型 (指标: {indicator})...")
            payload = {"indicator": indicator}
            if province:
                payload["province"] = province
            if station:
                payload["station"] = station

            response = self.session.post(
                f"{self.base_url}/api/model/hybrid/train",
                json=payload,
                timeout=300  # 5分钟超时
            )

            data = response.json()
            if data.get('success'):
                metrics = data['data']
                print(f"✅ 训练成功!")
                print(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                print(f"   MAE: {metrics.get('mae', 'N/A'):.4f}")
                print(f"   R²: {metrics.get('r2', 'N/A'):.4f}")
                print(f"   MSE: {metrics.get('mse', 'N/A'):.4f}")
                return metrics
            else:
                print(f"❌ 训练失败: {data.get('error')}")
                return None

        except requests.Timeout:
            print(f"❌ 请求超时（训练数据可能过多）")
            return None
        except Exception as e:
            print(f"❌ 异常: {e}")
            return None

    def optimize_model(self, indicator: str = "COD_mg_L",
                       n_trials: int = 5,
                       province: str = None, station: str = None) -> Dict[str, Any]:
        """优化模型超参数"""
        try:
            print(f"\n⚡ 优化混合模型 (指标: {indicator}, 试验次数: {n_trials})...")
            payload = {
                "indicator": indicator,
                "n_trials": n_trials
            }
            if province:
                payload["province"] = province
            if station:
                payload["station"] = station

            response = self.session.post(
                f"{self.base_url}/api/model/hybrid/optimize",
                json=payload,
                timeout=600  # 10分钟超时
            )

            data = response.json()
            if data.get('success'):
                result = data['data']
                print(f"✅ 优化完成!")
                print(f"   最佳 R²: {result.get('best_r2', 'N/A'):.4f}")
                print(f"   RMSE: {result['metrics'].get('rmse', 'N/A'):.4f}")
                print(f"   最优参数:")
                params = result.get('best_params', {})
                for key, val in list(params.items())[:5]:  # 只显示前5个参数
                    print(f"      - {key}: {val}")
                print(f"      ...")
                return result
            else:
                print(f"❌ 优化失败: {data.get('error')}")
                if "Optuna" in str(data.get('error', '')):
                    print(f"   提示: 需要安装 optuna, 运行: pip install optuna")
                return None

        except requests.Timeout:
            print(f"❌ 请求超时（优化可能耗时较长）")
            return None
        except Exception as e:
            print(f"❌ 异常: {e}")
            return None

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """进行预测"""
        try:
            print(f"\n🔮 使用混合模型进行预测...")
            payload = {"input_data": input_data}

            response = self.session.post(
                f"{self.base_url}/api/model/hybrid/predict",
                json=payload
            )

            data = response.json()
            if data.get('success'):
                predictions = data['data']
                print(f"✅ 预测成功!")
                print(f"   预测值数量: {len(predictions.get('predictions', []))}")
                print(f"   平均值: {predictions.get('mean', 'N/A'):.4f}")
                print(f"   标准差: {predictions.get('std', 'N/A'):.4f}")
                return predictions
            else:
                print(f"❌ 预测失败: {data.get('error')}")
                return None

        except Exception as e:
            print(f"❌ 异常: {e}")
            return None

    def run_full_test(self):
        """运行完整测试流程"""
        print("\n" + "="*60)
        print("  LSTM + RandomForest 混合模型 API 测试")
        print("="*60)

        # 1. 尝试登录，如果失败则注册
        if not self.login(USERNAME, PASSWORD):
            if not self.register_if_needed(USERNAME, PASSWORD):
                print("❌ 无法登录或注册")
                return

        # 2. 获取模型状态
        self.get_status()

        # 3. 训练模型（使用小数据集测试）
        print("\n" + "-"*60)
        print("测试 1: 训练模型")
        print("-"*60)
        metrics = self.train_model(indicator="COD_mg_L")

        if metrics is None:
            print("⚠️  训练失败，跳过后续测试")
            return

        # 4. 检查训练后的状态
        print("\n" + "-"*60)
        print("测试 2: 检查训练后的状态")
        print("-"*60)
        self.get_status()

        # 5. 尝试优化（如果 optuna 已安装）
        print("\n" + "-"*60)
        print("测试 3: 优化超参数 (仅3次试验)")
        print("-"*60)
        self.optimize_model(indicator="COD_mg_L", n_trials=3)

        # 6. 测试预测（生成样本数据）
        print("\n" + "-"*60)
        print("测试 4: 预测")
        print("-"*60)
        sample_data = [
            {
                "Month": 1,
                "DayOfYear": 15,
                "COD_mg_L": 25.5,
                "COD_mg_L_lag_1": 24.8,
                "COD_mg_L_lag_3": 23.2,
                "COD_mg_L_lag_7": 22.1,
                "COD_mg_L_lag_14": 21.5,
                "COD_mg_L_diff_1": 0.7,
                "COD_mg_L_rolling_mean_7": 23.8,
                "COD_mg_L_rolling_mean_14": 22.9
            }
        ]
        self.predict(sample_data)

        print("\n" + "="*60)
        print("✨ 测试完成!")
        print("="*60)


def main():
    """主函数"""
    print("\n📚 LSTM + RandomForest 混合模型 API 测试脚本\n")

    # 检查服务器是否运行
    try:
        response = requests.get(f"{BASE_URL}/api/model/hybrid/status", timeout=5)
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print(f"   请确保平台已启动: python start_all.py")
        print(f"   或查看服务器是否运行在 {BASE_URL}")
        return

    # 运行测试
    tester = HybridModelTester(BASE_URL)
    tester.run_full_test()


if __name__ == "__main__":
    main()
