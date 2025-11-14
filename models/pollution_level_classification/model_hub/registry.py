"""
模型注册表
管理所有注册的模型信息
"""

import json
import os
from datetime import datetime


class ModelRegistry:
    """
    模型注册表

    功能：
    1. 注册模型信息
    2. 查询模型元数据
    3. 生成注册报告
    """

    def __init__(self):
        """初始化"""
        self.models = {}
        self.registry_time = datetime.now().isoformat()

    def register(self, model_id, model_instance, category='custom', metadata=None):
        """
        注册模型

        参数:
        - model_id: 模型ID
        - model_instance: 模型实例
        - category: 模型类别
        - metadata: 元数据字典
        """
        self.models[model_id] = {
            'id': model_id,
            'name': getattr(model_instance, 'model_name', model_id),
            'category': category,
            'class': model_instance.__class__.__name__,
            'registered_time': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

    def get_model_info(self, model_id):
        """获取模型信息"""
        return self.models.get(model_id)

    def list_models(self, category=None):
        """
        列出模型

        参数:
        - category: 按类别筛选（可选）

        返回:
        - 模型ID列表
        """
        models = self.models.values()

        if category:
            models = [m for m in models if m['category'] == category]

        return [m['id'] for m in models]

    def get_statistics(self):
        """获取注册统计"""
        total = len(self.models)

        by_category = {}
        for model_info in self.models.values():
            category = model_info['category']
            by_category[category] = by_category.get(category, 0) + 1

        return {
            'total_models': total,
            'by_category': by_category
        }

    def print_summary(self):
        """打印摘要"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("模型注册表摘要")
        print("="*60)
        print(f"总模型数: {stats['total_models']}")
        print("\n按类别分布:")
        for category, count in stats['by_category'].items():
            print(f"  {category}: {count}")
        print("="*60 + "\n")

    def save_registry(self, filepath):
        """保存注册表"""
        registry_data = {
            'registry_time': self.registry_time,
            'total_models': len(self.models),
            'models': list(self.models.values())
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, ensure_ascii=False, indent=2)

        print(f"[保存] 注册表已保存: {filepath}")

    def load_registry(self, filepath):
        """加载注册表"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for model_data in data.get('models', []):
            model_id = model_data['id']
            self.models[model_id] = model_data

        print(f"[加载] 注册表已加载: {filepath}")
