<template>
  <div class="prediction-container">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card class="prediction-card equal-height-card">
          <template #header>
            <div class="card-header">
              <span>污染等级预测</span>
            </div>
          </template>

          <el-form :model="pollutionForm" label-width="140px" v-loading="loading">
            <el-form-item label="水质指数">
              <el-input-number v-model="pollutionForm.waterQualityIndex" :min="0" :max="200" />
            </el-form-item>
            <el-form-item label="COD (mg/L)">
              <el-input-number v-model="pollutionForm.cod" :min="0" :max="100" :precision="2" />
            </el-form-item>
            <el-form-item label="pH值">
              <el-input-number v-model="pollutionForm.ph" :min="0" :max="14" :precision="1" />
            </el-form-item>
            <el-form-item label="溶解氧 (mg/L)">
              <el-input-number v-model="pollutionForm.dissolvedOxygen" :min="0" :max="20" :precision="2" />
            </el-form-item>
            <el-form-item label="总氮 (mg/L)">
              <el-input-number v-model="pollutionForm.totalNitrogen" :min="0" :max="50" :precision="2" />
            </el-form-item>
            <el-form-item label="总磷 (mg/L)">
              <el-input-number v-model="pollutionForm.totalPhosphorus" :min="0" :max="10" :precision="3" />
            </el-form-item>
            <el-form-item label="浊度 (NTU)">
              <el-input-number v-model="pollutionForm.turbidity" :min="0" :max="100" :precision="1" />
            </el-form-item>

            <el-form-item>
              <el-button type="primary" @click="predictPollution">开始预测</el-button>
              <el-button @click="resetForm">重置</el-button>
            </el-form-item>
          </el-form>

          <template v-if="predictionResult">
            <el-divider />

            <div class="result-container">
              <h3>预测结果</h3>
              <el-result
                :icon="getResultIcon(predictionResult.level)"
                :title="predictionResult.levelText"
                :sub-title="`预测置信度: ${(predictionResult.confidence * 100).toFixed(1)}%`"
              >
                <template #extra>
                  <el-tag :type="getResultType(predictionResult.level)" size="large">
                    污染等级: {{ predictionResult.level }}
                  </el-tag>
                </template>
              </el-result>
            </div>
          </template>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card class="info-card equal-height-card">
          <template #header>
            <div class="card-header">
              <span>污染等级说明</span>
            </div>
          </template>

          <el-alert
            title="使用提示"
            type="info"
            :closable="false"
            show-icon
            style="margin-bottom: 20px;"
          >
            <template #default>
              请输入真实的水质监测数据，系统将使用机器学习模型预测污染等级。
            </template>
          </el-alert>

          <el-timeline>
            <el-timeline-item timestamp="等级 I" color="#67C23A">
              <h4>优秀</h4>
              <p>水质状况极佳，适合各类用途</p>
            </el-timeline-item>
            <el-timeline-item timestamp="等级 II" color="#95D475">
              <h4>良好</h4>
              <p>水质状况良好，适合饮用水源和游泳</p>
            </el-timeline-item>
            <el-timeline-item timestamp="等级 III" color="#E6A23C">
              <h4>轻度污染</h4>
              <p>水质一般，适合工业用水</p>
            </el-timeline-item>
            <el-timeline-item timestamp="等级 IV" color="#F56C6C">
              <h4>中度污染</h4>
              <p>水质较差，仅适合农业灌溉</p>
            </el-timeline-item>
            <el-timeline-item timestamp="等级 V" color="#909399">
              <h4>重度污染</h4>
              <p>水质极差，不适合任何用途</p>
            </el-timeline-item>
          </el-timeline>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useWaterQualityStore } from '@/stores/waterQuality'
import { ElMessage } from 'element-plus'

const store = useWaterQualityStore()
const loading = ref(false)

const pollutionForm = ref({
  waterQualityIndex: 75,
  cod: 15,
  ph: 7.2,
  dissolvedOxygen: 8,
  totalNitrogen: 2,
  totalPhosphorus: 0.1,
  turbidity: 5
})

const predictionResult = ref(null)

const getResultIcon = (prediction) => {
  if (prediction === 'Excellent' || prediction === 'Good') return 'success'
  if (prediction === 'Fair') return 'warning'
  return 'error'
}

const getResultType = (prediction) => {
  const typeMap = {
    'Excellent': 'success',
    'Good': 'primary',
    'Fair': 'warning',
    'Poor': 'danger',
    'Very Poor': 'info'
  }
  return typeMap[prediction] || 'info'
}

const predictPollution = async () => {
  loading.value = true
  try {
    const result = await store.predictPollutionLevel(pollutionForm.value)

    // 映射英文等级到中文
    const levelTexts = {
      'Excellent': '优秀 - 水质极佳',
      'Good': '良好 - 水质状况良好',
      'Fair': '轻度污染 - 水质一般',
      'Poor': '中度污染 - 水质较差',
      'Very Poor': '重度污染 - 水质极差'
    }

    if (result && result.data) {
      const prediction = result.data.prediction
      predictionResult.value = {
        level: prediction,
        levelText: levelTexts[prediction] || prediction,
        confidence: result.data.confidence || 0.85
      }
      ElMessage.success('预测完成')
    } else {
      throw new Error('预测结果格式错误')
    }
  } catch (error) {
    console.error('预测失败:', error)
    ElMessage.error('预测失败，请稍后重试')
  } finally {
    loading.value = false
  }
}

const resetForm = () => {
  pollutionForm.value = {
    waterQualityIndex: 75,
    cod: 15,
    ph: 7.2,
    dissolvedOxygen: 8,
    totalNitrogen: 2,
    totalPhosphorus: 0.1,
    turbidity: 5
  }
  predictionResult.value = null
}
</script>

<style scoped>
.prediction-container {
  padding: 20px;
}

.equal-height-card {
  height: 100%;
  display: flex;
  flex-direction: column;

  :deep(.el-card__body) {
    flex: 1;
    overflow-y: auto;
  }
}

.prediction-card,
.info-card,
.tips-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.result-container {
  margin-top: 20px;
}

.result-container h3 {
  margin-bottom: 20px;
  color: #303133;
}
</style>
