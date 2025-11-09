<template>
  <div class="prediction-container">
    <!-- 选项卡头 -->
    <el-tabs v-model="activeTab" class="prediction-tabs">
      <!-- 污染等级预测 -->
      <el-tab-pane label="污染等级预测" name="pollution">
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
                  <el-button @click="resetPollutionForm">重置</el-button>
                </el-form-item>
              </el-form>
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
      </el-tab-pane>

      <!-- 季节性分析 -->
      <el-tab-pane label="季节性趋势分析" name="seasonality">
        <el-card v-loading="seasonalityLoading">
          <template #header>
            <div class="card-header">
              <span>水质指标季节性特征分析</span>
            </div>
          </template>

          <el-form :inline="true" :model="seasonalityFilters" style="margin-bottom: 20px;">
            <el-form-item label="水质指标">
              <el-select v-model="seasonalityFilters.indicator" placeholder="选择指标" style="width: 150px" @change="analyzeSeasonality">
                <el-option label="水温" value="Water_Temperature_C" />
                <el-option label="pH值" value="pH" />
                <el-option label="溶解氧" value="Dissolved_Oxygen_mg_L" />
                <el-option label="电导率" value="Conductivity_uS_cm" />
                <el-option label="浊度" value="Turbidity_NTU" />
                <el-option label="硝酸盐" value="Nitrate_mg_L" />
                <el-option label="COD" value="COD_mg_L" />
                <el-option label="BOD" value="BOD_mg_L" />
                <el-option label="总氮" value="Total_Nitrogen_mg_L" />
                <el-option label="总磷" value="Total_Phosphorus_mg_L" />
              </el-select>
            </el-form-item>
            <el-form-item label="分组方式">
              <el-select v-model="seasonalityFilters.groupBy" placeholder="选择分组方式" style="width: 150px" @change="analyzeSeasonality">
                <el-option label="按省份" value="Province" />
                <el-option label="按监测站" value="Monitoring_Station" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="analyzeSeasonality">分析</el-button>
            </el-form-item>
          </el-form>

          <el-divider />

          <!-- 季节性分析结果展示 -->
          <div v-if="seasonalityResult" class="seasonality-result">
            <el-alert
              v-if="seasonalityResult.summary"
              :title="`${getIndicatorLabel(seasonalityFilters.indicator)} 季节性分析结果`"
              type="info"
              :closable="false"
              show-icon
              style="margin-bottom: 20px;"
            >
              <template #default>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 10px;">
                  <div>
                    <strong>平均季节性强度:</strong>
                    <el-progress
                      :percentage="Math.round(seasonalityResult.summary.avg_seasonality_strength * 100)"
                      :color="seasonalityResult.summary.avg_seasonality_strength > 0.5 ? '#67C23A' : '#E6A23C'"
                    />
                  </div>
                  <div>
                    <strong>最大季节性强度:</strong>
                    <el-progress
                      :percentage="Math.round(seasonalityResult.summary.max_seasonality_strength * 100)"
                      :color="'#409EFF'"
                    />
                  </div>
                  <div>
                    <strong>最小季节性强度:</strong>
                    <el-progress
                      :percentage="Math.round(seasonalityResult.summary.min_seasonality_strength * 100)"
                      :color="'#909399'"
                    />
                  </div>
                  <div>
                    <strong>分析的地区/站点数:</strong>
                    <p style="font-size: 20px; color: #409EFF; margin: 5px 0;">{{ seasonalityResult.summary.groups_count }}</p>
                  </div>
                </div>
              </template>
            </el-alert>

            <el-empty v-if="!seasonalityResult.summary" description="分析失败，请重试" style="margin-top: 30px;" />
          </div>

          <el-empty v-else description="请选择指标进行分析" style="margin-top: 30px;" />
        </el-card>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useWaterQualityStore } from '@/stores/waterQuality'
import { ElMessage, ElMessageBox } from 'element-plus'

const store = useWaterQualityStore()
const loading = ref(false)
const seasonalityLoading = ref(false)
const activeTab = ref('pollution')

// 污染等级预测表单
const pollutionForm = ref({
  waterQualityIndex: 75,
  cod: 15,
  ph: 7.2,
  dissolvedOxygen: 8,
  totalNitrogen: 2,
  totalPhosphorus: 0.1,
  turbidity: 5
})

// 季节性分析
const seasonalityFilters = ref({
  indicator: 'Water_Temperature_C',
  groupBy: 'Province'
})

const seasonalityResult = ref(null)

// 指标标签映射
const indicatorLabels = {
  'Water_Temperature_C': '水温',
  'pH': 'pH值',
  'Dissolved_Oxygen_mg_L': '溶解氧',
  'Conductivity_uS_cm': '电导率',
  'Turbidity_NTU': '浊度',
  'Nitrate_mg_L': '硝酸盐',
  'COD_mg_L': 'COD',
  'BOD_mg_L': 'BOD',
  'Total_Nitrogen_mg_L': '总氮',
  'Total_Phosphorus_mg_L': '总磷'
}

const getIndicatorLabel = (indicator) => {
  return indicatorLabels[indicator] || indicator
}

const showWarningsDialog = (warnings) => {
  let warningContent = '<div style="text-align: left;">'
  warnings.forEach((warning, index) => {
    warningContent += `<div style="margin-bottom: 15px; padding: 10px; background: #fff7e6; border-left: 4px solid #ff9800; border-radius: 2px;">`
    warningContent += `<strong style="color: #ff9800;">警告 ${index + 1}</strong>`
    warningContent += `<p style="margin: 5px 0 0 0; white-space: pre-wrap; color: #333;">${warning.message}</p>`
    warningContent += '</div>'
  })
  warningContent += '</div>'

  ElMessageBox.alert(warningContent, '⚠️ 数据范围警告', {
    dangerouslyUseHTMLString: true,
    confirmButtonText: '继续预测',
    type: 'warning'
  }).catch(() => {
    return
  })
}

const predictPollution = async () => {
  loading.value = true
  try {
    const result = await store.predictPollutionLevel(pollutionForm.value)

    if (result.has_warnings && result.warnings && result.warnings.length > 0) {
      showWarningsDialog(result.warnings)
    }

    const levelTexts = {
      'Excellent': '优秀 - 水质极佳',
      'Good': '良好 - 水质状况良好',
      'Fair': '轻度污染 - 水质一般',
      'Poor': '中度污染 - 水质较差',
      'Very Poor': '重度污染 - 水质极差'
    }

    const levelDescriptions = {
      'Excellent': '水质状况极佳，适合各类用途',
      'Good': '水质状况良好，适合饮用水源和游泳',
      'Fair': '水质一般，适合工业用水',
      'Poor': '水质较差，仅适合农业灌溉',
      'Very Poor': '水质极差，不适合任何用途'
    }

    const levelColors = {
      'Excellent': '#67C23A',
      'Good': '#95D475',
      'Fair': '#E6A23C',
      'Poor': '#F56C6C',
      'Very Poor': '#909399'
    }

    if (result && result.data) {
      const prediction = result.data.prediction

      // 构建漂亮的结果展示 HTML
      const resultHtml = `
        <div style="text-align: center; padding: 20px;">
          <div style="margin-bottom: 30px;">
            <div style="font-size: 24px; color: #303133; margin-bottom: 15px; font-weight: 600;">
              污染等级预测结果
            </div>
          </div>

          <div style="display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 30px;">
            <div style="padding: 30px; background: linear-gradient(135deg, ${levelColors[prediction]}15 0%, ${levelColors[prediction]}05 100%); border-left: 6px solid ${levelColors[prediction]}; border-radius: 8px;">
              <div style="font-size: 14px; color: #909399; margin-bottom: 10px;">污染等级</div>
              <div style="font-size: 36px; font-weight: 700; color: ${levelColors[prediction]}; margin-bottom: 10px;">
                ${levelTexts[prediction]}
              </div>
              <div style="font-size: 14px; color: #606266;">
                ${levelDescriptions[prediction]}
              </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
              <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
                <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">预测置信度</div>
                <div style="font-size: 24px; font-weight: 600; color: #409EFF;">
                  ${(result.data.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
                <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">样本水质指数</div>
                <div style="font-size: 24px; font-weight: 600; color: #409EFF;">
                  ${pollutionForm.value.waterQualityIndex}
                </div>
              </div>
            </div>
          </div>

          <div style="padding: 15px; background: #FFF7E6; border-left: 4px solid #FF9800; border-radius: 4px; text-align: left; font-size: 13px; color: #333;">
            <div style="margin-bottom: 8px;"><strong>📊 污染等级说明</strong></div>
            <div style="line-height: 1.8;">
              <div>• 优秀（Excellent）: 最佳水质状况</div>
              <div>• 良好（Good）: 水质良好，可安全利用</div>
              <div>• 轻度污染（Fair）: 水质受轻度污染</div>
              <div>• 中度污染（Poor）: 水质受中度污染</div>
              <div>• 重度污染（Very Poor）: 水质受严重污染</div>
            </div>
          </div>
        </div>
      `

      ElMessageBox.alert(resultHtml, '✓ 污染等级预测完成', {
        dangerouslyUseHTMLString: true,
        confirmButtonText: '确定',
        type: 'success',
        width: '500px'
      }).catch(() => {})
    } else {
      throw new Error('预测结果格式错误')
    }
  } catch (error) {
    console.error('预测失败:', error)

    ElMessageBox.alert(error.message || '预测失败，请检查输入数据后重试', '❌ 预测失败', {
      confirmButtonText: '确定',
      type: 'error'
    }).catch(() => {})
  } finally {
    loading.value = false
  }
}

const resetPollutionForm = () => {
  pollutionForm.value = {
    waterQualityIndex: 75,
    cod: 15,
    ph: 7.2,
    dissolvedOxygen: 8,
    totalNitrogen: 2,
    totalPhosphorus: 0.1,
    turbidity: 5
  }
}

const analyzeSeasonality = async () => {
  if (!seasonalityFilters.value.indicator) {
    ElMessage.warning('请选择要分析的指标')
    return
  }

  seasonalityLoading.value = true
  try {
    const response = await fetch(
      `/api/analysis/seasonality/quick?indicator=${seasonalityFilters.value.indicator}&group_by=${seasonalityFilters.value.groupBy}`
    )
    const data = await response.json()

    if (data.success) {
      seasonalityResult.value = data.data

      const strength = data.data.summary?.avg_seasonality_strength || 0
      const maxStrength = data.data.summary?.max_seasonality_strength || 0
      const minStrength = data.data.summary?.min_seasonality_strength || 0
      const groupsCount = data.data.summary?.groups_count || 0
      const hasStrong = data.data.has_strong_seasonality
      const indicatorLabel = getIndicatorLabel(seasonalityFilters.value.indicator)

      // 构建更漂亮的结果展示 HTML
      const resultHtml = `
        <div style="text-align: center; padding: 20px;">
          <div style="margin-bottom: 30px;">
            <div style="font-size: 24px; color: #303133; margin-bottom: 10px; font-weight: 600;">
              ${indicatorLabel} 季节性分析结果
            </div>
            <div style="font-size: 14px; color: #909399;">
              分析维度: ${seasonalityFilters.value.groupBy === 'Province' ? '按省份' : '按监测站'}
            </div>
          </div>

          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">平均季节性强度</div>
              <div style="font-size: 28px; font-weight: 600; color: ${strength > 0.5 ? '#67C23A' : '#E6A23C'};">
                ${(strength * 100).toFixed(1)}%
              </div>
            </div>
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">季节性特征</div>
              <div style="font-size: 20px; font-weight: 600; color: ${hasStrong ? '#67C23A' : '#E6A23C'};">
                ${hasStrong ? '明显' : '不明显'}
              </div>
            </div>
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">最大季节性强度</div>
              <div style="font-size: 22px; font-weight: 600; color: #409EFF;">
                ${(maxStrength * 100).toFixed(1)}%
              </div>
            </div>
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">分析的地区/站点数</div>
              <div style="font-size: 22px; font-weight: 600; color: #409EFF;">
                ${groupsCount}
              </div>
            </div>
          </div>

          <div style="padding: 15px; background: #FFF7E6; border-left: 4px solid #FF9800; border-radius: 4px; text-align: left; font-size: 13px; color: #333;">
            <div style="margin-bottom: 8px;"><strong>📊 分析说明</strong></div>
            <div>• 季节性强度 &gt; 50%: 水质指标具有明显的季节性特征</div>
            <div>• 季节性强度 ≤ 50%: 水质指标季节性特征不明显</div>
          </div>
        </div>
      `

      ElMessageBox.alert(resultHtml, '✓ 季节性分析完成', {
        dangerouslyUseHTMLString: true,
        confirmButtonText: '确定',
        type: 'success',
        width: '500px'
      }).catch(() => {})
    } else {
      throw new Error(data.error || '分析失败')
    }
  } catch (error) {
    console.error('季节性分析失败:', error)
    ElMessage.error('季节性分析失败，请稍后重试')
  } finally {
    seasonalityLoading.value = false
  }
}
</script>

<style scoped>
.prediction-container {
  padding: 20px;
}

.prediction-tabs {
  margin-bottom: 20px;
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
.info-card {
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

.seasonality-result {
  margin-top: 20px;
}
</style>
