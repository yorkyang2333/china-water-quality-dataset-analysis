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
                  <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
                    <el-input-number v-model="pollutionForm.waterQualityIndex" :min="0" :max="300" style="flex: 1;" />
                    <el-text type="info" style="font-size: 11px; white-space: nowrap;">0-300</el-text>
                  </div>
                </el-form-item>
                <el-form-item label="COD (mg/L)">
                  <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
                    <el-input-number v-model="pollutionForm.cod" :min="0" :max="100" :precision="2" style="flex: 1;" />
                    <el-text type="info" style="font-size: 11px; white-space: nowrap;">0-100</el-text>
                  </div>
                </el-form-item>
                <el-form-item label="pH值">
                  <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
                    <el-input-number v-model="pollutionForm.ph" :min="3" :max="12" :precision="1" style="flex: 1;" />
                    <el-text type="info" style="font-size: 11px; white-space: nowrap;">3-12</el-text>
                  </div>
                </el-form-item>
                <el-form-item label="溶解氧 (mg/L)">
                  <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
                    <el-input-number v-model="pollutionForm.dissolvedOxygen" :min="0" :max="25" :precision="2" style="flex: 1;" />
                    <el-text type="info" style="font-size: 11px; white-space: nowrap;">0-25</el-text>
                  </div>
                </el-form-item>
                <el-form-item label="总氮 (mg/L)">
                  <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
                    <el-input-number v-model="pollutionForm.totalNitrogen" :min="0" :max="10" :precision="2" style="flex: 1;" />
                    <el-text type="info" style="font-size: 11px; white-space: nowrap;">0-10</el-text>
                  </div>
                </el-form-item>
                <el-form-item label="总磷 (mg/L)">
                  <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
                    <el-input-number v-model="pollutionForm.totalPhosphorus" :min="0" :max="1" :precision="3" style="flex: 1;" />
                    <el-text type="info" style="font-size: 11px; white-space: nowrap;">0-1</el-text>
                  </div>
                </el-form-item>
                <el-form-item label="浊度 (NTU)">
                  <div style="display: flex; align-items: center; gap: 8px; width: 100%;">
                    <el-input-number v-model="pollutionForm.turbidity" :min="0" :max="60" :precision="1" style="flex: 1;" />
                    <el-text type="info" style="font-size: 11px; white-space: nowrap;">0-60</el-text>
                  </div>
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

      <!-- LSTM+RF 混合模型时间序列预测 -->
      <el-tab-pane label="时间序列预测 (LSTM+RF混合)" name="timeseries">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-card class="prediction-card equal-height-card" v-loading="timeseriesLoading">
              <template #header>
                <div class="card-header">
                  <span>LSTM + RandomForest混合模型</span>
                </div>
              </template>

              <el-form :model="timeseriesForm" label-width="140px">
                <el-form-item label="水质指标">
                  <el-select v-model="timeseriesForm.indicator" placeholder="选择要预测的指标" style="width: 100%">
                    <el-option label="COD (mg/L)" value="COD_mg_L" />
                    <el-option label="pH值" value="pH" />
                    <el-option label="溶解氧 (mg/L)" value="Dissolved_Oxygen_mg_L" />
                    <el-option label="总氮 (mg/L)" value="Total_Nitrogen_mg_L" />
                    <el-option label="总磷 (mg/L)" value="Total_Phosphorus_mg_L" />
                    <el-option label="浊度 (NTU)" value="Turbidity_NTU" />
                    <el-option label="BOD (mg/L)" value="BOD_mg_L" />
                    <el-option label="电导率 (uS/cm)" value="Conductivity_uS_cm" />
                  </el-select>
                </el-form-item>

                <el-form-item label="省份">
                  <el-select v-model="timeseriesForm.province" placeholder="选择省份（可选）" style="width: 100%" clearable>
                    <el-option label="北京" value="Beijing" />
                    <el-option label="广东" value="Guangdong" />
                    <el-option label="河南" value="Henan" />
                    <el-option label="湖北" value="Hubei" />
                    <el-option label="江苏" value="Jiangsu" />
                    <el-option label="山东" value="Shandong" />
                    <el-option label="上海" value="Shanghai" />
                    <el-option label="四川" value="Sichuan" />
                    <el-option label="云南" value="Yunnan" />
                    <el-option label="浙江" value="Zhejiang" />
                  </el-select>
                </el-form-item>

                <el-form-item label="监测站">
                  <el-input v-model="timeseriesForm.station" placeholder="输入站点名称（可选）" />
                </el-form-item>

                <el-form-item label="Optuna试验次数">
                  <el-input-number v-model="timeseriesForm.nTrials" :min="5" :max="50" :step="5" />
                  <el-text type="info" style="margin-left: 10px; font-size: 12px;">更多试验=更优参数，但耗时更长</el-text>
                </el-form-item>

                <el-form-item>
                  <el-button type="primary" @click="trainTimeseriesModel" :loading="timeseriesLoading">
                    <span v-if="!timeseriesLoading">训练模型</span>
                    <span v-else>训练中...</span>
                  </el-button>
                  <el-button @click="optimizeTimeseriesModel" :loading="timeseriesLoading" style="margin-left: 10px;">
                    <span v-if="!timeseriesLoading">优化超参数</span>
                    <span v-else>优化中...</span>
                  </el-button>
                  <el-button @click="resetTimeseriesForm">重置</el-button>
                </el-form-item>
              </el-form>
            </el-card>
          </el-col>

          <el-col :span="12">
            <el-card class="info-card equal-height-card">
              <template #header>
                <div class="card-header">
                  <span>模型说明</span>
                </div>
              </template>

              <el-alert
                title="使用指南"
                type="info"
                :closable="false"
                show-icon
                style="margin-bottom: 15px;"
              >
                <template #default>
                  <div style="font-size: 12px; line-height: 1.6;">
                    <div>• 选择要预测的水质指标</div>
                    <div>• 可选地指定省份或监测站进行精细预测</div>
                    <div>• 点击"训练模型"使用默认参数快速训练</div>
                    <div>• 点击"优化超参数"使用Optuna自动搜索最优参数（耗时较长）</div>
                  </div>
                </template>
              </el-alert>

              <el-divider />

              <div style="font-size: 12px; line-height: 1.8; color: #606266;">
                <div style="margin-bottom: 15px;">
                  <strong style="color: #303133;">模型特点：</strong>
                  <div>• LSTM：捕捉时间序列的长期依赖关系</div>
                  <div>• RandomForest：学习特征之间的复杂非线性关系</div>
                  <div>• 混合融合：50% LSTM + 50% RF 加权平均</div>
                </div>

                <div style="margin-bottom: 15px;">
                  <strong style="color: #303133;">可优化的超参数：</strong>
                  <div>• LSTM参数：隐层大小、层数、Dropout等</div>
                  <div>• RandomForest：树数量、最大深度等</div>
                  <div>• 融合权重：LSTM和RF的最优融合比例</div>
                </div>

                <div style="padding: 10px; background: #FFF7E6; border-left: 3px solid #FF9800; border-radius: 2px; color: #E6A23C;">
                  <strong>提示：</strong> 首次训练可能需要1-5分钟，优化需要5-30分钟，建议在后台运行。
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <!-- 显示训练结果 -->
        <el-row :gutter="20" v-if="timeseriesResult" style="margin-top: 20px;">
          <el-col :span="24">
            <el-card>
              <template #header>
                <div class="card-header">
                  <span>模型性能指标</span>
                </div>
              </template>

              <el-row :gutter="20">
                <el-col :span="6">
                  <div style="text-align: center; padding: 15px; background: #F5F7FA; border-radius: 8px;">
                    <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">R² 值</div>
                    <div style="font-size: 32px; font-weight: 600; color: #409EFF;">
                      {{ (timeseriesResult.r2 * 100).toFixed(2) }}%
                    </div>
                    <div style="font-size: 11px; color: #909399; margin-top: 5px;">模型拟合度</div>
                  </div>
                </el-col>
                <el-col :span="6">
                  <div style="text-align: center; padding: 15px; background: #F5F7FA; border-radius: 8px;">
                    <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">RMSE</div>
                    <div style="font-size: 32px; font-weight: 600; color: #E6A23C;">
                      {{ timeseriesResult.rmse.toFixed(4) }}
                    </div>
                    <div style="font-size: 11px; color: #909399; margin-top: 5px;">均方根误差</div>
                  </div>
                </el-col>
                <el-col :span="6">
                  <div style="text-align: center; padding: 15px; background: #F5F7FA; border-radius: 8px;">
                    <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">MAE</div>
                    <div style="font-size: 32px; font-weight: 600; color: #67C23A;">
                      {{ timeseriesResult.mae.toFixed(4) }}
                    </div>
                    <div style="font-size: 11px; color: #909399; margin-top: 5px;">平均绝对误差</div>
                  </div>
                </el-col>
                <el-col :span="6">
                  <div style="text-align: center; padding: 15px; background: #F5F7FA; border-radius: 8px;">
                    <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">MSE</div>
                    <div style="font-size: 32px; font-weight: 600; color: #F56C6C;">
                      {{ timeseriesResult.mse.toFixed(4) }}
                    </div>
                    <div style="font-size: 11px; color: #909399; margin-top: 5px;">均方误差</div>
                  </div>
                </el-col>
              </el-row>

              <el-divider v-if="timeseriesResult.best_params" style="margin: 20px 0;" />

              <div v-if="timeseriesResult.best_params" style="font-size: 12px;">
                <strong style="color: #303133; display: block; margin-bottom: 10px;">优化的超参数：</strong>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                  <div v-for="(value, key) in timeseriesResult.best_params" :key="key"
                       style="padding: 8px; background: #F5F7FA; border-radius: 4px; font-size: 11px;">
                    <strong>{{ key }}:</strong> {{ typeof value === 'number' ? value.toFixed(2) : value }}
                  </div>
                </div>
              </div>
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
const timeseriesLoading = ref(false)
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

// 时间序列模型预测
const timeseriesForm = ref({
  indicator: 'COD_mg_L',
  province: '',
  station: '',
  nTrials: 10
})

const timeseriesResult = ref(null)

// 季节性分析
const seasonalityFilters = ref({
  indicator: 'Water_Temperature_C',
  groupBy: 'Province'
})

const seasonalityResult = ref(null)

// 指标标签映射
// 水质指标标签
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

// 污染等级预测输入值的合理范围（宽松范围，约为实际数据范围的1.5-2倍）
const pollutionInputRanges = {
  'waterQualityIndex': { min: 0, max: 300, unit: '', label: '水质指数', expectedMin: 0, expectedMax: 100 },
  'cod': { min: 0, max: 100, unit: 'mg/L', label: 'COD', expectedMin: 0, expectedMax: 60 },
  'ph': { min: 3, max: 12, unit: '', label: 'pH值', expectedMin: 5, expectedMax: 9 },
  'dissolvedOxygen': { min: 0, max: 25, unit: 'mg/L', label: '溶解氧', expectedMin: 2, expectedMax: 15 },
  'totalNitrogen': { min: 0, max: 10, unit: 'mg/L', label: '总氮', expectedMin: 1, expectedMax: 6 },
  'totalPhosphorus': { min: 0, max: 1, unit: 'mg/L', label: '总磷', expectedMin: 0, expectedMax: 0.5 },
  'turbidity': { min: 0, max: 60, unit: 'NTU', label: '浊度', expectedMin: 0, expectedMax: 30 }
}

// 检查污染等级预测的输入值
const checkPollutionInputValues = () => {
  const warnings = []

  // 检查每个输入字段
  for (const [key, value] of Object.entries(pollutionForm.value)) {
    if (key in pollutionInputRanges && value !== null && value !== '') {
      const range = pollutionInputRanges[key]
      const numValue = parseFloat(value)

      // 检查是否超出合理范围
      if (numValue < range.min || numValue > range.max) {
        warnings.push({
          field: key,
          label: range.label,
          value: numValue,
          minExpected: range.expectedMin,
          maxExpected: range.expectedMax,
          min: range.min,
          max: range.max,
          unit: range.unit
        })
      } else if (numValue < range.expectedMin || numValue > range.expectedMax) {
        // 在合理范围内但偏离预期范围
        warnings.push({
          field: key,
          label: range.label,
          value: numValue,
          minExpected: range.expectedMin,
          maxExpected: range.expectedMax,
          min: range.min,
          max: range.max,
          unit: range.unit,
          isWarning: true  // 标记为警告而非错误
        })
      }
    }
  }

  return warnings
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

const showRangeWarning = (warnings) => {
  // 分离超出范围的错误和轻微偏离的警告
  const errors = warnings.filter(w => !w.isWarning)
  const minorWarnings = warnings.filter(w => w.isWarning)

  let warningContent = '<div style="text-align: left; font-size: 12px;">'

  if (errors.length > 0) {
    warningContent += '<div style="margin-bottom: 15px;">'
    warningContent += '<strong style="color: #F56C6C; display: block; margin-bottom: 8px;">❌ 超出合理范围的值（必须修改）：</strong>'
    errors.forEach(w => {
      warningContent += `
        <div style="margin-bottom: 10px; padding: 10px; background: #FEF0F0; border-left: 3px solid #F56C6C; border-radius: 2px;">
          <strong>${w.label}:</strong> ${w.value} ${w.unit}<br/>
          <span style="color: #606266;">合理范围: ${w.min} ~ ${w.max} ${w.unit}</span><br/>
          <span style="color: #909399; font-size: 11px;">实际数据范围: ${w.minExpected} ~ ${w.maxExpected} ${w.unit}</span>
        </div>
      `
    })
    warningContent += '</div>'
  }

  if (minorWarnings.length > 0) {
    warningContent += '<div style="margin-bottom: 15px;">'
    warningContent += '<strong style="color: #E6A23C; display: block; margin-bottom: 8px;">⚠️ 偏离预期范围的值（可接受，但请确认）：</strong>'
    minorWarnings.forEach(w => {
      warningContent += `
        <div style="margin-bottom: 10px; padding: 10px; background: #FFF7E6; border-left: 3px solid #E6A23C; border-radius: 2px;">
          <strong>${w.label}:</strong> ${w.value} ${w.unit}<br/>
          <span style="color: #606266;">预期范围: ${w.minExpected} ~ ${w.maxExpected} ${w.unit}</span><br/>
          <span style="color: #909399; font-size: 11px;">可接受范围: ${w.min} ~ ${w.max} ${w.unit}</span>
        </div>
      `
    })
    warningContent += '</div>'
  }

  warningContent += '<div style="padding: 10px; background: #F0F9FF; border-left: 3px solid #409EFF; border-radius: 2px; color: #606266; font-size: 11px;">'
  warningContent += '<strong>提示：</strong> 请检查输入值是否正确，如有超出范围的值请修改后重新预测。'
  warningContent += '</div>'
  warningContent += '</div>'

  return {
    html: warningContent,
    hasErrors: errors.length > 0,
    errors: errors,
    warnings: minorWarnings
  }
}

const predictPollution = async () => {
  // 检查输入值范围
  const rangeWarnings = checkPollutionInputValues()

  // 如果有超出范围的值，显示警告
  if (rangeWarnings.length > 0) {
    const warningResult = showRangeWarning(rangeWarnings)

    // 如果有错误（超出范围），显示对话框并中断
    if (warningResult.hasErrors) {
      ElMessageBox.alert(warningResult.html, '⚠️ 输入值超出合理范围', {
        dangerouslyUseHTMLString: true,
        confirmButtonText: '关闭',
        type: 'error'
      }).catch(() => {})
      return
    }

    // 如果只有警告（轻微偏离），询问是否继续
    ElMessageBox.confirm(warningResult.html, '⚠️ 输入值偏离预期范围', {
      dangerouslyUseHTMLString: true,
      confirmButtonText: '继续预测',
      cancelButtonText: '修改数据',
      type: 'warning'
    }).then(() => {
      // 用户选择继续预测
      doPrediction()
    }).catch(() => {
      // 用户选择修改数据
      return
    })
  } else {
    // 输入值正常，直接预测
    doPrediction()
  }
}

const doPrediction = async () => {
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
            <div style="margin-bottom: 8px;"><strong>污染等级说明</strong></div>
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

// 时间序列模型 - 训练
const trainTimeseriesModel = async () => {
  if (!timeseriesForm.value.indicator) {
    ElMessage.warning('请选择水质指标')
    return
  }

  timeseriesLoading.value = true
  try {
    const payload = {
      indicator: timeseriesForm.value.indicator
    }
    if (timeseriesForm.value.province) {
      payload.province = timeseriesForm.value.province
    }
    if (timeseriesForm.value.station) {
      payload.station = timeseriesForm.value.station
    }

    const response = await fetch('/api/model/hybrid/train', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${store.token}`
      },
      body: JSON.stringify(payload)
    })

    const data = await response.json()

    if (data.success) {
      timeseriesResult.value = data.data
      ElMessage.success('✅ 模型训练成功！')

      // 显示结果对话框
      const resultHtml = `
        <div style="text-align: center; padding: 20px;">
          <div style="margin-bottom: 30px;">
            <div style="font-size: 20px; color: #303133; margin-bottom: 10px; font-weight: 600;">
              LSTM + RandomForest 混合模型训练完成
            </div>
          </div>

          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">R² 值</div>
              <div style="font-size: 28px; font-weight: 600; color: #409EFF;">
                ${(data.data.r2 * 100).toFixed(2)}%
              </div>
            </div>
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">RMSE</div>
              <div style="font-size: 28px; font-weight: 600; color: #E6A23C;">
                ${data.data.rmse.toFixed(4)}
              </div>
            </div>
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">MAE</div>
              <div style="font-size: 28px; font-weight: 600; color: #67C23A;">
                ${data.data.mae.toFixed(4)}
              </div>
            </div>
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">MSE</div>
              <div style="font-size: 28px; font-weight: 600; color: #F56C6C;">
                ${data.data.mse.toFixed(4)}
              </div>
            </div>
          </div>

          <div style="padding: 15px; background: #FFF7E6; border-left: 4px solid #FF9800; border-radius: 4px; text-align: left; font-size: 12px; color: #333;">
            <strong>模型说明</strong>
            <div style="margin-top: 8px; line-height: 1.6;">
              <div>• R² > 0.8: 模型拟合效果优秀</div>
              <div>• RMSE: 均方根误差越小越好</div>
              <div>• MAE: 平均绝对误差越小越好</div>
            </div>
          </div>
        </div>
      `

      ElMessageBox.alert(resultHtml, '✓ 训练完成', {
        dangerouslyUseHTMLString: true,
        confirmButtonText: '确定',
        type: 'success',
        width: '500px'
      }).catch(() => {})
    } else {
      throw new Error(data.error || '训练失败')
    }
  } catch (error) {
    console.error('训练失败:', error)
    ElMessage.error('❌ ' + (error.message || '训练失败，请检查后重试'))
  } finally {
    timeseriesLoading.value = false
  }
}

// 时间序列模型 - 优化超参数
const optimizeTimeseriesModel = async () => {
  if (!timeseriesForm.value.indicator) {
    ElMessage.warning('请选择水质指标')
    return
  }

  if (timeseriesForm.value.nTrials < 5) {
    ElMessage.warning('Optuna试验次数至少需要5次')
    return
  }

  timeseriesLoading.value = true
  ElMessage.loading({
    message: '正在优化超参数，请耐心等待...',
    duration: 0
  })

  try {
    const payload = {
      indicator: timeseriesForm.value.indicator,
      n_trials: timeseriesForm.value.nTrials
    }
    if (timeseriesForm.value.province) {
      payload.province = timeseriesForm.value.province
    }
    if (timeseriesForm.value.station) {
      payload.station = timeseriesForm.value.station
    }

    const response = await fetch('/api/model/hybrid/optimize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${store.token}`
      },
      body: JSON.stringify(payload),
      timeout: 600000 // 10分钟超时
    })

    const data = await response.json()

    if (data.success) {
      timeseriesResult.value = {
        ...data.data.metrics,
        best_params: data.data.best_params,
        best_r2: data.data.best_r2
      }
      ElMessage.closeAll()
      ElMessage.success('✅ 超参数优化完成！')

      // 显示优化结果
      const resultHtml = `
        <div style="text-align: center; padding: 20px;">
          <div style="margin-bottom: 30px;">
            <div style="font-size: 20px; color: #303133; margin-bottom: 10px; font-weight: 600;">
              Optuna 超参数优化完成
            </div>
            <div style="font-size: 14px; color: #606266;">
              优化试验数: ${timeseriesForm.value.nTrials} | 最佳 R²: ${(data.data.best_r2 * 100).toFixed(2)}%
            </div>
          </div>

          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">最佳 R²</div>
              <div style="font-size: 28px; font-weight: 600; color: #67C23A;">
                ${(data.data.best_r2 * 100).toFixed(2)}%
              </div>
            </div>
            <div style="padding: 15px; background: #F5F7FA; border-radius: 8px;">
              <div style="font-size: 12px; color: #909399; margin-bottom: 8px;">RMSE</div>
              <div style="font-size: 28px; font-weight: 600; color: #E6A23C;">
                ${data.data.metrics.rmse.toFixed(4)}
              </div>
            </div>
          </div>

          <div style="padding: 15px; background: #FFF7E6; border-left: 4px solid #FF9800; border-radius: 4px; text-align: left; font-size: 12px; color: #333;">
            <strong>优化的超参数：</strong>
            <div style="margin-top: 10px; line-height: 1.8; font-size: 11px;">
              ${Object.entries(data.data.best_params).map(([key, val]) =>
                `<div>• ${key}: ${typeof val === 'number' ? val.toFixed(2) : val}</div>`
              ).join('')}
            </div>
          </div>
        </div>
      `

      ElMessageBox.alert(resultHtml, '✓ 优化完成', {
        dangerouslyUseHTMLString: true,
        confirmButtonText: '确定',
        type: 'success',
        width: '500px'
      }).catch(() => {})
    } else {
      throw new Error(data.error || '优化失败')
    }
  } catch (error) {
    console.error('优化失败:', error)
    ElMessage.closeAll()
    ElMessage.error('❌ ' + (error.message || '优化失败，请检查后重试'))
  } finally {
    timeseriesLoading.value = false
  }
}

const resetTimeseriesForm = () => {
  timeseriesForm.value = {
    indicator: 'COD_mg_L',
    province: '',
    station: '',
    nTrials: 10
  }
  timeseriesResult.value = null
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
            <div style="margin-bottom: 8px;"><strong>分析说明</strong></div>
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

.card-header span {
  font-weight: 600;
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
