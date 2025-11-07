<template>
  <div class="dashboard">
    <!-- 页面标题和筛选器 -->
    <div class="dashboard-header">
      <div class="header-content">
        <h1 class="page-title">数据概览</h1>
        <p class="page-subtitle">全国水质监测数据实时监控</p>
      </div>
      
      <div class="header-filters">
        <el-select
          v-model="selectedProvince"
          placeholder="选择省份"
          clearable
          @change="handleProvinceChange"
          style="width: 200px"
        >
          <el-option
            v-for="province in provinces"
            :key="province"
            :label="province"
            :value="province"
          />
        </el-select>
        
        <el-date-picker
          v-model="dateRange"
          type="daterange"
          range-separator="至"
          start-placeholder="开始日期"
          end-placeholder="结束日期"
          @change="handleDateChange"
          style="width: 280px"
        />
      </div>
    </div>

    <!-- 统计卡片 -->
    <div class="stats-grid">
      <div class="stat-card glass-card fade-in">
        <div class="stat-icon bg-info">
          <el-icon><Location /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ stationCount }}</div>
          <div class="stat-label">监测站点</div>
          <div class="stat-change positive">
            <el-icon><TrendCharts /></el-icon>
            <span>+12% 本月新增</span>
          </div>
        </div>
      </div>

      <div class="stat-card glass-card fade-in">
        <div class="stat-icon bg-success">
          <el-icon><CircleCheck /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ statistics.excellent || 0 }}</div>
          <div class="stat-label">优秀水质</div>
          <div class="stat-change positive">
            <el-icon><TrendCharts /></el-icon>
            <span>+5% 较上月</span>
          </div>
        </div>
      </div>

      <div class="stat-card glass-card fade-in">
        <div class="stat-icon bg-warning">
          <el-icon><Warning /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ warningAlerts.length }}</div>
          <div class="stat-label">预警站点</div>
          <div class="stat-change negative">
            <el-icon><TrendCharts /></el-icon>
            <span>-3% 较上月</span>
          </div>
        </div>
      </div>

      <div class="stat-card glass-card fade-in">
        <div class="stat-icon bg-danger">
          <el-icon><CircleClose /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ criticalAlerts.length }}</div>
          <div class="stat-label">严重污染</div>
          <div class="stat-change negative">
            <el-icon><TrendCharts /></el-icon>
            <span>+2% 需关注</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 图表和地图区域 -->
    <div class="dashboard-content">
      <!-- 左侧图表 -->
      <div class="charts-section">
        <!-- 污染等级分布 -->
        <div class="chart-container glass-card fade-in">
          <div class="chart-header">
            <h3 class="chart-title">污染等级分布</h3>
            <el-select
              v-model="chartProvince"
              placeholder="选择省份"
              clearable
              @change="updatePollutionChart"
              style="width: 150px"
            >
              <el-option
                v-for="province in provinces"
                :key="province"
                :label="province"
                :value="province"
              />
            </el-select>
          </div>
          <div class="chart-content">
            <canvas ref="pollutionChart"></canvas>
          </div>
        </div>

        <!-- 趋势图表 -->
        <div class="chart-container glass-card fade-in">
          <div class="chart-header">
            <h3 class="chart-title">水质趋势</h3>
            <el-select
              v-model="trendIndicator"
              @change="updateTrendChart"
              style="width: 150px"
            >
              <el-option label="水质指数" value="Water_Quality_Index" />
              <el-option label="COD" value="COD_mg_L" />
              <el-option label="总氮" value="Total_Nitrogen_mg_L" />
              <el-option label="总磷" value="Total_Phosphorus_mg_L" />
            </el-select>
          </div>
          <div class="chart-content">
            <canvas ref="trendChart"></canvas>
          </div>
        </div>
      </div>

      <!-- 右侧地图和告警 -->
      <div class="side-section">
        <!-- 实时告警 -->
        <div class="alerts-container glass-card fade-in">
          <div class="alerts-header">
            <h3 class="alerts-title">
              <el-icon><Warning /></el-icon>
              实时告警
            </h3>
            <el-badge :value="totalAlerts" type="danger" />
          </div>
          <div class="alerts-list custom-scrollbar">
            <div
              v-for="alert in recentAlerts"
              :key="alert.id"
              class="alert-item"
              :class="alert.level"
            >
              <div class="alert-icon">
                <el-icon>
                  <CircleClose v-if="alert.level === 'critical'" />
                  <Warning v-else />
                </el-icon>
              </div>
              <div class="alert-content">
                <div class="alert-title">{{ alert.station }}</div>
                <div class="alert-message">{{ alert.message }}</div>
                <div class="alert-time">{{ alert.time }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 快速操作 -->
        <div class="quick-actions glass-card fade-in">
          <h3 class="actions-title">快速操作</h3>
          <div class="action-buttons">
            <el-button
              type="primary"
              @click="$router.push('/monitoring')"
              class="action-btn"
            >
              <el-icon><Monitor /></el-icon>
              实时监测
            </el-button>
            
            <el-button
              type="success"
              @click="$router.push('/prediction')"
              class="action-btn"
            >
              <el-icon><TrendCharts /></el-icon>
              智能预测
            </el-button>
            
            <el-button
              type="warning"
              @click="$router.push('/analysis')"
              class="action-btn"
            >
              <el-icon><DataAnalysis /></el-icon>
              数据分析
            </el-button>
            
            <el-button
              type="info"
              @click="$router.push('/spatial')"
              class="action-btn"
            >
              <el-icon><Location /></el-icon>
              空间分析
            </el-button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useWaterQualityStore } from '@/stores/waterQuality'
import { useLoadingStore } from '@/stores/loading'
import { Chart, registerables } from 'chart.js'
import {
  Location,
  CircleCheck,
  Warning,
  CircleClose,
  TrendCharts,
  Monitor,
  DataAnalysis
} from '@element-plus/icons-vue'

// 注册Chart.js组件
Chart.register(...registerables)

const waterQualityStore = useWaterQualityStore()
const loadingStore = useLoadingStore()

// 响应式数据
const selectedProvince = ref('')
const dateRange = ref([])
const chartProvince = ref('')
const trendIndicator = ref('Water_Quality_Index')
const pollutionChart = ref(null)
const trendChart = ref(null)

// 图表实例
let pollutionChartInstance = null
let trendChartInstance = null

// 计算属性
const stationCount = computed(() => waterQualityStore.stationCount)
const statistics = computed(() => waterQualityStore.statistics)
const criticalAlerts = computed(() => waterQualityStore.criticalAlerts)
const warningAlerts = computed(() => waterQualityStore.warningAlerts)
const totalAlerts = computed(() => waterQualityStore.totalAlerts)

const provinces = computed(() => [
  '北京', '上海', '广州', '深圳', '成都', 
  '杭州', '武汉', '西安', '南京', '重庆'
])

const recentAlerts = computed(() => {
  const allAlerts = [...criticalAlerts.value, ...warningAlerts.value]
  return allAlerts.slice(0, 10).map((alert, index) => ({
    id: index,
    station: alert.station_name || `站点${index + 1}`,
    message: alert.message || '水质指标异常',
    level: alert.level || 'warning',
    time: alert.date || new Date().toLocaleString()
  }))
})

// 方法
const handleProvinceChange = async (province) => {
  loadingStore.setPageLoading(true)
  try {
    await Promise.all([
      waterQualityStore.fetchStatistics(province),
      waterQualityStore.fetchRealtimeData(province),
      waterQualityStore.fetchAlerts(province)
    ])
    updatePollutionChart()
    updateTrendChart()
  } finally {
    loadingStore.setPageLoading(false)
  }
}

const handleDateChange = async (dates) => {
  if (dates && dates.length === 2) {
    loadingStore.setPageLoading(true)
    try {
      // 根据日期范围更新数据
      await updateCharts()
    } finally {
      loadingStore.setPageLoading(false)
    }
  }
}

const initPollutionChart = () => {
  if (!pollutionChart.value) return

  const ctx = pollutionChart.value.getContext('2d')
  pollutionChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['优秀', '良好', '一般', '较差', '很差'],
      datasets: [{
        data: [30, 25, 20, 15, 10],
        backgroundColor: [
          '#10b981',
          '#3b82f6',
          '#f59e0b',
          '#f97316',
          '#ef4444'
        ],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom'
        }
      }
    }
  })
}

const initTrendChart = () => {
  if (!trendChart.value) return

  const ctx = trendChart.value.getContext('2d')
  trendChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
      datasets: [{
        label: '水质指数',
        data: [65, 62, 68, 70, 66, 72],
        borderColor: '#667eea',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  })
}

const updatePollutionChart = async () => {
  try {
    const data = await waterQualityStore.fetchPollutionDistribution({
      province: chartProvince.value
    })
    
    if (pollutionChartInstance && data.data) {
      pollutionChartInstance.data.datasets[0].data = data.data
      pollutionChartInstance.update()
    }
  } catch (error) {
    console.error('更新污染分布图表失败:', error)
  }
}

const updateTrendChart = async () => {
  try {
    const data = await waterQualityStore.getTrends({
      indicator: trendIndicator.value,
      province: selectedProvince.value
    })
    
    if (trendChartInstance && data.data) {
      trendChartInstance.data.datasets[0].data = data.data
      trendChartInstance.update()
    }
  } catch (error) {
    console.error('更新趋势图表失败:', error)
  }
}

const updateCharts = async () => {
  await Promise.all([
    updatePollutionChart(),
    updateTrendChart()
  ])
}

const loadInitialData = async () => {
  loadingStore.setGlobalLoading(true)
  try {
    await Promise.all([
      waterQualityStore.fetchStations(),
      waterQualityStore.fetchStatistics(),
      waterQualityStore.fetchRealtimeData(),
      waterQualityStore.fetchAlerts()
    ])
  } finally {
    loadingStore.setGlobalLoading(false)
  }
}

// 生命周期
onMounted(async () => {
  await loadInitialData()
  
  // 初始化图表
  setTimeout(() => {
    initPollutionChart()
    initTrendChart()
  }, 100)
})
</script>

<style lang="scss" scoped>
.dashboard {
  padding: $spacing-6;
  max-width: 1400px;
  margin: 0 auto;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: $spacing-8;
  gap: $spacing-6;
}

.header-content {
  flex: 1;
}

.page-title {
  font-size: $font-size-3xl;
  font-weight: 700;
  color: $white;
  margin-bottom: $spacing-2;
}

.page-subtitle {
  font-size: $font-size-lg;
  color: rgba(255, 255, 255, 0.8);
  margin: 0;
}

.header-filters {
  display: flex;
  gap: $spacing-4;
  flex-shrink: 0;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: $spacing-6;
  margin-bottom: $spacing-8;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: $spacing-4;
  padding: $spacing-6;
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: $border-radius-xl;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: $font-size-2xl;
  color: $white;
  flex-shrink: 0;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: $font-size-3xl;
  font-weight: 700;
  color: $gray-800;
  margin-bottom: $spacing-1;
}

.stat-label {
  font-size: $font-size-sm;
  color: $gray-600;
  margin-bottom: $spacing-2;
  font-weight: 500;
}

.stat-change {
  display: flex;
  align-items: center;
  gap: $spacing-1;
  font-size: $font-size-sm;
  font-weight: 500;

  &.positive {
    color: $success-color;
  }

  &.negative {
    color: $danger-color;
  }
}

.dashboard-content {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: $spacing-8;
}

.charts-section {
  display: flex;
  flex-direction: column;
  gap: $spacing-6;
}

.chart-container {
  padding: $spacing-6;
  min-height: 350px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: $spacing-6;
}

.chart-title {
  font-size: $font-size-lg;
  font-weight: 600;
  color: $gray-800;
  margin: 0;
}

.chart-content {
  height: 280px;
  position: relative;
}

.side-section {
  display: flex;
  flex-direction: column;
  gap: $spacing-6;
}

.alerts-container {
  padding: $spacing-6;
  max-height: 500px;
}

.alerts-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: $spacing-4;
}

.alerts-title {
  display: flex;
  align-items: center;
  gap: $spacing-2;
  font-size: $font-size-lg;
  font-weight: 600;
  color: $gray-800;
  margin: 0;
}

.alerts-list {
  max-height: 400px;
  overflow-y: auto;
}

.alert-item {
  display: flex;
  gap: $spacing-3;
  padding: $spacing-4;
  border-radius: $border-radius-lg;
  margin-bottom: $spacing-3;
  border-left: 4px solid;

  &.critical {
    background: rgba(239, 68, 68, 0.1);
    border-left-color: $danger-color;
  }

  &.warning {
    background: rgba(245, 158, 11, 0.1);
    border-left-color: $warning-color;
  }
}

.alert-icon {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  border-radius: $border-radius;
  display: flex;
  align-items: center;
  justify-content: center;
  color: $white;

  .alert-item.critical & {
    background: $danger-color;
  }

  .alert-item.warning & {
    background: $warning-color;
  }
}

.alert-content {
  flex: 1;
}

.alert-title {
  font-weight: 600;
  color: $gray-800;
  margin-bottom: $spacing-1;
}

.alert-message {
  font-size: $font-size-sm;
  color: $gray-600;
  margin-bottom: $spacing-1;
}

.alert-time {
  font-size: $font-size-xs;
  color: $gray-500;
}

.quick-actions {
  padding: $spacing-6;
}

.actions-title {
  font-size: $font-size-lg;
  font-weight: 600;
  color: $gray-800;
  margin: 0 0 $spacing-4 0;
}

.action-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: $spacing-3;
}

.action-btn {
  height: 48px;
  border-radius: $border-radius-lg;
  font-weight: 500;
}

// 响应式设计
@media (max-width: $breakpoint-lg) {
  .dashboard-header {
    flex-direction: column;
    align-items: stretch;
    gap: $spacing-4;
  }

  .header-filters {
    justify-content: flex-start;
    flex-wrap: wrap;
  }

  .dashboard-content {
    grid-template-columns: 1fr;
  }

  .stats-grid {
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  }
}

@media (max-width: $breakpoint-md) {
  .dashboard {
    padding: $spacing-4;
  }

  .action-buttons {
    grid-template-columns: 1fr;
  }

  .header-filters {
    flex-direction: column;
    gap: $spacing-3;

    .el-select,
    .el-date-picker {
      width: 100% !important;
    }
  }
}

@media (max-width: $breakpoint-sm) {
  .stats-grid {
    grid-template-columns: 1fr;
  }

  .stat-card {
    flex-direction: column;
    text-align: center;
  }

  .page-title {
    font-size: $font-size-2xl;
  }
}
</style>
