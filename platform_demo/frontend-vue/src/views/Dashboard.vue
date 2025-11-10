<template>
  <div class="dashboard">
    <!-- 页面标题 -->
    <div class="page-header">
      <div>
        <h1 class="page-title">数据概览</h1>
        <p class="page-subtitle">全国水质监测数据实时监控</p>
      </div>

      <div class="header-actions">
        <el-select
          v-model="selectedProvince"
          placeholder="选择省份"
          clearable
          @change="handleProvinceChange"
          style="width: 180px"
        >
          <el-option label="全部" value="" />
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
          style="width: 260px"
        />
      </div>
    </div>

    <!-- 统计卡片 -->
    <el-row :gutter="20" class="stats-row">
      <el-col :xs="24" :sm="12" :md="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-icon-wrapper blue">
            <el-icon :size="32"><Location /></el-icon>
          </div>
          <div class="stat-value">{{ stationCount }}</div>
          <div class="stat-label">监测站点</div>
          <div class="stat-trend" :class="stationCountTrend >= 0 ? 'up' : 'down'">
            <el-icon v-if="stationCountTrend >= 0"><Top /></el-icon>
            <el-icon v-else><Bottom /></el-icon>
            <span v-if="stationCountTrend >= 0">+{{ stationCountTrend }}% 本月新增</span>
            <span v-else>{{ stationCountTrend }}% 本月减少</span>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="12" :md="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-icon-wrapper green">
            <el-icon :size="32"><CircleCheck /></el-icon>
          </div>
          <div class="stat-value">{{ statistics.excellent || statistics.Excellent || 0 }}</div>
          <div class="stat-label">优秀水质</div>
          <div class="stat-trend" :class="excellentCountTrend >= 0 ? 'up' : 'down'">
            <el-icon v-if="excellentCountTrend >= 0"><Top /></el-icon>
            <el-icon v-else><Bottom /></el-icon>
            <span v-if="excellentCountTrend >= 0">+{{ excellentCountTrend }}% 较上月</span>
            <span v-else>{{ excellentCountTrend }}% 较上月</span>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="12" :md="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-icon-wrapper orange">
            <el-icon :size="32"><Warning /></el-icon>
          </div>
          <div class="stat-value">{{ warningAlerts.length }}</div>
          <div class="stat-label">预警站点</div>
          <div class="stat-trend" :class="warningStationsTrend >= 0 ? 'up' : 'down'">
            <el-icon v-if="warningStationsTrend >= 0"><Top /></el-icon>
            <el-icon v-else><Bottom /></el-icon>
            <span v-if="warningStationsTrend >= 0">+{{ warningStationsTrend }}% 较上月</span>
            <span v-else>{{ warningStationsTrend }}% 较上月</span>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="12" :md="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-icon-wrapper red">
            <el-icon :size="32"><CircleClose /></el-icon>
          </div>
          <div class="stat-value">{{ criticalAlerts.length }}</div>
          <div class="stat-label">严重污染</div>
          <div class="stat-trend" :class="criticalStationsTrend >= 0 ? 'up' : 'down'">
            <el-icon v-if="criticalStationsTrend >= 0"><Top /></el-icon>
            <el-icon v-else><Bottom /></el-icon>
            <span v-if="criticalStationsTrend >= 0">+{{ criticalStationsTrend }}% 需关注</span>
            <span v-else>{{ criticalStationsTrend }}% 已改善</span>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 图表区域 -->
    <el-row :gutter="20" class="charts-row">
      <!-- 污染等级分布 -->
      <el-col :xs="24" :md="12">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <span class="card-title">污染等级分布</span>
            </div>
          </template>
          <div class="chart-wrapper">
            <canvas ref="pollutionChart"></canvas>
          </div>
        </el-card>
      </el-col>

      <!-- 水质趋势 -->
      <el-col :xs="24" :md="12">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <span class="card-title">水质趋势</span>
              <el-select
                v-model="trendIndicator"
                @change="updateTrendChart"
                size="small"
                style="width: 120px"
              >
                <el-option label="水质指数" value="Water_Quality_Index" />
                <el-option label="COD" value="COD_mg_L" />
                <el-option label="总氮" value="Total_Nitrogen_mg_L" />
                <el-option label="总磷" value="Total_Phosphorus_mg_L" />
              </el-select>
            </div>
          </template>
          <div class="chart-wrapper">
            <canvas ref="trendChart"></canvas>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 实时告警和快捷操作 -->
    <el-row :gutter="20" class="bottom-row">
      <!-- 实时告警 -->
      <el-col :xs="24" :md="16">
        <el-card shadow="hover" class="alerts-card">
          <template #header>
            <div class="card-header">
              <span class="card-title">
                <el-icon><Bell /></el-icon>
                实时告警
              </span>
              <el-badge :value="recentAlerts.length" type="danger" />
            </div>
          </template>
          <el-scrollbar height="320px">
            <div v-if="recentAlerts.length === 0" class="empty-state">
              <el-icon :size="48" color="#C0C4CC"><CircleCheck /></el-icon>
              <p>暂无告警信息</p>
            </div>
            <div v-else class="alerts-list">
              <div
                v-for="alert in recentAlerts"
                :key="alert.id"
                class="alert-item"
                :class="alert.level"
              >
                <div class="alert-icon">
                  <el-icon v-if="alert.level === 'critical'"><CircleClose /></el-icon>
                  <el-icon v-else><Warning /></el-icon>
                </div>
                <div class="alert-content">
                  <div class="alert-title">{{ alert.station }}</div>
                  <div class="alert-message">{{ alert.message }}</div>
                  <div class="alert-time">{{ alert.time }}</div>
                </div>
              </div>
            </div>
          </el-scrollbar>
        </el-card>
      </el-col>

      <!-- 快捷操作 -->
      <el-col :xs="24" :md="8">
        <el-card shadow="hover" class="quick-actions-card">
          <template #header>
            <span class="card-title">快捷操作</span>
          </template>
          <div class="quick-actions">
            <el-button type="primary" @click="$router.push('/monitoring')">
              <el-icon><View /></el-icon>
              <span>实时监测</span>
            </el-button>
            <el-button type="primary" @click="$router.push('/prediction')">
              <el-icon><TrendCharts /></el-icon>
              <span>智能预测</span>
            </el-button>
            <el-button type="primary" @click="$router.push('/analysis')">
              <el-icon><PieChart /></el-icon>
              <span>数据分析</span>
            </el-button>
            <el-button type="primary" @click="$router.push('/spatial')">
              <el-icon><Location /></el-icon>
              <span>空间分析</span>
            </el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  Location,
  CircleCheck,
  Warning,
  CircleClose,
  Bell,
  View,
  TrendCharts,
  PieChart,
  Top,
  Bottom
} from '@element-plus/icons-vue'
import { Chart } from 'chart.js/auto'
import { useWaterQualityStore } from '@/stores/waterQuality'
import { useLoadingStore } from '@/stores/loading'

const router = useRouter()
const waterQualityStore = useWaterQualityStore()
const loadingStore = useLoadingStore()

// 响应式数据
const selectedProvince = ref('')  // 默认为空，显示全部省份数据
const dateRange = ref([new Date('2023-01-01'), new Date('2023-12-31')])
const trendIndicator = ref('Water_Quality_Index')
const pollutionChart = ref(null)
const trendChart = ref(null)

// 趋势数据
const stationCountTrend = ref(0)
const excellentCountTrend = ref(0)
const warningStationsTrend = ref(0)
const criticalStationsTrend = ref(0)

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
  '北京', '广东', '河南', '湖北', '江苏',
  '山东', '上海', '四川', '云南', '浙江'
])

// 中文到英文的映射（用于API调用）
const chineseToEnglish = {
  '北京': 'Beijing',
  '广东': 'Guangdong',
  '河南': 'Henan',
  '湖北': 'Hubei',
  '江苏': 'Jiangsu',
  '山东': 'Shandong',
  '上海': 'Shanghai',
  '四川': 'Sichuan',
  '云南': 'Yunnan',
  '浙江': 'Zhejiang'
}

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
  console.log('Province changed to:', province)

  loadingStore.setPageLoading(true)
  try {
    // 转换中文省份名为英文用于API调用
    const englishProvince = province ? (chineseToEnglish[province] || province) : ''

    // 重新加载所有数据
    await Promise.all([
      waterQualityStore.fetchStations(),
      waterQualityStore.fetchStatistics(englishProvince),
      waterQualityStore.fetchRealtimeData(englishProvince),
      waterQualityStore.fetchAlerts(englishProvince),
      loadStatisticsTrend(englishProvince)
    ])
    console.log('Statistics after province change:', statistics.value)
    updatePollutionChart()
    updateTrendChart()
  } catch (error) {
    console.error('Failed to load data for province:', error)
  } finally {
    loadingStore.setPageLoading(false)
  }
}

const handleDateChange = async (dates) => {
  if (dates && dates.length === 2) {
    loadingStore.setPageLoading(true)
    try {
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
          '#67C23A',
          '#409EFF',
          '#E6A23C',
          '#F56C6C',
          '#909399'
        ],
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 15,
            font: {
              size: 12
            }
          }
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
        borderColor: '#409EFF',
        backgroundColor: 'rgba(64, 158, 255, 0.1)',
        tension: 0.4,
        fill: true,
        pointRadius: 4,
        pointHoverRadius: 6
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
          beginAtZero: true,
          grid: {
            color: 'rgba(0, 0, 0, 0.05)'
          }
        },
        x: {
          grid: {
            display: false
          }
        }
      }
    }
  })
}

const updatePollutionChart = async () => {
  if (!pollutionChartInstance) return

  try {
    console.log('Updating pollution chart for province:', selectedProvince.value)

    const englishProvince = selectedProvince.value ? (chineseToEnglish[selectedProvince.value] || selectedProvince.value) : ''

    const result = await waterQualityStore.fetchPollutionDistribution({
      province: englishProvince
    })

    console.log('Pollution distribution API result:', result)

    // API返回格式: { data: { distribution: [{level, count, percentage}], total_records } }
    if (result?.data?.distribution && Array.isArray(result.data.distribution) && result.data.distribution.length > 0) {
      // 提取count值用于图表
      const counts = result.data.distribution.map(item => item.count || 0)
      console.log('Extracted counts for chart:', counts)
      pollutionChartInstance.data.datasets[0].data = counts
    } else {
      // 使用默认数据
      console.log('Using default data for pollution chart')
      pollutionChartInstance.data.datasets[0].data = [30, 50, 40, 60, 20]
    }
    pollutionChartInstance.update()
  } catch (error) {
    console.error('更新污染分布图表失败:', error)
    // 保持现有数据或使用默认数据
    if (!pollutionChartInstance.data.datasets[0].data ||
        pollutionChartInstance.data.datasets[0].data.length === 0) {
      pollutionChartInstance.data.datasets[0].data = [30, 50, 40, 60, 20]
    }
    pollutionChartInstance.update()
  }
}

const updateTrendChart = async () => {
  if (!trendChartInstance) return

  try {
    console.log('Updating trend chart for province:', selectedProvince.value)
    console.log('Indicator:', trendIndicator.value)

    const englishProvince = selectedProvince.value ? (chineseToEnglish[selectedProvince.value] || selectedProvince.value) : ''

    const result = await waterQualityStore.getTrends({
      indicator: trendIndicator.value,
      province: englishProvince
    })

    console.log('Trend API result:', result)

    // API返回格式: { data: [{period: string, value: number}] }
    if (result?.data && Array.isArray(result.data) && result.data.length > 0) {
      // 提取value值和period标签
      const values = result.data.map(item => item.value)
      const labels = result.data.map(item => item.period)

      console.log('Extracted values:', values)
      console.log('Extracted labels:', labels)

      trendChartInstance.data.labels = labels
      trendChartInstance.data.datasets[0].data = values
    } else {
      // 使用默认数据
      console.log('Using default data for trend chart')
      const mockData = Array.from({ length: 6 }, () => 50 + Math.random() * 30)
      trendChartInstance.data.datasets[0].data = mockData
    }
    trendChartInstance.update()
  } catch (error) {
    console.error('更新趋势图表失败:', error)
    // 保持现有数据或使用默认数据
    if (!trendChartInstance.data.datasets[0].data ||
        trendChartInstance.data.datasets[0].data.length === 0) {
      const mockData = Array.from({ length: 6 }, () => 50 + Math.random() * 30)
      trendChartInstance.data.datasets[0].data = mockData
    }
    trendChartInstance.update()
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
    console.log('Loading initial data for province:', selectedProvince.value)

    const englishProvince = selectedProvince.value ? (chineseToEnglish[selectedProvince.value] || selectedProvince.value) : ''

    // 使用默认省份加载初始数据
    await Promise.all([
      waterQualityStore.fetchStations().catch(e => {
        console.warn('Failed to fetch stations:', e)
        return []
      }),
      waterQualityStore.fetchStatistics(englishProvince).catch(e => {
        console.warn('Failed to fetch statistics:', e)
        return {}
      }),
      waterQualityStore.fetchRealtimeData(englishProvince).catch(e => {
        console.warn('Failed to fetch realtime data:', e)
        return []
      }),
      waterQualityStore.fetchAlerts(englishProvince).catch(e => {
        console.warn('Failed to fetch alerts:', e)
        return []
      }),
      loadStatisticsTrend(englishProvince).catch(e => {
        console.warn('Failed to load statistics trend:', e)
      })
    ])
    console.log('Initial statistics loaded:', statistics.value)
  } catch (error) {
    console.error('Failed to load initial data:', error)
  } finally {
    loadingStore.setGlobalLoading(false)
  }
}

const loadStatisticsTrend = async (province = '') => {
  try {
    const response = await fetch(`/api/statistics/trend${province ? `?province=${province}` : ''}`)
    const result = await response.json()

    if (result.success && result.data) {
      stationCountTrend.value = result.data.station_count_change
      excellentCountTrend.value = result.data.excellent_count_change
      warningStationsTrend.value = result.data.warning_stations_change
      criticalStationsTrend.value = result.data.critical_stations_change
      console.log('Statistics trend loaded:', result.data)
      console.log(`对比周期: ${result.data.previous_month} → ${result.data.current_month}`)
    }
  } catch (error) {
    console.error('Failed to load statistics trend:', error)
    // 使用默认值 0 如果加载失败
  }
}

// 生命周期
onMounted(async () => {
  await loadInitialData()

  setTimeout(() => {
    initPollutionChart()
    initTrendChart()
  }, 100)
})
</script>

<style lang="scss" scoped>
.dashboard {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  flex-wrap: wrap;
  gap: 20px;
}

.page-title {
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  margin: 0 0 8px 0;
}

.page-subtitle {
  font-size: 14px;
  color: #909399;
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

// 统计卡片
.stats-row {
  margin-bottom: 20px;
}

.stat-card {
  text-align: center;
  padding: 20px 10px;
  transition: all 0.3s;

  &:hover {
    transform: translateY(-4px);
  }

  .stat-icon-wrapper {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px;
    color: #fff;

    &.blue {
      background-color: #409EFF;
    }

    &.green {
      background-color: #67C23A;
    }

    &.orange {
      background-color: #E6A23C;
    }

    &.red {
      background-color: #F56C6C;
    }
  }

  .stat-value {
    font-size: 32px;
    font-weight: 700;
    color: #303133;
    margin-bottom: 8px;
  }

  .stat-label {
    font-size: 14px;
    color: #909399;
    margin-bottom: 12px;
  }

  .stat-trend {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 12px;

    &.up {
      color: #67C23A;
      background-color: rgba(103, 194, 58, 0.1);
    }

    &.down {
      color: #F56C6C;
      background-color: rgba(245, 108, 108, 0.1);
    }
  }
}

// 图表区域
.charts-row {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  display: flex;
  align-items: center;
  gap: 8px;
}

.chart-wrapper {
  height: 300px;
  padding: 10px;
}

// 告警列表
.bottom-row {
  margin-bottom: 20px;
}

.alerts-card {
  :deep(.el-card__body) {
    padding: 0 20px 20px;
  }
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #909399;

  p {
    margin-top: 16px;
    font-size: 14px;
  }
}

.alerts-list {
  padding: 10px 0;
}

.alert-item {
  display: flex;
  gap: 12px;
  padding: 12px;
  border-radius: 8px;
  margin-bottom: 10px;
  border-left: 3px solid;
  transition: all 0.3s;

  &:hover {
    transform: translateX(4px);
  }

  &.critical {
    background-color: rgba(245, 108, 108, 0.05);
    border-left-color: #F56C6C;
  }

  &.warning {
    background-color: rgba(230, 162, 60, 0.05);
    border-left-color: #E6A23C;
  }

  .alert-icon {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 18px;
    color: #fff;
  }

  &.critical .alert-icon {
    background-color: #F56C6C;
  }

  &.warning .alert-icon {
    background-color: #E6A23C;
  }

  .alert-content {
    flex: 1;
  }

  .alert-title {
    font-size: 14px;
    font-weight: 600;
    color: #303133;
    margin-bottom: 4px;
  }

  .alert-message {
    font-size: 13px;
    color: #606266;
    margin-bottom: 4px;
  }

  .alert-time {
    font-size: 12px;
    color: #909399;
  }
}

// 快捷操作
.quick-actions-card {
  height: 100%;
  display: flex;
  flex-direction: column;

  :deep(.el-card__body) {
    flex: 1;
    display: flex;
    flex-direction: column;
  }
}

.quick-actions {
  display: flex;
  flex-direction: column;
  gap: 16px;
  justify-content: center;
  align-items: stretch;
  flex: 1;

  .el-button {
    width: 100%;
    height: 50px;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0;
    margin: 0;

    .el-icon {
      font-size: 18px;
      width: 20px;
      height: 20px;
      margin-right: 8px;
      flex-shrink: 0;
    }

    span {
      flex-shrink: 0;
    }
  }
}

// 响应式
@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: stretch;
  }

  .header-actions {
    flex-direction: column;

    .el-select,
    .el-date-picker {
      width: 100% !important;
    }
  }

  .page-title {
    font-size: 24px;
  }
}
</style>
