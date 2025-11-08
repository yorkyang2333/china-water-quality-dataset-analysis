<template>
  <div class="quality-container">
    <el-card class="summary-card">
      <template #header>
        <div class="card-header">
          <span>数据质量概览</span>
          <el-button type="primary" :icon="Refresh" @click="refreshData">刷新</el-button>
        </div>
      </template>

      <el-row :gutter="20">
        <el-col :span="6">
          <el-statistic title="数据完整度" :value="qualityMetrics.completeness" suffix="%" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="数据准确度" :value="qualityMetrics.accuracy" suffix="%" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="及时性" :value="qualityMetrics.timeliness" suffix="%" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="一致性" :value="qualityMetrics.consistency" suffix="%" />
        </el-col>
      </el-row>
    </el-card>

    <el-row :gutter="20">
      <el-col :span="12">
        <el-card class="chart-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <span>数据质量评分</span>
            </div>
          </template>
          <div ref="radarChart" style="height: 400px;"></div>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card class="chart-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <span>数据完整性趋势</span>
            </div>
          </template>
          <div ref="lineChart" style="height: 400px;"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="table-card" v-loading="loading">
      <template #header>
        <div class="card-header">
          <span>站点数据质量详情</span>
          <el-select v-model="selectedProvince" placeholder="选择省份" clearable @change="fetchData">
            <el-option label="全部省份" value="" />
            <el-option v-for="p in provinces" :key="p" :label="p" :value="p" />
          </el-select>
        </div>
      </template>

      <el-table :data="stationQuality" style="width: 100%">
        <el-table-column prop="station" label="监测站点" width="200" />
        <el-table-column prop="province" label="省份" width="120" />
        <el-table-column prop="dataCount" label="数据量" width="100" />
        <el-table-column prop="completeness" label="完整度" width="120">
          <template #default="{ row }">
            <el-progress :percentage="row.completeness" :color="getProgressColor(row.completeness)" />
          </template>
        </el-table-column>
        <el-table-column prop="accuracy" label="准确度" width="120">
          <template #default="{ row }">
            <el-progress :percentage="row.accuracy" :color="getProgressColor(row.accuracy)" />
          </template>
        </el-table-column>
        <el-table-column prop="lastUpdate" label="最后更新" width="180" />
        <el-table-column prop="missingFields" label="缺失字段" />
        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getQualityStatus(row.completeness)">
              {{ row.completeness >= 90 ? '优秀' : row.completeness >= 70 ? '良好' : '需改进' }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="totalStations"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        style="margin-top: 20px; text-align: right;"
        @size-change="fetchData"
        @current-change="fetchData"
      />
    </el-card>

    <el-card class="issues-card">
      <template #header>
        <div class="card-header">
          <span>数据质量问题</span>
        </div>
      </template>

      <el-table :data="qualityIssues">
        <el-table-column type="index" label="#" width="50" />
        <el-table-column prop="station" label="监测站点" width="200" />
        <el-table-column prop="issue" label="问题描述" />
        <el-table-column prop="severity" label="严重程度" width="120">
          <template #default="{ row }">
            <el-tag :type="getSeverityType(row.severity)">
              {{ row.severity }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="detectedAt" label="发现时间" width="180" />
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <span class="action-text primary" @click="handleIssue(row)">处理</span>
            <span class="action-text danger" @click="ignoreIssue(row)">忽略</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { Refresh } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'

const loading = ref(false)
const selectedProvince = ref('')
const provinces = ref(['广东省', '浙江省', '江苏省', '山东省', '河南省', '四川省', '湖北省', '湖南省', '福建省', '安徽省'])

const qualityMetrics = ref({
  completeness: 92.5,
  accuracy: 95.2,
  timeliness: 88.7,
  consistency: 91.3
})

const stationQuality = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const totalStations = ref(180)

const qualityIssues = ref([
  { station: '广州站点01', issue: '缺失pH值数据', severity: '中等', detectedAt: '2024-01-15 14:30' },
  { station: '深圳站点05', issue: '溶解氧数值异常', severity: '严重', detectedAt: '2024-01-15 13:20' },
  { station: '杭州站点03', issue: '数据更新延迟超过24小时', severity: '轻微', detectedAt: '2024-01-15 12:10' },
  { station: '苏州站点02', issue: '总氮数据缺失', severity: '中等', detectedAt: '2024-01-15 10:45' }
])

const radarChart = ref(null)
const lineChart = ref(null)

let radarChartInstance = null
let lineChartInstance = null

const getProgressColor = (percentage) => {
  if (percentage >= 90) return '#67C23A'
  if (percentage >= 70) return '#E6A23C'
  return '#F56C6C'
}

const getQualityStatus = (completeness) => {
  if (completeness >= 90) return 'success'
  if (completeness >= 70) return 'warning'
  return 'danger'
}

const getSeverityType = (severity) => {
  if (severity === '严重') return 'danger'
  if (severity === '中等') return 'warning'
  return 'info'
}

const initRadarChart = () => {
  if (radarChartInstance) {
    radarChartInstance.dispose()
  }
  radarChartInstance = echarts.init(radarChart.value)

  const option = {
    tooltip: {},
    radar: {
      indicator: [
        { name: '完整度', max: 100 },
        { name: '准确度', max: 100 },
        { name: '及时性', max: 100 },
        { name: '一致性', max: 100 },
        { name: '有效性', max: 100 }
      ]
    },
    series: [
      {
        name: '数据质量评分',
        type: 'radar',
        data: [
          {
            value: [92.5, 95.2, 88.7, 91.3, 93.8],
            name: '当前评分',
            areaStyle: {
              color: 'rgba(88, 127, 255, 0.3)'
            }
          }
        ]
      }
    ]
  }

  radarChartInstance.setOption(option)
}

const initLineChart = () => {
  if (lineChartInstance) {
    lineChartInstance.dispose()
  }
  lineChartInstance = echarts.init(lineChart.value)

  const days = Array.from({ length: 30 }, (_, i) => `${i + 1}日`)
  const data = days.map(() => 85 + Math.random() * 15)

  const option = {
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      data: days
    },
    yAxis: {
      type: 'value',
      name: '完整度 (%)',
      min: 80,
      max: 100
    },
    series: [
      {
        data: data,
        type: 'line',
        smooth: true,
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(88, 127, 255, 0.3)' },
            { offset: 1, color: 'rgba(88, 127, 255, 0.05)' }
          ])
        }
      }
    ]
  }

  lineChartInstance.setOption(option)
}

const fetchData = async () => {
  loading.value = true
  try {
    // 模拟获取站点质量数据
    stationQuality.value = Array.from({ length: pageSize.value }, (_, i) => ({
      station: `${selectedProvince.value || '各省'}站点${(currentPage.value - 1) * pageSize.value + i + 1}`,
      province: selectedProvince.value || provinces.value[i % provinces.value.length],
      dataCount: Math.floor(1000 + Math.random() * 2000),
      completeness: Math.floor(70 + Math.random() * 30),
      accuracy: Math.floor(75 + Math.random() * 25),
      lastUpdate: new Date().toLocaleString('zh-CN'),
      missingFields: Math.random() > 0.7 ? 'pH, 浊度' : Math.random() > 0.5 ? '总磷' : '无'
    }))

    await nextTick()
    initRadarChart()
    initLineChart()
  } catch (error) {
    ElMessage.error('数据加载失败')
  } finally {
    loading.value = false
  }
}

const refreshData = () => {
  fetchData()
  ElMessage.success('数据已刷新')
}

const handleIssue = (row) => {
  ElMessage.info(`正在处理: ${row.issue}`)
}

const ignoreIssue = (row) => {
  const index = qualityIssues.value.findIndex(item => item === row)
  if (index > -1) {
    qualityIssues.value.splice(index, 1)
    ElMessage.success('已忽略该问题')
  }
}

onMounted(() => {
  fetchData()

  window.addEventListener('resize', () => {
    radarChartInstance?.resize()
    lineChartInstance?.resize()
  })
})
</script>

<style scoped>
.quality-container {
  padding: 20px;
}

.summary-card,
.chart-card,
.table-card,
.issues-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.action-text {
  cursor: pointer;
  margin-right: 12px;
  font-size: 14px;
  user-select: none;

  &.primary {
    color: var(--el-color-primary);

    &:hover {
      color: var(--el-color-primary-light-3);
    }
  }

  &.danger {
    color: var(--el-color-danger);

    &:hover {
      color: var(--el-color-danger-light-3);
    }
  }
}
</style>
