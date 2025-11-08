<template>
  <div class="monitoring-container">
    <el-card class="header-card">
      <template #header>
        <div class="card-header">
          <span>实时监测</span>
          <el-button type="primary" :icon="Refresh" @click="refreshData">刷新数据</el-button>
        </div>
      </template>

      <el-row :gutter="20">
        <el-col :span="6">
          <el-statistic title="在线站点" :value="onlineStations" suffix="个" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="监测指标" :value="monitoringIndicators" suffix="项" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="告警数量" :value="alertCount" suffix="条" />
        </el-col>
        <el-col :span="6">
          <div class="update-time">
            <div class="statistic-title">数据更新</div>
            <div class="statistic-value">{{ lastUpdateTime }}</div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- 告警信息卡片 -->
    <el-card class="alerts-card" v-if="store.alerts && store.alerts.length > 0">
      <template #header>
        <div class="card-header">
          <span>实时告警信息</span>
          <el-badge :value="alertCount" type="danger" />
        </div>
      </template>

      <el-table :data="displayAlerts" style="width: 100%">
        <el-table-column prop="station_name" label="监测站点" width="200" />
        <el-table-column prop="province" label="省份" width="120">
          <template #default="{ row }">
            {{ backendToChineseMap[row.province] || row.province }}
          </template>
        </el-table-column>
        <el-table-column prop="message" label="告警信息" />
        <el-table-column prop="level" label="级别" width="100">
          <template #default="{ row }">
            <el-tag :type="row.level === 'critical' ? 'danger' : 'warning'">
              {{ row.level === 'critical' ? '严重' : '警告' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="date" label="日期" width="120" />
        <el-table-column prop="value" label="指标值" width="100">
          <template #default="{ row }">
            {{ row.value?.toFixed(2) || 'N/A' }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card class="content-card" v-loading="loading">
      <template #header>
        <div class="card-header">
          <span>实时监测数据</span>
          <el-select v-model="selectedProvince" placeholder="选择省份" clearable @change="handleProvinceChange">
            <el-option label="全部省份" value="" />
            <el-option v-for="province in provinces" :key="province" :label="province" :value="province" />
          </el-select>
        </div>
      </template>

      <el-table :data="realtimeData" style="width: 100%">
        <el-table-column prop="station" label="监测站点" width="200" />
        <el-table-column prop="province" label="省份" width="120" />
        <el-table-column prop="waterQualityIndex" label="水质指数" width="120">
          <template #default="{ row }">
            <el-tag :type="getQualityType(row.waterQualityIndex)">
              {{ row.waterQualityIndex?.toFixed(2) || 'N/A' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="cod" label="COD (mg/L)" width="120" />
        <el-table-column prop="ph" label="pH值" width="100" />
        <el-table-column prop="dissolvedOxygen" label="溶解氧 (mg/L)" width="140" />
        <el-table-column prop="turbidity" label="浊度 (NTU)" width="120" />
        <el-table-column prop="updateTime" label="更新时间" width="180" />
        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === 'normal' ? 'success' : 'danger'">
              {{ row.status === 'normal' ? '正常' : '异常' }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { Refresh } from '@element-plus/icons-vue'
import { useWaterQualityStore } from '@/stores/waterQuality'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const store = useWaterQualityStore()
const loading = ref(false)
const selectedProvince = ref('')
const realtimeData = ref([])

const provinces = ref(['广东省', '浙江省', '江苏省', '山东省', '河南省', '四川省', '湖北省', '湖南省', '福建省', '安徽省', '北京', '上海', '云南省'])

// 中文省份名到后端英文名的映射
const provinceToEnglish = {
  '北京': 'Beijing',
  '上海': 'Shanghai',
  '广东省': 'Guangdong',
  '江苏省': 'Jiangsu',
  '浙江省': 'Zhejiang',
  '山东省': 'Shandong',
  '河南省': 'Henan',
  '四川省': 'Sichuan',
  '湖北省': 'Hubei',
  '湖南省': 'Hunan',
  '福建省': 'Fujian',
  '安徽省': 'Anhui',
  '云南省': 'Yunnan'
}

// 反向映射：后端省份名到中文
const backendToChineseMap = {
  'Beijing': '北京',
  'Shanghai': '上海',
  'Guangdong': '广东省',
  'Jiangsu': '江苏省',
  'Zhejiang': '浙江省',
  'Shandong': '山东省',
  'Henan': '河南省',
  'Sichuan': '四川省',
  'Hubei': '湖北省',
  'Hunan': '湖南省',
  'Fujian': '福建省',
  'Anhui': '安徽省',
  'Yunnan': '云南省'
}

const onlineStations = computed(() => realtimeData.value.filter(d => d.status === 'normal').length)
const monitoringIndicators = computed(() => 7)
const alertCount = computed(() => store.totalAlerts)
const lastUpdateTime = computed(() => new Date().toLocaleString('zh-CN'))

// 显示前20条告警
const displayAlerts = computed(() => {
  const alerts = store.alerts || []
  return alerts.slice(0, 20)
})

const getQualityType = (index) => {
  if (!index) return 'info'
  if (index < 50) return 'success'
  if (index < 100) return 'warning'
  return 'danger'
}

const fetchData = async () => {
  loading.value = true
  try {
    // 转换中文省份名为英文
    const englishProvince = selectedProvince.value ?
      (provinceToEnglish[selectedProvince.value] || selectedProvince.value) : ''

    console.log('Fetching monitoring data for province:', selectedProvince.value, '->', englishProvince)

    // 同时获取站点数据和告警信息
    await Promise.all([
      store.fetchStations(),
      store.fetchAlerts(englishProvince)
    ])

    // 获取站点数据
    const stations = store.stations || []

    console.log('Loaded stations:', stations.length)

    // 转换数据格式用于表格显示，并转换省份名为中文
    let filteredStations = stations.map(item => {
      const chineseProvince = backendToChineseMap[item.province] || item.province
      return {
        station: item.name || '未知站点',
        province: chineseProvince,
        waterQualityIndex: item.latest_wqi || 0,
        cod: (Math.random() * 30).toFixed(2), // TODO: 需要后端提供实际数据
        ph: (6 + Math.random() * 2).toFixed(1),
        dissolvedOxygen: (5 + Math.random() * 5).toFixed(2),
        turbidity: (Math.random() * 10).toFixed(1),
        updateTime: item.latest_date || new Date().toLocaleString('zh-CN'),
        status: item.latest_wqi < 60 ? 'normal' : 'abnormal'
      }
    })

    // 省份筛选
    if (selectedProvince.value) {
      filteredStations = filteredStations.filter(item => item.province === selectedProvince.value)
      console.log('After province filter:', filteredStations.length)
    }

    realtimeData.value = filteredStations.slice(0, 50) // 显示前50个站点

    console.log('Displayed stations:', realtimeData.value.length)
  } catch (error) {
    console.error('Error fetching data:', error)
    ElMessage.error('获取监测数据失败')
  } finally {
    loading.value = false
  }
}

const handleProvinceChange = () => {
  fetchData()
}

const refreshData = () => {
  fetchData()
  ElMessage.success('数据已刷新')
}

onMounted(() => {
  fetchData()
})
</script>

<style scoped>
.monitoring-container {
  padding: 20px;
}

.header-card {
  margin-bottom: 20px;
}

.alerts-card {
  margin-bottom: 20px;
}

.content-card {
  min-height: 600px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.update-time {
  text-align: center;
}

.statistic-title {
  font-size: 14px;
  color: var(--el-text-color-secondary);
  margin-bottom: 8px;
}

.statistic-value {
  font-size: 16px;
  color: var(--el-text-color-primary);
  font-weight: 500;
}
</style>
