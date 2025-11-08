import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { waterQualityApi } from '@/api/waterQuality'

export const useWaterQualityStore = defineStore('waterQuality', () => {
  // 状态
  const stations = ref([])
  const statistics = ref({})
  const realtimeData = ref([])
  const alerts = ref([])
  const selectedProvince = ref('')
  const selectedDateRange = ref([])

  // 计算属性 - 添加数组防护
  const stationCount = computed(() => stations.value.length)
  const criticalAlerts = computed(() => Array.isArray(alerts.value) ? alerts.value.filter(alert => alert.level === 'critical') : [])
  const warningAlerts = computed(() => Array.isArray(alerts.value) ? alerts.value.filter(alert => alert.level === 'warning') : [])
  const totalAlerts = computed(() => Array.isArray(alerts.value) ? alerts.value.length : 0)

  // 获取所有站点
  const fetchStations = async () => {
    try {
      const response = await waterQualityApi.getStations()
      stations.value = response.data.data || []
      return stations.value
    } catch (error) {
      console.error('获取站点数据失败:', error)
      throw error
    }
  }

  // 获取统计数据
  const fetchStatistics = async (province = '') => {
    try {
      const response = await waterQualityApi.getStatistics(province)
      statistics.value = response.data.data || {}
      return statistics.value
    } catch (error) {
      console.error('获取统计数据失败:', error)
      throw error
    }
  }

  // 获取实时监测数据
  const fetchRealtimeData = async (province = '') => {
    try {
      const response = await waterQualityApi.getRealtimeMonitoring(province)
      realtimeData.value = response.data.data || []
      return realtimeData.value
    } catch (error) {
      console.error('获取实时数据失败:', error)
      throw error
    }
  }

  // 获取告警信息
  const fetchAlerts = async (province = '') => {
    try {
      const response = await waterQualityApi.getAlerts(province)
      // 确保返回的是数组格式
      const data = response.data.data
      if (data && typeof data === 'object') {
        // 后端返回的是对象 { total_alerts, critical_count, warning_count, alerts: [] }
        alerts.value = data.alerts || []
      } else if (Array.isArray(data)) {
        alerts.value = data
      } else {
        alerts.value = []
      }
      return alerts.value
    } catch (error) {
      console.error('获取告警信息失败:', error)
      alerts.value = [] // 确保出错时也是数组
      throw error
    }
  }

  // 获取预测结果
  const predictPollutionLevel = async (formData) => {
    try {
      const response = await waterQualityApi.predictPollutionLevel(formData)
      return response.data
    } catch (error) {
      console.error('污染等级预测失败:', error)
      throw error
    }
  }

  // 获取图表数据
  const fetchPollutionDistribution = async (params = {}) => {
    try {
      const response = await waterQualityApi.getPollutionDistribution(params)
      return response.data
    } catch (error) {
      console.error('获取污染分布数据失败:', error)
      throw error
    }
  }

  // 获取趋势数据
  const getTrends = async (params = {}) => {
    try {
      const response = await waterQualityApi.getTrends(params)
      return response.data
    } catch (error) {
      console.error('获取趋势数据失败:', error)
      throw error
    }
  }

  // 获取省份对比数据
  const getProvinceComparison = async (params = {}) => {
    try {
      const response = await waterQualityApi.getProvinceComparison(params)
      return response.data
    } catch (error) {
      console.error('获取省份对比数据失败:', error)
      throw error
    }
  }

  // 设置筛选条件
  const setProvince = (province) => {
    selectedProvince.value = province
  }

  const setDateRange = (dateRange) => {
    selectedDateRange.value = dateRange
  }

  // 重置筛选条件
  const resetFilters = () => {
    selectedProvince.value = ''
    selectedDateRange.value = []
  }

  return {
    // 状态
    stations,
    statistics,
    realtimeData,
    alerts,
    selectedProvince,
    selectedDateRange,
    
    // 计算属性
    stationCount,
    criticalAlerts,
    warningAlerts,
    totalAlerts,
    
    // 方法
    fetchStations,
    fetchStatistics,
    fetchRealtimeData,
    fetchAlerts,
    predictPollutionLevel,
    fetchPollutionDistribution,
    getTrends,
    getProvinceComparison,
    setProvince,
    setDateRange,
    resetFilters
  }
})
