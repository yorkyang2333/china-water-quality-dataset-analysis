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

  // 计算属性
  const stationCount = computed(() => stations.value.length)
  const criticalAlerts = computed(() => alerts.value.filter(alert => alert.level === 'critical'))
  const warningAlerts = computed(() => alerts.value.filter(alert => alert.level === 'warning'))
  const totalAlerts = computed(() => alerts.value.length)

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
      alerts.value = response.data.data || []
      return alerts.value
    } catch (error) {
      console.error('获取告警信息失败:', error)
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
    setProvince,
    setDateRange,
    resetFilters
  }
})
