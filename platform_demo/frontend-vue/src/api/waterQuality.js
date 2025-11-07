import axios from 'axios'
import { ElMessage } from 'element-plus'

// 创建axios实例
const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加token等认证信息
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('响应错误:', error)
    
    // 显示错误消息
    const message = error.response?.data?.message || error.message || '请求失败'
    ElMessage.error(message)
    
    return Promise.reject(error)
  }
)

export const waterQualityApi = {
  // 获取所有监测站点
  getStations() {
    return api.get('/stations')
  },

  // 获取统计数据
  getStatistics(province = '') {
    const params = province ? { province } : {}
    return api.get('/statistics/summary', { params })
  },

  // 获取实时监测数据
  getRealtimeMonitoring(province = '') {
    const params = province ? { province } : {}
    return api.get('/monitoring/realtime', { params })
  },

  // 获取告警信息
  getAlerts(province = '') {
    const params = province ? { province } : {}
    return api.get('/monitoring/alerts', { params })
  },

  // 获取数据质量信息
  getDataQuality(province = '') {
    const params = province ? { province } : {}
    return api.get('/monitoring/data-quality', { params })
  },

  // 获取时间轴数据
  getTimelineData(params = {}) {
    return api.get('/data/timeline', { params })
  },

  // 空间查询
  spatialQuery(data) {
    return api.post('/data/spatial', data)
  },

  // 污染等级预测
  predictPollutionLevel(formData) {
    return api.post('/predict/pollution_level', formData)
  },

  // 水质指标预测
  predictWaterQuality(formData) {
    return api.post('/predict/water_quality', formData)
  },

  // 获取趋势数据
  getTrends(params = {}) {
    return api.get('/trends', { params })
  },

  // 获取污染分布数据
  getPollutionDistribution(params = {}) {
    return api.get('/charts/pollution-distribution', { params })
  },

  // 获取省份对比数据
  getProvinceComparison(params = {}) {
    return api.get('/charts/province-comparison', { params })
  },

  // 获取指标相关性数据
  getIndicatorCorrelation(params = {}) {
    return api.get('/charts/indicator-correlation', { params })
  },

  // 获取多指标趋势数据
  getMultiIndicatorTrend(params = {}) {
    return api.get('/charts/multi-indicator-trend', { params })
  }
}

export default api
