import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useLoadingStore = defineStore('loading', () => {
  const globalLoading = ref(false)
  const pageLoading = ref(false)
  const chartLoading = ref(false)
  const mapLoading = ref(false)

  // 设置全局加载状态
  const setGlobalLoading = (loading) => {
    globalLoading.value = loading
  }

  // 设置页面加载状态
  const setPageLoading = (loading) => {
    pageLoading.value = loading
  }

  // 设置图表加载状态
  const setChartLoading = (loading) => {
    chartLoading.value = loading
  }

  // 设置地图加载状态
  const setMapLoading = (loading) => {
    mapLoading.value = loading
  }

  return {
    globalLoading,
    pageLoading,
    chartLoading,
    mapLoading,
    setGlobalLoading,
    setPageLoading,
    setChartLoading,
    setMapLoading
  }
})
