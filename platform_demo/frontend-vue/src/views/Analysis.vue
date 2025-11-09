<template>
  <div class="analysis-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="filters">
        <el-form-item label="省份">
          <el-select v-model="filters.province" placeholder="选择省份" clearable style="width: 150px">
            <el-option label="全部" value="" />
            <el-option v-for="p in provinces" :key="p" :label="p" :value="p" />
          </el-select>
        </el-form-item>
        <el-form-item label="时间范围">
          <el-date-picker
            v-model="filters.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="fetchData">查询</el-button>
          <el-button @click="resetFilters">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-row :gutter="20">
      <el-col :span="12">
        <el-card class="chart-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <span>污染等级分布</span>
            </div>
          </template>
          <div ref="pieChart" style="height: 400px;"></div>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card class="chart-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <span>省份水质对比</span>
            </div>
          </template>
          <div ref="barChart" style="height: 400px;"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="12">
        <el-card class="chart-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <span>水质指标趋势分析</span>
              <el-select v-model="selectedIndicator" placeholder="选择指标" @change="updateTrendChart" style="width: 150px">
                <el-option label="水质指数" value="Water_Quality_Index" />
                <el-option label="COD" value="COD_mg_L" />
                <el-option label="pH值" value="pH" />
                <el-option label="溶解氧" value="Dissolved_Oxygen_mg_L" />
                <el-option label="总氮" value="Total_Nitrogen_mg_L" />
                <el-option label="总磷" value="Total_Phosphorus_mg_L" />
              </el-select>
            </div>
          </template>
          <div ref="lineChart" style="height: 400px;"></div>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card class="chart-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <span>指标相关性分析</span>
            </div>
          </template>
          <div ref="heatmapChart" style="height: 400px;"></div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { waterQualityApi } from '@/api/waterQuality'

const loading = ref(false)
const provinces = ref([
  '北京', '上海', '广东省', '江苏省', '浙江省',
  '山东省', '河南省', '四川省', '湖北省', '湖南省',
  '福建省', '安徽省', '云南省'
])

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

const filters = ref({
  province: '北京',  // 默认省份
  dateRange: [new Date('2023-01-01'), new Date('2024-12-31')]  // 默认时间范围
})

const selectedIndicator = ref('Water_Quality_Index')

const pieChart = ref(null)
const barChart = ref(null)
const lineChart = ref(null)
const heatmapChart = ref(null)

let pieChartInstance = null
let barChartInstance = null
let lineChartInstance = null
let heatmapChartInstance = null

const initPieChart = async () => {
  if (!pieChart.value) {
    console.warn('Pie chart DOM element not found')
    return
  }

  if (pieChartInstance) {
    pieChartInstance.dispose()
  }
  pieChartInstance = echarts.init(pieChart.value)

  try {
    // 转换省份名
    const englishProvince = filters.value.province ?
      (provinceToEnglish[filters.value.province] || filters.value.province) : ''

    console.log('Fetching pollution distribution for:', filters.value.province, '->', englishProvince)

    // 调用API获取污染等级分布
    const response = await waterQualityApi.getPollutionDistribution({
      province: englishProvince,
      start_date: filters.value.dateRange?.[0]?.toISOString().split('T')[0],
      end_date: filters.value.dateRange?.[1]?.toISOString().split('T')[0]
    })

    console.log('Pollution distribution response:', response.data)

    let chartData = []
    let chartTitle = filters.value.province || '全部'

    if (response.data?.data?.distribution && Array.isArray(response.data.data.distribution)) {
      // 使用真实数据
      const levelMap = {
        'Excellent': '优秀',
        'Good': '良好',
        'Fair': '一般',
        'Poor': '较差',
        'Very Poor': '很差'
      }
      chartData = response.data.data.distribution.map(item => ({
        value: item.count,
        name: levelMap[item.level] || item.level
      }))
    } else {
      // 备用数据
      chartData = [
        { value: 1048, name: '优秀' },
        { value: 735, name: '良好' },
        { value: 580, name: '一般' },
        { value: 484, name: '较差' },
        { value: 300, name: '很差' }
      ]
    }

    const option = {
      title: {
        text: `污染等级分布 (${chartTitle})`,
        left: 'center',
        top: 'top'
      },
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} ({d}%)'
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        top: 40
      },
      series: [
        {
          name: '污染等级',
          type: 'pie',
          radius: '50%',
          center: ['60%', '50%'],
          data: chartData,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    }

    pieChartInstance.setOption(option)
  } catch (error) {
    console.error('加载污染分布数据失败:', error)
  }
}

const initBarChart = async () => {
  if (barChartInstance) {
    barChartInstance.dispose()
  }
  barChartInstance = echarts.init(barChart.value)

  try {
    console.log('Fetching province comparison data')

    // 调用API获取省份对比数据
    const response = await waterQualityApi.getProvinceComparison({
      indicator: 'Water_Quality_Index'
    })

    console.log('Province comparison response:', response.data)

    let provinceNames = []
    let provinceValues = []
    let colors = []

    if (response.data?.data?.provinces && Array.isArray(response.data.data.provinces)) {
      // 使用真实数据
      const englishToChineseMap = {
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

      // 获取选中的省份（转换为英文）
      const selectedProvinceEnglish = filters.value.province ?
        (provinceToEnglish[filters.value.province] || '') : ''

      // 首先获取所有数据
      let allData = response.data.data.provinces.map(p => ({
        province: p.province,
        chineseName: englishToChineseMap[p.province] || p.province,
        average: p.average
      }))

      // 如果选中了省份，把它排到最前面
      if (selectedProvinceEnglish) {
        const selectedIndex = allData.findIndex(p => p.province === selectedProvinceEnglish)
        if (selectedIndex > -1) {
          const [selectedData] = allData.splice(selectedIndex, 1)
          allData.unshift(selectedData)
        }
      }

      // 提取数据和颜色
      provinceNames = allData.map(p => p.chineseName)
      provinceValues = allData.map(p => p.average)

      // 第一个是选中的省份，用特殊颜色，其他用默认颜色
      colors = allData.map((_, index) => index === 0 ? '#FF6B6B' : '#409EFF')
    } else {
      // 备用数据
      provinceNames = provinces.value.slice(0, 10)
      provinceValues = provinceNames.map(() => 50 + Math.random() * 40)
      colors = provinceNames.map((_, index) => index === 0 ? '#FF6B6B' : '#409EFF')
    }

    const option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        },
        formatter: '{b}<br/>平均水质指数: {c}'
      },
      xAxis: {
        type: 'category',
        data: provinceNames,
        axisLabel: {
          rotate: 45,
          interval: 0
        }
      },
      yAxis: {
        type: 'value',
        name: '平均水质指数'
      },
      grid: {
        bottom: 80
      },
      series: [
        {
          data: provinceValues.map((value, index) => ({
            value,
            itemStyle: {
              color: colors[index]
            }
          })),
          type: 'bar',
          barWidth: '60%'
        }
      ]
    }

    barChartInstance.setOption(option)
  } catch (error) {
    console.error('加载省份对比数据失败:', error)
  }
}

const initLineChart = async () => {
  if (lineChartInstance) {
    lineChartInstance.dispose()
  }
  lineChartInstance = echarts.init(lineChart.value)

  try {
    // 转换省份名
    const englishProvince = filters.value.province ?
      (provinceToEnglish[filters.value.province] || filters.value.province) : ''

    console.log('Fetching trend data for:', selectedIndicator.value, 'province:', englishProvince)

    // 调用API获取趋势数据
    const response = await waterQualityApi.getTrends({
      indicator: selectedIndicator.value,
      province: englishProvince,
      interval: 'month'
    })

    console.log('Trend response:', response.data)

    let periods = []
    let values = []

    if (response.data?.data && Array.isArray(response.data.data) && response.data.data.length > 0) {
      // 使用真实数据
      periods = response.data.data.map(item => item.period)
      values = response.data.data.map(item => item.value)
    } else {
      // 备用数据
      const months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
      periods = months
      values = months.map(() => 50 + Math.random() * 40)
    }

    const indicatorNames = {
      'Water_Quality_Index': '水质指数',
      'COD_mg_L': 'COD (mg/L)',
      'pH': 'pH值',
      'Dissolved_Oxygen_mg_L': '溶解氧 (mg/L)',
      'Total_Nitrogen_mg_L': '总氮 (mg/L)',
      'Total_Phosphorus_mg_L': '总磷 (mg/L)'
    }

    const option = {
      tooltip: {
        trigger: 'axis',
        formatter: '{b}<br/>{a}: {c}'
      },
      xAxis: {
        type: 'category',
        data: periods,
        axisLabel: {
          rotate: 45,
          interval: 0
        }
      },
      yAxis: {
        type: 'value',
        name: indicatorNames[selectedIndicator.value] || '指标值'
      },
      grid: {
        bottom: 80
      },
      series: [
        {
          name: indicatorNames[selectedIndicator.value] || '指标值',
          data: values,
          type: 'line',
          smooth: true,
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(58, 77, 233, 0.3)' },
              { offset: 1, color: 'rgba(58, 77, 233, 0.05)' }
            ])
          }
        }
      ]
    }

    lineChartInstance.setOption(option)
  } catch (error) {
    console.error('加载趋势数据失败:', error)
  }
}

const initHeatmapChart = async () => {
  if (heatmapChartInstance) {
    heatmapChartInstance.dispose()
  }
  heatmapChartInstance = echarts.init(heatmapChart.value)

  try {
    // 转换省份名
    const englishProvince = filters.value.province ?
      (provinceToEnglish[filters.value.province] || filters.value.province) : ''

    console.log('Fetching correlation data for province:', englishProvince)

    // 调用API获取相关性数据
    const response = await waterQualityApi.getIndicatorCorrelation({
      province: englishProvince
    })

    console.log('Correlation response:', response.data)

    let indicators = []
    let data = []

    if (response.data?.data?.indicators && response.data.data.correlation_matrix) {
      // 使用真实数据
      const indicatorNameMap = {
        'Water_Quality_Index': '水质指数',
        'COD_mg_L': 'COD',
        'pH': 'pH',
        'Total_Nitrogen_mg_L': '总氮',
        'Total_Phosphorus_mg_L': '总磷',
        'Dissolved_Oxygen_mg_L': '溶解氧',
        'Ammonia_Nitrogen_mg_L': '氨氮',
        'Turbidity_NTU': '浊度'
      }

      indicators = response.data.data.indicators.map(ind =>
        indicatorNameMap[ind] || ind
      )

      // 转换相关性矩阵为heatmap数据格式
      const matrix = response.data.data.correlation_matrix
      for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
          data.push([j, i, matrix[i][j].value])
        }
      }
    } else {
      // 备用数据
      indicators = ['水质指数', 'COD', 'pH', '溶解氧', '总氮', '总磷', '浊度']
      for (let i = 0; i < indicators.length; i++) {
        for (let j = 0; j < indicators.length; j++) {
          const value = i === j ? 1 : Math.random()
          data.push([j, i, parseFloat(value.toFixed(2))])
        }
      }
    }

    const option = {
      tooltip: {
        position: 'top',
        formatter: (params) => {
          return `${indicators[params.data[1]]} vs ${indicators[params.data[0]]}<br/>相关系数: ${params.data[2].toFixed(3)}`
        }
      },
      grid: {
        left: 80,
        top: 30,
        right: 30,
        bottom: 60
      },
      xAxis: {
        type: 'category',
        data: indicators,
        splitArea: {
          show: true
        },
        axisLabel: {
          rotate: 45,
          interval: 0,
          fontSize: 11
        }
      },
      yAxis: {
        type: 'category',
        data: indicators,
        splitArea: {
          show: true
        },
        axisLabel: {
          fontSize: 11
        }
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 10,
        itemWidth: 15,
        itemHeight: 120,
        text: ['高', '低'],
        inRange: {
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        }
      },
      series: [
        {
          name: '相关系数',
          type: 'heatmap',
          data: data,
          label: {
            show: true,
            fontSize: 10,
            formatter: (params) => params.data[2].toFixed(2)
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    }

    heatmapChartInstance.setOption(option)
  } catch (error) {
    console.error('加载相关性数据失败:', error)
  }
}

const updateTrendChart = () => {
  initLineChart()
}

// 监听省份变化，更新图表
watch(() => filters.value.province, () => {
  initPieChart()
  initBarChart()
  initHeatmapChart()
})

const fetchData = async () => {
  loading.value = true
  try {
    console.log('Fetching all chart data with filters:', filters.value)
    await nextTick()
    await Promise.all([
      initPieChart(),
      initBarChart(),
      initLineChart(),
      initHeatmapChart()
    ])
    console.log('All charts initialized successfully')
  } catch (error) {
    console.error('数据加载失败:', error)
    ElMessage.error('数据加载失败')
  } finally {
    loading.value = false
  }
}

const resetFilters = () => {
  filters.value = {
    province: '北京',
    dateRange: [new Date('2023-01-01'), new Date('2024-12-31')]
  }
  fetchData()
}

onMounted(() => {
  fetchData()

  window.addEventListener('resize', () => {
    pieChartInstance?.resize()
    barChartInstance?.resize()
    lineChartInstance?.resize()
    heatmapChartInstance?.resize()
  })
})
</script>

<style scoped>
.analysis-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.chart-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
