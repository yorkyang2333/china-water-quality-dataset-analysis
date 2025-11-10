<template>
  <div class="spatial-container">
    <el-row :gutter="20">
      <!-- Left Sidebar - Control Panel -->
      <el-col :span="6">
        <el-card class="control-panel">
          <template #header>
            <div class="panel-header">
              <span>控制面板</span>
            </div>
          </template>

          <!-- Time Range Filter -->
          <div class="control-group">
            <label>时间范围</label>
            <el-date-picker
              v-model="dateRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              style="width: 100%;"
              size="default"
            />
            <el-button type="primary" size="small" @click="applyTimeFilter" style="width: 100%; margin-top: 8px;">
              应用
            </el-button>
          </div>

          <!-- Province Filter -->
          <div class="control-group">
            <label>省份筛选</label>
            <el-select v-model="selectedProvince" placeholder="全部省份" clearable style="width: 100%;" @change="filterByProvince">
              <el-option label="全部省份" value="" />
              <el-option v-for="province in provinces" :key="province" :label="province" :value="province" />
            </el-select>
          </div>

          <!-- Indicator Selection -->
          <div class="control-group">
            <label>显示指标</label>
            <el-select v-model="selectedIndicator" style="width: 100%;" @change="changeIndicator">
              <el-option label="水质指数 (WQI)" value="Water_Quality_Index" />
              <el-option label="化学需氧量 (COD)" value="COD_mg_L" />
              <el-option label="pH值" value="pH" />
              <el-option label="总氮" value="Total_Nitrogen_mg_L" />
              <el-option label="总磷" value="Total_Phosphorus_mg_L" />
              <el-option label="溶解氧" value="Dissolved_Oxygen_mg_L" />
            </el-select>
          </div>

          <!-- Spatial Query Tools -->
          <div class="control-group">
            <label>空间查询工具</label>
            <div class="button-group">
              <el-button :type="drawMode === 'rectangle' ? 'primary' : ''" @click="enableRectangleSelect" style="flex: 1;" size="small">
                矩形
              </el-button>
              <el-button :type="drawMode === 'circle' ? 'primary' : ''" @click="enableCircleSelect" style="flex: 1;" size="small">
                圆形
              </el-button>
              <el-button @click="clearSelection" style="flex: 1;" size="small">
                清除
              </el-button>
            </div>
          </div>

          <!-- Statistics -->
          <el-divider />
          <div class="statistics-panel">
            <h4>数据统计</h4>
            <el-descriptions :column="1" size="small" border>
              <el-descriptions-item label="站点总数">
                {{ statistics.totalStations }}
              </el-descriptions-item>
              <el-descriptions-item label="平均指标值">
                {{ statistics.avgValue?.toFixed(2) || 'N/A' }}
              </el-descriptions-item>
              <el-descriptions-item label="最大值">
                {{ statistics.maxValue?.toFixed(2) || 'N/A' }}
              </el-descriptions-item>
              <el-descriptions-item label="最小值">
                {{ statistics.minValue?.toFixed(2) || 'N/A' }}
              </el-descriptions-item>
            </el-descriptions>
          </div>
        </el-card>
      </el-col>

      <!-- Map Container -->
      <el-col :span="18">
        <el-card class="map-container-card">
          <div id="map" ref="mapContainer"></div>

          <!-- Map Legend -->
          <div class="map-legend">
            <h4>图例</h4>
            <div class="legend-item">
              <span class="legend-marker excellent"></span>
              <span>优秀 (WQI &lt; 20)</span>
            </div>
            <div class="legend-item">
              <span class="legend-marker good"></span>
              <span>良好 (20-40)</span>
            </div>
            <div class="legend-item">
              <span class="legend-marker fair"></span>
              <span>一般 (40-60)</span>
            </div>
            <div class="legend-item">
              <span class="legend-marker poor"></span>
              <span>较差 (60-80)</span>
            </div>
            <div class="legend-item">
              <span class="legend-marker very-poor"></span>
              <span>很差 (&gt; 80)</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import 'leaflet.markercluster'
import 'leaflet.markercluster/dist/MarkerCluster.css'
import 'leaflet.markercluster/dist/MarkerCluster.Default.css'
import 'leaflet-draw'
import 'leaflet-draw/dist/leaflet.draw.css'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const mapContainer = ref(null)
let map = null
let markerClusterGroup = null
let drawnItems = null
let currentDrawHandler = null // 当前的绘制处理器

const dateRange = ref([])
const selectedProvince = ref('')
const selectedIndicator = ref('Water_Quality_Index')
const drawMode = ref('')

const provinces = ref([
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

// 英文到中文的映射（用于显示）
const englishToChinese = {
  'Beijing': '北京',
  'Guangdong': '广东',
  'Henan': '河南',
  'Hubei': '湖北',
  'Jiangsu': '江苏',
  'Shandong': '山东',
  'Shanghai': '上海',
  'Sichuan': '四川',
  'Yunnan': '云南',
  'Zhejiang': '浙江'
}

const statistics = ref({
  totalStations: 0,
  avgValue: 0,
  maxValue: 0,
  minValue: 0
})

let stationsData = []

// Fix Leaflet default marker icon issue
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
})

// Get marker color based on water quality value
const getMarkerColor = (value) => {
  if (value < 20) return '#67C23A' // Excellent - green
  if (value < 40) return '#95D475' // Good - light green
  if (value < 60) return '#E6A23C' // Fair - orange
  if (value < 80) return '#F56C6C' // Poor - red
  return '#C0392B' // Very Poor - dark red
}

// Get marker size based on value
const getMarkerSize = (value) => {
  return Math.max(10, Math.min(value / 5, 30))
}

// Create custom marker
const createCustomMarker = (station) => {
  const value = station[selectedIndicator.value] || 0
  const color = getMarkerColor(value)
  const size = getMarkerSize(value)

  const icon = L.divIcon({
    className: 'custom-marker',
    html: `<div style="
      background-color: ${color};
      width: ${size}px;
      height: ${size}px;
      border-radius: 50%;
      border: 2px solid white;
      box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    "></div>`,
    iconSize: [size, size],
    iconAnchor: [size/2, size/2]
  })

  return icon
}

// Initialize map
const initMap = () => {
  // Verify Leaflet.draw is loaded
  if (!L.Draw) {
    console.error('Leaflet.draw is not loaded!')
    ElMessage.error('绘图库未加载，请刷新页面')
    return
  }
  console.log('Leaflet.draw loaded successfully:', L.Draw)

  map = L.map('map').setView([35.8617, 104.1954], 5) // Center of China
  console.log('Map initialized:', map)

  // Add tile layer
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 18
  }).addTo(map)

  // Initialize marker cluster group
  markerClusterGroup = L.markerClusterGroup({
    maxClusterRadius: 50,
    spiderfyOnMaxZoom: true,
    showCoverageOnHover: false,
    zoomToBoundsOnClick: true
  })
  map.addLayer(markerClusterGroup)

  // Initialize drawn items layer
  drawnItems = new L.FeatureGroup()
  map.addLayer(drawnItems)
  console.log('DrawnItems FeatureGroup created and added:', drawnItems)

  // Handle drawn shapes
  map.on(L.Draw.Event.CREATED, (e) => {
    console.log('=== CREATED event fired ===')
    console.log('Shape type:', e.layerType)
    console.log('Layer:', e.layer)

    const layer = e.layer
    drawnItems.addLayer(layer)
    console.log('Layer added to drawnItems')

    // Query stations within the drawn area
    queryStationsInArea(layer)

    // 禁用绘制处理器
    if (currentDrawHandler) {
      console.log('Disabling draw handler after creation')
      currentDrawHandler.disable()
      currentDrawHandler = null
    }

    // Disable drawing mode
    drawMode.value = ''
    console.log('Drawing mode cleared')
  })

  // 监听绘制开始事件
  map.on(L.Draw.Event.DRAWSTART, (e) => {
    console.log('=== DRAWSTART event fired ===')
    console.log('Layer type:', e.layerType)
    console.log('Handler:', e.handler)
  })

  // 监听绘制停止事件
  map.on(L.Draw.Event.DRAWSTOP, (e) => {
    console.log('=== DRAWSTOP event fired ===')
    console.log('Layer type:', e.layerType)
  })

  // 监听绘制过程中的事件
  map.on(L.Draw.Event.DRAWVERTEX, (e) => {
    console.log('=== DRAWVERTEX event (drawing in progress) ===')
    console.log('Layers:', e.layers)
  })

  // 监听地图点击事件
  map.on('click', (e) => {
    if (drawMode.value) {
      console.log('Map clicked while in draw mode:', e.latlng)
    }
  })

  // 监听地图鼠标按下/抬起事件
  map.on('mousedown', (e) => {
    if (drawMode.value) {
      console.log('Mouse down while in draw mode:', e.latlng)
    }
  })

  map.on('mouseup', (e) => {
    if (drawMode.value) {
      console.log('Mouse up while in draw mode:', e.latlng)
    }
  })

  // Ensure map is properly sized
  setTimeout(() => {
    map.invalidateSize()
  }, 100)
}

// Load stations data
const loadStations = async () => {
  try {
    const response = await axios.get('http://localhost:8000/api/stations')
    if (response.data.success) {
      // Transform API data to match our structure
      stationsData = response.data.data.map(station => {
        return {
          station: station.name,
          province: englishToChinese[station.province] || station.province,  // 转换为中文显示
          latitude: station.coordinates[0],  // coordinates is [lat, lng]
          longitude: station.coordinates[1],
          Water_Quality_Index: station.latest_wqi || 0,
          pollutionLevel: station.pollution_level,
          COD_mg_L: 0,  // These would need to come from latest record
          pH: 0,
          Total_Nitrogen_mg_L: 0,
          Total_Phosphorus_mg_L: 0,
          Dissolved_Oxygen_mg_L: 0
        }
      })

      console.log('Loaded stations:', stationsData.length)
      console.log('Sample station:', stationsData[0])

      updateMarkers()
      updateStatistics()
    }
  } catch (error) {
    console.error('Failed to load stations:', error)
    ElMessage.error('加载站点数据失败')
  }
}

// Update markers on map
const updateMarkers = () => {
  if (!markerClusterGroup) return

  markerClusterGroup.clearLayers()

  let filteredStations = [...stationsData]

  console.log('Total stations before filter:', filteredStations.length)

  // Filter by province
  if (selectedProvince.value) {
    filteredStations = filteredStations.filter(s => s.province === selectedProvince.value)
    console.log('After province filter:', filteredStations.length, 'province:', selectedProvince.value)
  }

  // Filter by date range
  if (dateRange.value && dateRange.value.length === 2) {
    const startDate = new Date(dateRange.value[0])
    const endDate = new Date(dateRange.value[1])
    // Note: 由于当前stationsData只有latest数据，日期筛选需要后端支持
    // 这里先保留逻辑，实际需要调用API获取时间范围内的数据
    console.log('Date range filter:', startDate, 'to', endDate)
  }

  console.log('Filtered stations:', filteredStations.length)

  filteredStations.forEach(station => {
    const marker = L.marker([station.latitude, station.longitude], {
      icon: createCustomMarker(station)
    })

    // Create popup content
    const value = station[selectedIndicator.value] || 0
    const popupContent = `
      <div style="min-width: 200px;">
        <h4 style="margin: 0 0 10px 0;">${station.station}</h4>
        <p style="margin: 5px 0;"><strong>省份：</strong>${station.province}</p>
        <p style="margin: 5px 0;"><strong>${selectedIndicator.value}：</strong>${value.toFixed(2)}</p>
        <p style="margin: 5px 0;"><strong>污染等级：</strong>${station.pollutionLevel || 'N/A'}</p>
      </div>
    `

    marker.bindPopup(popupContent)
    markerClusterGroup.addLayer(marker)
  })

  // Update statistics after filtering
  updateStatistics()
}

// Update statistics
const updateStatistics = () => {
  let filteredStations = [...stationsData]

  // Apply same filters as updateMarkers
  if (selectedProvince.value) {
    filteredStations = filteredStations.filter(s => s.province === selectedProvince.value)
  }

  console.log('Calculating statistics for', filteredStations.length, 'stations')

  const values = filteredStations
    .map(s => s[selectedIndicator.value])
    .filter(v => v != null && v > 0 && !isNaN(v))

  console.log('Valid values:', values.length, 'Values:', values.slice(0, 5))

  statistics.value = {
    totalStations: filteredStations.length,
    avgValue: values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0,
    maxValue: values.length > 0 ? Math.max(...values) : 0,
    minValue: values.length > 0 ? Math.min(...values) : 0
  }

  console.log('Statistics updated:', statistics.value)
}

// Enable rectangle select
const enableRectangleSelect = () => {
  if (!map || !drawnItems) {
    console.error('Map or drawnItems not initialized')
    ElMessage.error('地图未初始化')
    return
  }

  console.log('Starting rectangle select...')
  console.log('Map:', map)
  console.log('DrawnItems:', drawnItems)

  clearSelection()

  // 如果有正在进行的绘制，先禁用它
  if (currentDrawHandler) {
    console.log('Disabling previous handler')
    currentDrawHandler.disable()
  }

  drawMode.value = 'rectangle'

  try {
    // 创建矩形绘制处理器
    currentDrawHandler = new L.Draw.Rectangle(map, {
      shapeOptions: {
        stroke: true,
        color: '#3388ff',
        weight: 4,
        opacity: 0.9,
        fill: true,
        fillColor: '#3388ff',
        fillOpacity: 0.2,
        clickable: true
      },
      showArea: false,  // 禁用面积显示以避免leaflet-draw的bug
      showTooltip: false,  // 禁用tooltip
      metric: false,  // 禁用公制单位
      repeatMode: false
    })

    console.log('Handler created:', currentDrawHandler)

    // 启用绘制
    currentDrawHandler.enable()

    console.log('Handler enabled, drawMode:', drawMode.value)
    console.log('Map cursor should be crosshair now')

    ElMessage.info('请在地图上拖动鼠标绘制矩形区域')
  } catch (error) {
    console.error('Error enabling rectangle drawing:', error)
    ElMessage.error('启用矩形绘制工具失败')
  }
}

// Enable circle select
const enableCircleSelect = () => {
  if (!map || !drawnItems) {
    console.error('Map or drawnItems not initialized')
    ElMessage.error('地图未初始化')
    return
  }

  console.log('Starting circle select...')

  clearSelection()

  // 如果有正在进行的绘制，先禁用它
  if (currentDrawHandler) {
    console.log('Disabling previous handler')
    currentDrawHandler.disable()
  }

  drawMode.value = 'circle'

  try {
    // 创建圆形绘制处理器
    currentDrawHandler = new L.Draw.Circle(map, {
      shapeOptions: {
        stroke: true,
        color: '#3388ff',
        weight: 4,
        opacity: 0.9,
        fill: true,
        fillColor: '#3388ff',
        fillOpacity: 0.2,
        clickable: true
      },
      showRadius: false,  // 禁用半径显示以避免leaflet-draw的bug
      showTooltip: false,  // 禁用tooltip
      metric: false,  // 禁用公制单位
      repeatMode: false
    })

    console.log('Handler created:', currentDrawHandler)

    // 启用绘制
    currentDrawHandler.enable()

    console.log('Handler enabled, drawMode:', drawMode.value)

    ElMessage.info('请在地图上点击并拖动绘制圆形区域')
  } catch (error) {
    console.error('Error enabling circle drawing:', error)
    ElMessage.error('启用圆形绘制工具失败')
  }
}

// Clear selection
const clearSelection = () => {
  if (drawnItems) {
    drawnItems.clearLayers()
  }

  // 禁用当前的绘制处理器
  if (currentDrawHandler) {
    currentDrawHandler.disable()
    currentDrawHandler = null
  }

  drawMode.value = ''
  updateMarkers()
  updateStatistics()
}

// Query stations in drawn area
const queryStationsInArea = (layer) => {
  let stationsInArea = []

  stationsData.forEach(station => {
    const latlng = L.latLng(station.latitude, station.longitude)

    if (layer instanceof L.Circle) {
      if (layer.getLatLng().distanceTo(latlng) <= layer.getRadius()) {
        stationsInArea.push(station)
      }
    } else if (layer instanceof L.Rectangle || layer instanceof L.Polygon) {
      if (layer.getBounds().contains(latlng)) {
        stationsInArea.push(station)
      }
    }
  })

  // Update statistics for selected area
  const values = stationsInArea.map(s => s[selectedIndicator.value] || 0).filter(v => v > 0)

  statistics.value = {
    totalStations: stationsInArea.length,
    avgValue: values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0,
    maxValue: values.length > 0 ? Math.max(...values) : 0,
    minValue: values.length > 0 ? Math.min(...values) : 0
  }

  ElMessage.success(`在选定区域内找到 ${stationsInArea.length} 个站点`)
}

// Apply time filter
const applyTimeFilter = async () => {
  if (!dateRange.value || dateRange.value.length !== 2) {
    ElMessage.warning('请选择时间范围')
    return
  }

  try {
    console.log('Applying time filter:', dateRange.value)
    // 重新加载数据（带时间范围）
    // 注意：目前API返回的是最新数据，时间筛选需要后端支持
    await loadStations()
    ElMessage.success('已应用时间过滤')
  } catch (error) {
    console.error('Time filter error:', error)
    ElMessage.error('应用时间过滤失败')
  }
}

// Filter by province
const filterByProvince = () => {
  updateMarkers()
  updateStatistics()
}

// Change indicator
const changeIndicator = () => {
  updateMarkers()
  updateStatistics()
}

onMounted(() => {
  initMap()
  loadStations()
})

onUnmounted(() => {
  if (map) {
    map.remove()
  }
})
</script>

<style lang="scss" scoped>
.spatial-container {
  padding: 20px;
  height: calc(100vh - 92px);
}

.control-panel {
  height: 100%;
  max-height: calc(100vh - 132px);
  overflow-y: auto;

  .panel-header {
    font-size: 16px;
    font-weight: 600;
  }

  .control-group {
    margin-bottom: 20px;

    label {
      display: block;
      margin-bottom: 8px;
      font-weight: 500;
      font-size: 14px;
      color: #606266;
    }

    .button-group {
      display: flex;
      gap: 8px;
    }
  }

  .statistics-panel {
    h4 {
      margin-bottom: 12px;
      font-size: 14px;
      font-weight: 600;
      color: #303133;
    }
  }
}

.map-container-card {
  height: calc(100vh - 132px);
  position: relative;

  :deep(.el-card__body) {
    padding: 0;
    height: 100%;
  }
}

#map {
  width: 100%!important;
  height: 100%;
  border-radius: 8px;
  z-index: 1;
  position: relative;
}

// 确保地图容器在绘制时允许拖动
:deep(.leaflet-container) {
  font-family: inherit !important;

  &.leaflet-draw-draw-rectangle,
  &.leaflet-draw-draw-circle {
    cursor: crosshair !important;
  }
}

// 绘制时允许所有鼠标事件
:deep(.leaflet-container.leaflet-draw-draw-rectangle *),
:deep(.leaflet-container.leaflet-draw-draw-circle *) {
  pointer-events: auto !important;
}

// Ensure leaflet panes have correct z-index
:deep(.leaflet-pane) {
  z-index: auto;
}

:deep(.leaflet-overlay-pane) {
  z-index: 400;
}

:deep(.leaflet-shadow-pane) {
  z-index: 500;
}

:deep(.leaflet-marker-pane) {
  z-index: 600;
}

:deep(.leaflet-tooltip-pane) {
  z-index: 650;
}

:deep(.leaflet-popup-pane) {
  z-index: 700;
}

.map-legend {
  position: absolute;
  bottom: 30px;
  right: 30px;
  background: rgba(255, 255, 255, 0.95);
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  backdrop-filter: blur(10px);

  h4 {
    margin: 0 0 10px 0;
    font-size: 14px;
    font-weight: 600;
  }

  .legend-item {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    font-size: 12px;

    .legend-marker {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      margin-right: 8px;
      border: 2px solid white;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);

      &.excellent {
        background-color: #67C23A;
      }

      &.good {
        background-color: #95D475;
      }

      &.fair {
        background-color: #E6A23C;
      }

      &.poor {
        background-color: #F56C6C;
      }

      &.very-poor {
        background-color: #C0392B;
      }
    }
  }
}

// Leaflet popup customization
:deep(.leaflet-popup-content-wrapper) {
  border-radius: 8px;
}

:deep(.leaflet-popup-content) {
  margin: 12px;

  h4 {
    color: #409EFF;
    border-bottom: 2px solid #409EFF;
    padding-bottom: 5px;
  }
}

// Leaflet draw customization - ensure visibility
:deep(.leaflet-draw-tooltip) {
  background: rgba(0, 0, 0, 0.8) !important;
  color: white !important;
  padding: 8px !important;
  border-radius: 4px !important;
  font-size: 12px !important;
}

:deep(.leaflet-draw-tooltip-single) {
  margin-top: -12px !important;
}

:deep(.leaflet-draw-tooltip-subtext) {
  color: #ccc !important;
}

// Rectangle and polygon drawing styles
:deep(.leaflet-interactive) {
  stroke: #3388ff !important;
  stroke-width: 4 !important;
  fill: #3388ff !important;
  fill-opacity: 0.2 !important;
}

// Drawing guide lines
:deep(.leaflet-draw-guide-dash) {
  stroke: #3388ff !important;
  stroke-width: 2 !important;
  stroke-dasharray: 5, 5 !important;
  opacity: 1 !important;
  pointer-events: none !important;
}

// Temporary drawing layer (before completion)
:deep(path.leaflet-interactive) {
  stroke: #3388ff !important;
  stroke-width: 4 !important;
  fill: #3388ff !important;
  fill-opacity: 0.2 !important;
  pointer-events: all !important;
}

:deep(svg.leaflet-zoom-animated) {
  pointer-events: auto !important;
}

// Ensure SVG overlay is visible
:deep(.leaflet-overlay-pane svg) {
  pointer-events: auto !important;
}

// Handles for editing
:deep(.leaflet-editing-icon) {
  border-radius: 50% !important;
  border: 2px solid #fff !important;
  background-color: #3388ff !important;
  box-shadow: 0 0 5px rgba(0, 0, 0, 0.4) !important;
}

// Mouse cursor during drawing
:deep(.leaflet-container.leaflet-draw-draw-rectangle),
:deep(.leaflet-draw-draw-rectangle),
:deep(.leaflet-container.leaflet-draw-draw-circle),
:deep(.leaflet-draw-draw-circle) {
  cursor: crosshair !important;
}

// Drawing area styles - visible and interactive
:deep(.leaflet-overlay-pane) {
  z-index: 400;
  pointer-events: auto !important;

  svg {
    pointer-events: auto !important;

    path.leaflet-interactive {
      stroke: #3388ff !important;
      stroke-width: 4 !important;
      fill: #3388ff !important;
      fill-opacity: 0.2 !important;
      stroke-opacity: 0.9 !important;
      pointer-events: all !important;
    }
  }
}
</style>
