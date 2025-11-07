/**
 * 水质监测可视化平台 - 地图模块
 * 负责地图初始化、站点渲染、空间查询等功能
 */

const API_BASE_URL = 'http://localhost:5000/api';

class WaterQualityApp {
    constructor() {
        this.map = null;
        this.stations = [];
        this.allStations = [];  // 存储所有站点数据
        this.markerClusterGroup = null;
        this.drawnItems = null;
        this.currentIndicator = 'Water_Quality_Index';
        this.currentProvince = null;
        this.dateRange = { start: null, end: null };
        this.timelineRange = { start: null, end: null };  // 时间轴的实际范围
        this.currentTimelineDate = null;  // 当前时间轴选中的日期
        this.trendChart = null;
    }

    /**
     * 初始化应用
     */
    async init() {
        console.log('🗺️ Initializing map...');
        this.initMap();
        this.initDrawControls();
        this.initEventListeners();

        await this.loadStations();
        await this.loadStatistics();
        this.updateProvinceFilter();
        this.initDateInputs();

        this.hideLoading();
        console.log('✅ Application initialized successfully');
    }

    /**
     * 初始化地图
     */
    initMap() {
        this.map = L.map('map', {
            center: [35.8617, 104.1954],
            zoom: 5,
            zoomControl: true
        });

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }).addTo(this.map);

        // 初始化聚类图层
        this.markerClusterGroup = L.markerClusterGroup({
            maxClusterRadius: 60,
            spiderfyOnMaxZoom: true,
            showCoverageOnHover: false,
            zoomToBoundsOnClick: true
        });
        this.map.addLayer(this.markerClusterGroup);
    }

    /**
     * 初始化绘制控件（矩形和圆形选择）
     */
    initDrawControls() {
        this.drawnItems = new L.FeatureGroup();
        this.map.addLayer(this.drawnItems);

        this.drawControl = new L.Control.Draw({
            position: 'topleft',
            draw: {
                polyline: false,
                polygon: false,
                circle: true,
                rectangle: true,
                marker: false,
                circlemarker: false
            },
            edit: {
                featureGroup: this.drawnItems,
                remove: true
            }
        });

        // 监听绘制事件
        this.map.on(L.Draw.Event.CREATED, (e) => {
            this.drawnItems.clearLayers();
            this.drawnItems.addLayer(e.layer);
            this.handleSpatialQuery(e);
        });
    }

    /**
     * 初始化事件监听器
     */
    initEventListeners() {
        // 时间轴滑块 - input事件用于实时更新显示
        const slider = document.getElementById('timelineSlider');
        if (slider) {
            console.log('🎚️ Timeline slider found, binding events');
            slider.addEventListener('input', (e) => {
                console.log('Timeline slider input:', e.target.value);
                this.updateTimelineSlider(e.target.value);
            });

            // change事件用于实际过滤数据（减少API调用）
            slider.addEventListener('change', (e) => {
                console.log('Timeline slider change:', e.target.value);
                this.applyTimelineFilter(e.target.value);
            });
        } else {
            console.warn('⚠️ Timeline slider not found!');
        }
    }

    /**
     * 初始化日期输入框
     */
    initDateInputs() {
        const today = new Date();
        const oneYearAgo = new Date();
        oneYearAgo.setFullYear(today.getFullYear() - 1);

        document.getElementById('startDate').value = this.formatDate(oneYearAgo);
        document.getElementById('endDate').value = this.formatDate(today);
    }

    /**
     * 加载所有监测站点
     */
    async loadStations() {
        try {
            this.showLoading();
            const response = await fetch(`${API_BASE_URL}/stations`);
            const result = await response.json();

            if (result.success) {
                this.allStations = result.data;
                this.stations = result.data;
                this.renderStations();
                await this.initTimelineRange();  // 初始化时间轴范围

                // 加载趋势图表
                await this.loadTrends('Water_Quality_Index', null, 'month');

                console.log(`✅ Loaded ${this.stations.length} stations`);
            }
        } catch (error) {
            console.error('❌ Error loading stations:', error);
            this.showError('加载监测站点失败');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * 渲染站点到地图
     */
    renderStations() {
        this.markerClusterGroup.clearLayers();

        this.stations.forEach(station => {
            const marker = this.createStationMarker(station);
            this.markerClusterGroup.addLayer(marker);
        });
    }

    /**
     * 创建站点标记
     */
    createStationMarker(station) {
        const color = this.getPollutionColor(station.pollution_level);
        const radius = this.getMarkerRadius(station.latest_wqi);

        const marker = L.circleMarker([station.coordinates[0], station.coordinates[1]], {
            radius: radius,
            fillColor: color,
            color: 'white',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        });

        // 弹出窗口
        const popupContent = `
            <div class="popup-title">${station.name}</div>
            <div class="popup-info"><strong>省份:</strong> ${station.province}</div>
            <div class="popup-info"><strong>水质指数:</strong> ${station.latest_wqi.toFixed(2)}</div>
            <div class="popup-info"><strong>污染等级:</strong> <span style="color: ${color}; font-weight: bold;">${station.pollution_level}</span></div>
            <div class="popup-info"><strong>记录数:</strong> ${station.record_count}</div>
            <div class="popup-info"><strong>最新日期:</strong> ${station.latest_date || 'N/A'}</div>
        `;

        marker.bindPopup(popupContent);

        return marker;
    }

    /**
     * 根据污染等级获取颜色
     */
    getPollutionColor(level) {
        const colorMap = {
            'Excellent': '#28a745',
            'Good': '#5cb85c',
            'Fair': '#ffc107',
            'Poor': '#fd7e14',
            'Very Poor': '#dc3545'
        };
        return colorMap[level] || '#6c757d';
    }

    /**
     * 根据水质指数获取标记大小
     */
    getMarkerRadius(wqi) {
        if (wqi < 20) return 6;
        if (wqi < 40) return 8;
        if (wqi < 60) return 10;
        if (wqi < 80) return 12;
        return 14;
    }

    /**
     * 加载统计信息
     */
    async loadStatistics(province = null) {
        try {
            const url = province
                ? `${API_BASE_URL}/statistics/summary?province=${province}`
                : `${API_BASE_URL}/statistics/summary`;

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.updateStatisticsPanel(result.data);
                this.updateHeaderStats(result.data);
            }
        } catch (error) {
            console.error('❌ Error loading statistics:', error);
        }
    }

    /**
     * 更新头部统计数据
     */
    updateHeaderStats(stats) {
        document.getElementById('totalStations').textContent = stats.total_stations;
        document.getElementById('totalRecords').textContent = stats.total_records.toLocaleString();
        document.getElementById('avgWQI').textContent = stats.avg_wqi.toFixed(1);
    }

    /**
     * 更新统计面板
     */
    updateStatisticsPanel(stats) {
        const content = `
            <div class="stat-row">
                <span class="stat-row-label">总记录数</span>
                <span class="stat-row-value">${stats.total_records.toLocaleString()}</span>
            </div>
            <div class="stat-row">
                <span class="stat-row-label">监测站点</span>
                <span class="stat-row-value">${stats.total_stations}</span>
            </div>
            <div class="stat-row">
                <span class="stat-row-label">平均水质指数</span>
                <span class="stat-row-value">${stats.avg_wqi.toFixed(2)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-row-label">平均COD</span>
                <span class="stat-row-value">${stats.indicators.cod.mean.toFixed(2)} mg/L</span>
            </div>
            <div class="stat-row">
                <span class="stat-row-label">平均pH</span>
                <span class="stat-row-value">${stats.indicators.ph.mean.toFixed(2)}</span>
            </div>
            <hr style="margin: 15px 0; border: none; border-top: 1px solid #e2e8f0;">
            <h4 style="font-size: 13px; margin-bottom: 10px;">污染等级分布</h4>
            ${this.renderPollutionDistribution(stats.pollution_distribution)}
        `;

        document.getElementById('statisticsContent').innerHTML = content;
    }

    /**
     * 渲染污染等级分布
     */
    renderPollutionDistribution(distribution) {
        let html = '';
        const total = Object.values(distribution).reduce((a, b) => a + b, 0);

        for (const [level, count] of Object.entries(distribution)) {
            const percentage = ((count / total) * 100).toFixed(1);
            const color = this.getPollutionColor(level);

            html += `
                <div class="stat-row">
                    <span style="color: ${color};">● ${level}</span>
                    <span class="stat-row-value">${percentage}%</span>
                </div>
            `;
        }

        return html;
    }

    /**
     * 更新省份过滤器
     */
    updateProvinceFilter() {
        const provinces = [...new Set(this.stations.map(s => s.province))].sort();
        const select = document.getElementById('provinceFilter');

        provinces.forEach(province => {
            const option = document.createElement('option');
            option.value = province;
            option.textContent = province;
            select.appendChild(option);
        });
    }

    /**
     * 处理空间查询
     */
    async handleSpatialQuery(e) {
        const layer = e.layer;
        const type = e.layerType;

        let query = { start_date: this.dateRange.start, end_date: this.dateRange.end };

        if (type === 'rectangle') {
            const bounds = layer.getBounds();
            query.type = 'rectangle';
            query.bounds = [
                [bounds.getSouthWest().lat, bounds.getSouthWest().lng],
                [bounds.getNorthEast().lat, bounds.getNorthEast().lng]
            ];
        } else if (type === 'circle') {
            const center = layer.getLatLng();
            const radius = layer.getRadius();
            query.type = 'circle';
            query.center = [center.lat, center.lng];
            query.radius = radius;
        }

        try {
            this.showLoading();
            const response = await fetch(`${API_BASE_URL}/data/spatial`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(query)
            });

            const result = await response.json();

            if (result.success) {
                this.displaySpatialQueryResult(result.data);
            }
        } catch (error) {
            console.error('❌ Spatial query error:', error);
            this.showError('空间查询失败');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * 显示空间查询结果
     */
    displaySpatialQueryResult(data) {
        const content = `
            <div class="info-item"><strong>记录数:</strong> ${data.record_count}</div>
            <div class="info-item"><strong>站点数:</strong> ${data.statistics.station_count || 0}</div>
            <div class="info-item"><strong>平均WQI:</strong> ${(data.statistics.avg_wqi || 0).toFixed(2)}</div>
            <div class="info-item"><strong>平均COD:</strong> ${(data.statistics.avg_cod || 0).toFixed(2)} mg/L</div>
            <div class="info-item"><strong>平均pH:</strong> ${(data.statistics.avg_ph || 0).toFixed(2)}</div>
        `;

        document.getElementById('selectedAreaInfo').innerHTML = content;
    }

    /**
     * 省份过滤
     */
    async filterByProvince() {
        const province = document.getElementById('provinceFilter').value;
        this.currentProvince = province || null;
        await this.loadStatistics(this.currentProvince);
    }

    /**
     * 应用时间过滤
     */
    applyTimeFilter() {
        this.dateRange.start = document.getElementById('startDate').value;
        this.dateRange.end = document.getElementById('endDate').value;
        console.log('📅 Time filter applied:', this.dateRange);
    }

    /**
     * 启用矩形选择
     */
    enableRectangleSelect() {
        new L.Draw.Rectangle(this.map, this.drawControl.options.draw.rectangle).enable();
    }

    /**
     * 启用圆形选择
     */
    enableCircleSelect() {
        new L.Draw.Circle(this.map, this.drawControl.options.draw.circle).enable();
    }

    /**
     * 清除选择
     */
    clearSelection() {
        this.drawnItems.clearLayers();
        document.getElementById('selectedAreaInfo').innerHTML = '<p class="info-text">请使用空间查询工具选择区域</p>';
    }

    /**
     * 切换预测标签页
     */
    switchPredictionTab(tab) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.prediction-form').forEach(form => form.classList.remove('active'));

        event.target.classList.add('active');
        document.getElementById(`predictionForm${tab === 'level' ? 'Level' : 'Quality'}`).classList.add('active');
    }

    /**
     * 改变指标
     */
    changeIndicator() {
        this.currentIndicator = document.getElementById('indicatorSelect').value;
        console.log('📊 Indicator changed to:', this.currentIndicator);
    }

    /**
     * 初始化时间轴范围（基于实际数据）
     */
    async initTimelineRange() {
        try {
            console.log('📅 Initializing timeline range...');
            const response = await fetch(`${API_BASE_URL}/data/date-range`);
            const result = await response.json();

            console.log('Timeline API response:', result);

            if (result.success && result.data) {
                this.timelineRange.start = new Date(result.data.min_date);
                this.timelineRange.end = new Date(result.data.max_date);
                this.currentTimelineDate = this.timelineRange.end;  // 默认显示最新数据

                console.log('Timeline range:', {
                    start: this.timelineRange.start,
                    end: this.timelineRange.end,
                    startFormatted: this.formatDisplayDate(this.timelineRange.start),
                    endFormatted: this.formatDisplayDate(this.timelineRange.end)
                });

                // 更新UI
                const startElem = document.getElementById('timelineStart');
                const endElem = document.getElementById('timelineEnd');
                const currentElem = document.getElementById('timelineCurrent');

                if (startElem) startElem.textContent = this.formatDisplayDate(this.timelineRange.start);
                if (endElem) endElem.textContent = this.formatDisplayDate(this.timelineRange.end);
                if (currentElem) currentElem.textContent = this.formatDisplayDate(this.currentTimelineDate);

                // 设置滑块初始值为最大值（最新日期）
                const slider = document.getElementById('timelineSlider');
                if (slider) slider.value = 100;

                console.log(`✅ Timeline initialized: ${result.data.min_date} to ${result.data.max_date}`);
            } else {
                console.error('❌ Timeline API returned unsuccessful or no data:', result);
            }
        } catch (error) {
            console.error('❌ Error initializing timeline:', error);
        }
    }

    /**
     * 更新时间轴滑块显示（拖动时实时更新，不过滤数据）
     */
    updateTimelineSlider(sliderValue) {
        console.log('Updating timeline slider, value:', sliderValue);
        console.log('Timeline range:', this.timelineRange);

        if (!this.timelineRange.start || !this.timelineRange.end) {
            console.warn('Timeline range not initialized');
            return;
        }

        const percentage = sliderValue / 100;
        const totalMs = this.timelineRange.end - this.timelineRange.start;
        const selectedMs = totalMs * percentage;
        const selectedDate = new Date(this.timelineRange.start.getTime() + selectedMs);

        console.log('Calculated date:', this.formatDisplayDate(selectedDate));

        // 只更新显示，不过滤数据
        const currentElem = document.getElementById('timelineCurrent');
        if (currentElem) {
            currentElem.textContent = this.formatDisplayDate(selectedDate);
        } else {
            console.warn('Timeline current element not found');
        }
    }

    /**
     * 应用时间轴过滤（松开滑块时执行）
     */
    async applyTimelineFilter(sliderValue) {
        if (!this.timelineRange.start || !this.timelineRange.end) return;

        const percentage = sliderValue / 100;
        const totalMs = this.timelineRange.end - this.timelineRange.start;
        const selectedMs = totalMs * percentage;
        this.currentTimelineDate = new Date(this.timelineRange.start.getTime() + selectedMs);

        try {
            this.showLoading();

            // 调用后端API获取截止到该日期的数据
            const dateStr = this.formatDate(this.currentTimelineDate);
            const response = await fetch(`${API_BASE_URL}/data/by-date?date=${dateStr}`);
            const result = await response.json();

            if (result.success) {
                // 更新站点数据
                this.stations = result.data.stations || [];
                this.renderStations();

                // 更新统计信息
                if (result.data.statistics) {
                    this.updateHeaderStats(result.data.statistics);
                }

                console.log(`✅ Filtered to date: ${dateStr}, showing ${this.stations.length} stations`);
            }
        } catch (error) {
            console.error('❌ Error applying timeline filter:', error);
            this.showError('时间轴过滤失败');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * 格式化显示日期
     */
    formatDisplayDate(date) {
        if (!date) return '-';
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }

    // 工具函数
    formatDate(date) {
        return date.toISOString().split('T')[0];
    }

    showLoading() {
        document.getElementById('loadingOverlay').classList.add('show');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.remove('show');
    }

    showError(message) {
        alert(message);
    }
}
