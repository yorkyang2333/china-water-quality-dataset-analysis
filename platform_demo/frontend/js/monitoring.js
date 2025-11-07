/**
 * GIS检测功能模块
 * 负责实时监测、告警、数据质量和各种检测图表的展示
 */

class MonitoringSystem {
    constructor(apiBaseUrl) {
        this.API_BASE_URL = apiBaseUrl;
        this.charts = {
            pollution: null,
            comparison: null,
            radar: null,
            multiTrend: null
        };
        this.currentProvince = null;
    }

    /**
     * 初始化监测系统
     */
    async init() {
        console.log('Initializing monitoring system...');
        await this.loadRealtimeMonitoring();
        await this.loadAlerts();
        await this.loadDataQuality();
        await this.loadPollutionDistribution();
        await this.loadProvinceComparison();
    }

    /**
     * 设置省份过滤
     */
    setProvince(province) {
        this.currentProvince = province || null;
        this.refresh();
    }

    /**
     * 刷新所有监测数据
     */
    async refresh() {
        await this.loadRealtimeMonitoring();
        await this.loadAlerts();
        await this.loadDataQuality();
        await this.loadPollutionDistribution();
    }

    /**
     * 加载实时监测指标
     */
    async loadRealtimeMonitoring() {
        try {
            const url = this.currentProvince
                ? `${this.API_BASE_URL}/monitoring/realtime?province=${this.currentProvince}`
                : `${this.API_BASE_URL}/monitoring/realtime`;

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.renderRealtimeMonitoring(result.data);
            }
        } catch (error) {
            console.error('Error loading realtime monitoring:', error);
        }
    }

    /**
     * 渲染实时监测指标
     */
    renderRealtimeMonitoring(data) {
        const container = document.getElementById('realtimeMonitoringContent');
        if (!container) return;

        const statusIcons = {
            good: '✅',
            warning: '⚠️',
            danger: '🚨'
        };

        const statusColors = {
            good: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444'
        };

        const trendIcons = {
            up: '↗️',
            down: '↘️',
            stable: '→'
        };

        let html = `
            <div class="monitoring-header">
                <span class="monitoring-time">更新时间: ${data.timestamp || 'N/A'}</span>
                <span class="monitoring-stats">监测站点: ${data.total_stations} | 记录数: ${data.records_count}</span>
            </div>
            <div class="indicator-grid">
        `;

        data.indicators.forEach(indicator => {
            const statusColor = statusColors[indicator.status];
            const statusIcon = statusIcons[indicator.status];
            const trendIcon = trendIcons[indicator.trend];

            html += `
                <div class="indicator-card" data-status="${indicator.status}">
                    <div class="indicator-header">
                        <span class="indicator-name">${this.formatIndicatorName(indicator.name)}</span>
                        <span class="indicator-status" style="color: ${statusColor}">${statusIcon}</span>
                    </div>
                    <div class="indicator-value">
                        <span class="value-number">${indicator.value}</span>
                        <span class="value-unit">${indicator.unit}</span>
                    </div>
                    <div class="indicator-trend">
                        <span class="trend-icon">${trendIcon}</span>
                        <span class="trend-text" style="color: ${indicator.change_percent > 0 ? '#ef4444' : '#10b981'}">
                            ${Math.abs(indicator.change_percent)}%
                        </span>
                    </div>
                    <div class="indicator-footer">
                        <span class="footer-label">平均值</span>
                        <span class="footer-value">${indicator.all_time_avg} ${indicator.unit}</span>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * 加载告警信息
     */
    async loadAlerts() {
        try {
            const url = this.currentProvince
                ? `${this.API_BASE_URL}/monitoring/alerts?province=${this.currentProvince}`
                : `${this.API_BASE_URL}/monitoring/alerts`;

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.renderAlerts(result.data);
            }
        } catch (error) {
            console.error('Error loading alerts:', error);
        }
    }

    /**
     * 渲染告警信息
     */
    renderAlerts(data) {
        const container = document.getElementById('alertsContent');
        if (!container) return;

        const levelIcons = {
            high: '🚨',
            medium: '⚠️',
            low: 'ℹ️'
        };

        const levelColors = {
            high: '#ef4444',
            medium: '#f59e0b',
            low: '#3b82f6'
        };

        let html = `
            <div class="alerts-summary">
                <div class="alert-stat critical">
                    <span class="stat-number">${data.critical_count}</span>
                    <span class="stat-label">严重告警</span>
                </div>
                <div class="alert-stat warning">
                    <span class="stat-number">${data.warning_count}</span>
                    <span class="stat-label">警告信息</span>
                </div>
                <div class="alert-stat total">
                    <span class="stat-number">${data.total_alerts}</span>
                    <span class="stat-label">总告警数</span>
                </div>
            </div>
            <div class="alerts-list">
        `;

        if (data.alerts.length === 0) {
            html += '<div class="no-alerts">暂无告警信息</div>';
        } else {
            data.alerts.forEach(alert => {
                const levelColor = levelColors[alert.level];
                const levelIcon = levelIcons[alert.level];

                html += `
                    <div class="alert-item" data-level="${alert.level}">
                        <div class="alert-icon" style="color: ${levelColor}">${levelIcon}</div>
                        <div class="alert-content">
                            <div class="alert-message">${alert.message}</div>
                            <div class="alert-details">
                                <span class="alert-station">${alert.station}</span>
                                <span class="alert-date">${alert.date}</span>
                                <span class="alert-value">值: ${alert.value.toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                `;
            });
        }

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * 加载数据质量信息
     */
    async loadDataQuality() {
        try {
            const url = this.currentProvince
                ? `${this.API_BASE_URL}/monitoring/data-quality?province=${this.currentProvince}`
                : `${this.API_BASE_URL}/monitoring/data-quality`;

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.renderDataQuality(result.data);
            }
        } catch (error) {
            console.error('Error loading data quality:', error);
        }
    }

    /**
     * 渲染数据质量信息
     */
    renderDataQuality(data) {
        const container = document.getElementById('dataQualityContent');
        if (!container) return;

        const getScoreColor = (score) => {
            if (score >= 80) return '#10b981';
            if (score >= 60) return '#f59e0b';
            return '#ef4444';
        };

        const scoreColor = getScoreColor(data.quality_score);

        let html = `
            <div class="quality-overview">
                <div class="quality-score-card">
                    <div class="score-circle" style="border-color: ${scoreColor}">
                        <span class="score-value" style="color: ${scoreColor}">${data.quality_score}</span>
                        <span class="score-label">总分</span>
                    </div>
                    <div class="score-details">
                        <div class="score-item">
                            <span class="score-item-label">完整性</span>
                            <span class="score-item-value">${data.completeness}%</span>
                        </div>
                        <div class="score-item">
                            <span class="score-item-label">时效性</span>
                            <span class="score-item-value">${data.timeliness}%</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="quality-stats">
                <div class="quality-stat">
                    <span class="stat-label">总记录数</span>
                    <span class="stat-value">${data.total_records.toLocaleString()}</span>
                </div>
                <div class="quality-stat">
                    <span class="stat-label">监测站点</span>
                    <span class="stat-value">${data.total_stations}</span>
                </div>
                <div class="quality-stat">
                    <span class="stat-label">最后更新</span>
                    <span class="stat-value">${data.latest_update}</span>
                </div>
                <div class="quality-stat">
                    <span class="stat-label">更新间隔</span>
                    <span class="stat-value">${data.avg_sampling_interval} 天</span>
                </div>
            </div>

            <div class="completeness-chart">
                <h4>数据完整性统计</h4>
        `;

        data.missing_data.forEach(item => {
            const completeness = item.completeness;
            const barColor = getScoreColor(completeness);

            html += `
                <div class="completeness-item">
                    <span class="completeness-label">${this.formatIndicatorName(item.indicator)}</span>
                    <div class="completeness-bar">
                        <div class="completeness-fill" style="width: ${completeness}%; background-color: ${barColor}"></div>
                    </div>
                    <span class="completeness-value">${completeness}%</span>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * 加载污染等级分布（饼图）
     */
    async loadPollutionDistribution() {
        try {
            const url = this.currentProvince
                ? `${this.API_BASE_URL}/charts/pollution-distribution?province=${this.currentProvince}`
                : `${this.API_BASE_URL}/charts/pollution-distribution`;

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.renderPollutionDistribution(result.data);
            }
        } catch (error) {
            console.error('Error loading pollution distribution:', error);
        }
    }

    /**
     * 渲染污染等级分布饼图
     */
    renderPollutionDistribution(data) {
        const canvas = document.getElementById('pollutionChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // 销毁旧图表
        if (this.charts.pollution) {
            this.charts.pollution.destroy();
        }

        const labels = data.distribution.map(d => d.level);
        const values = data.distribution.map(d => d.count);
        const colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#7c2d12'];

        this.charts.pollution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
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
                            font: {
                                size: 11
                            },
                            padding: 10
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * 加载省份对比数据（柱状图）
     */
    async loadProvinceComparison(indicator = 'Water_Quality_Index') {
        try {
            const response = await fetch(`${this.API_BASE_URL}/charts/province-comparison?indicator=${indicator}`);
            const result = await response.json();

            if (result.success) {
                this.renderProvinceComparison(result.data);
            }
        } catch (error) {
            console.error('Error loading province comparison:', error);
        }
    }

    /**
     * 渲染省份对比柱状图
     */
    renderProvinceComparison(data) {
        const canvas = document.getElementById('comparisonChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // 销毁旧图表
        if (this.charts.comparison) {
            this.charts.comparison.destroy();
        }

        const labels = data.provinces.map(p => p.province);
        const values = data.provinces.map(p => p.average);

        this.charts.comparison = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: this.formatIndicatorName(data.indicator),
                    data: values,
                    backgroundColor: 'rgba(102, 126, 234, 0.7)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
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
                        },
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                const province = data.provinces[context.dataIndex];
                                return [
                                    `站点数: ${province.station_count}`,
                                    `最小值: ${province.min}`,
                                    `最大值: ${province.max}`,
                                    `标准差: ${province.std}`
                                ];
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * 格式化指标名称
     */
    formatIndicatorName(name) {
        const names = {
            'Water_Quality_Index': '水质指数',
            'COD_mg_L': 'COD',
            'pH': 'pH值',
            'Total_Nitrogen_mg_L': '总氮',
            'Total_Phosphorus_mg_L': '总磷',
            'Dissolved_Oxygen_mg_L': '溶解氧',
            'Ammonia_Nitrogen_mg_L': '氨氮',
            'Turbidity_NTU': '浊度',
            'BOD_mg_L': 'BOD'
        };
        return names[name] || name;
    }

    /**
     * 销毁所有图表
     */
    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
    }
}

// 导出到全局
window.MonitoringSystem = MonitoringSystem;
