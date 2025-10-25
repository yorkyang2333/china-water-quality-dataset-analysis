/**
 * 时间轴模块
 * 处理时间线数据加载和趋势图表显示
 */

// 扩展WaterQualityApp类，添加时间轴功能
if (typeof WaterQualityApp !== 'undefined') {
    /**
     * 加载时间轴数据
     */
    WaterQualityApp.prototype.loadTimelineData = async function(startDate, endDate, station = null, province = null) {
        try {
            let url = `${API_BASE_URL}/data/timeline?`;
            if (startDate) url += `start_date=${startDate}&`;
            if (endDate) url += `end_date=${endDate}&`;
            if (station) url += `station=${station}&`;
            if (province) url += `province=${province}`;

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.updateTimelineDisplay(result.data);
                return result.data;
            }
        } catch (error) {
            console.error('❌ Error loading timeline data:', error);
            return [];
        }
    };

    /**
     * 更新时间轴显示
     */
    WaterQualityApp.prototype.updateTimelineDisplay = function(timelineData) {
        if (timelineData.length === 0) return;

        const dates = timelineData.map(d => d.Sampling_Date);
        const startDate = dates[0];
        const endDate = dates[dates.length - 1];

        document.getElementById('timelineStart').textContent = startDate;
        document.getElementById('timelineEnd').textContent = endDate;
        document.getElementById('timelineCurrent').textContent = endDate;

        console.log(`📅 Timeline updated: ${startDate} to ${endDate}`);
    };

    /**
     * 加载趋势数据
     */
    WaterQualityApp.prototype.loadTrends = async function(indicator = 'Water_Quality_Index', province = null, interval = 'month') {
        try {
            let url = `${API_BASE_URL}/trends?indicator=${indicator}&interval=${interval}`;
            if (province) url += `&province=${province}`;

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.renderTrendChart(result.data, indicator);
                return result.data;
            }
        } catch (error) {
            console.error('❌ Error loading trends:', error);
            return [];
        }
    };

    /**
     * 渲染趋势图表
     */
    WaterQualityApp.prototype.renderTrendChart = function(trendData, indicator) {
        const canvas = document.getElementById('trendChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // 销毁旧图表
        if (this.trendChart) {
            this.trendChart.destroy();
        }

        const labels = trendData.map(d => d.period);
        const values = trendData.map(d => d.value);

        // 设置canvas固定尺寸防止伸张
        canvas.style.width = '100%';
        canvas.style.height = '200px';
        canvas.width = canvas.offsetWidth;
        canvas.height = 200;

        this.trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: this.getIndicatorLabel(indicator),
                    data: values,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: false,  // 关键修复：关闭响应式避免伸张
                maintainAspectRatio: false,
                animation: {
                    duration: 0  // 关闭动画避免重绘问题
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: {
                                size: 11,
                                family: 'Poppins'
                            },
                            padding: 10
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleFont: {
                            size: 12,
                            family: 'Poppins'
                        },
                        bodyFont: {
                            size: 11,
                            family: 'Poppins'
                        },
                        padding: 10,
                        cornerRadius: 8
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            font: {
                                size: 10
                            },
                            padding: 5
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)',
                            drawBorder: false
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: 10
                            },
                            maxRotation: 45,
                            minRotation: 0,
                            maxTicksLimit: 8  // 限制X轴标签数量
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });

        console.log('📈 Trend chart rendered with fixed size');
    };

    /**
     * 获取指标标签
     */
    WaterQualityApp.prototype.getIndicatorLabel = function(indicator) {
        const labels = {
            'Water_Quality_Index': '水质指数 (WQI)',
            'COD_mg_L': '化学需氧量 (mg/L)',
            'pH': 'pH值',
            'Total_Nitrogen_mg_L': '总氮 (mg/L)',
            'Total_Phosphorus_mg_L': '总磷 (mg/L)',
            'Dissolved_Oxygen_mg_L': '溶解氧 (mg/L)',
            'Ammonia_N_mg_L': '氨氮 (mg/L)',
            'Turbidity_NTU': '浊度 (NTU)'
        };
        return labels[indicator] || indicator;
    };

    /**
     * 初始化时间轴功能（在init方法后调用）
     */
    WaterQualityApp.prototype.initTimeline = async function() {
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        await this.loadTimelineData(startDate, endDate);
        await this.loadTrends('Water_Quality_Index', null, 'month');
    };

    // 修改原有的init方法，添加时间轴初始化
    const originalInit = WaterQualityApp.prototype.init;
    WaterQualityApp.prototype.init = async function() {
        await originalInit.call(this);
        await this.initTimeline();
    };
}
