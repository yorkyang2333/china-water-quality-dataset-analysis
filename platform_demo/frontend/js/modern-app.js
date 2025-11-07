/**
 * Modern Water Quality Platform Application
 * Handles page routing, interactions, and modern UI functionality
 */

class ModernWaterQualityApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000/api';
        this.currentPage = 'dashboard';
        this.charts = {};
        this.maps = {};
        this.data = {
            stations: [],
            statistics: {},
            realtime: {},
            alerts: [],
            quality: {}
        };
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Modern Water Quality Platform...');
        
        // Initialize navigation
        this.initNavigation();
        
        // Initialize dashboard
        this.initDashboard();
        
        // Initialize page content
        this.initPageContent();
        
        // Initialize forms and interactions
        this.initForms();
        
        // Initialize animations
        this.initAnimations();
        
        // Load initial data
        await this.loadInitialData();
        
        console.log('✅ Modern platform initialized successfully');
    }

    initNavigation() {
        // Handle navigation clicks
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const page = item.getAttribute('data-page');
                this.navigateToPage(page);
            });
        });

        // Handle action card clicks
        document.querySelectorAll('.action-card').forEach(card => {
            card.addEventListener('click', () => {
                const page = card.getAttribute('data-page');
                if (page) {
                    this.navigateToPage(page);
                }
            });
        });

        // Handle hash changes
        window.addEventListener('hashchange', () => {
            const hash = window.location.hash.substring(1);
            if (hash) {
                this.navigateToPage(hash);
            }
        });
    }

    navigateToPage(page) {
        if (page === this.currentPage) return;

        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-page="${page}"]`)?.classList.add('active');

        // Update page content
        document.querySelectorAll('.page-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${page}-page`)?.classList.add('active');

        // Update current page
        this.currentPage = page;

        // Update URL hash
        window.location.hash = page;

        // Initialize page-specific content
        this.initPageSpecificContent(page);

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    initPageSpecificContent(page) {
        switch (page) {
            case 'dashboard':
                this.initDashboardCharts();
                this.animateStatNumbers();
                break;
            case 'monitoring':
                this.initMonitoringPage();
                break;
            case 'prediction':
                this.initPredictionPage();
                break;
            case 'analysis':
                this.initAnalysisPage();
                break;
            case 'quality':
                this.initQualityPage();
                break;
            case 'spatial':
                this.initSpatialPage();
                break;
        }
    }

    async initDashboard() {
        // Initialize dashboard charts
        this.initDashboardCharts();
        
        // Set up quick action handlers
        this.setupQuickActions();
    }

    initDashboardCharts() {
        // Water Level Chart
        this.createWaterLevelChart();
        
        // Pollution Distribution Chart
        this.createPollutionDistributionChart();
    }

    createWaterLevelChart() {
        const ctx = document.getElementById('waterLevelChart');
        if (!ctx) return;

        this.charts.waterLevel = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
                datasets: [{
                    label: '平均WQI',
                    data: [45, 43, 42, 41, 43, 42.3],
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            display: false
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    createPollutionDistributionChart() {
        const ctx = document.getElementById('pollutionDistChart');
        if (!ctx) return;

        this.charts.pollutionDist = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['优秀', '良好', '一般', '较差', '很差'],
                datasets: [{
                    data: [67, 20, 8, 3, 2],
                    backgroundColor: [
                        '#22c55e',
                        '#10b981',
                        '#f59e0b',
                        '#f97316',
                        '#ef4444'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    animateStatNumbers() {
        const statNumbers = document.querySelectorAll('.stat-number');
        
        statNumbers.forEach(stat => {
            const target = parseFloat(stat.getAttribute('data-count'));
            const duration = 2000;
            const increment = target / (duration / 16);
            let current = 0;
            
            const updateNumber = () => {
                current += increment;
                if (current < target) {
                    stat.textContent = Math.floor(current).toLocaleString();
                    requestAnimationFrame(updateNumber);
                } else {
                    stat.textContent = target % 1 === 0 ? 
                        target.toLocaleString() : 
                        target.toFixed(1);
                }
            };
            
            updateNumber();
        });
    }

    setupQuickActions() {
        // Quick action handlers are already set up in navigation
    }

    async initMonitoringPage() {
        // Initialize monitoring map
        await this.initMonitoringMap();
        
        // Load real-time indicators
        await this.loadRealtimeIndicators();
        
        // Initialize monitoring charts
        this.initMonitoringCharts();
    }

    async initMonitoringMap() {
        const mapContainer = document.getElementById('monitoringMap');
        if (!mapContainer || this.maps.monitoring) return;

        // Initialize Leaflet map
        this.maps.monitoring = L.map('monitoringMap').setView([35.8617, 104.1954], 5);

        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.maps.monitoring);

        // Load stations
        await this.loadStations();
    }

    async loadRealtimeIndicators() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/monitoring/realtime`);
            const data = await response.json();
            
            if (data.success) {
                this.data.realtime = data.data;
                this.renderRealtimeIndicators(data.data);
            }
        } catch (error) {
            console.error('Error loading real-time indicators:', error);
            this.renderMockRealtimeIndicators();
        }
    }

    renderRealtimeIndicators(indicators) {
        const container = document.getElementById('realtimeIndicators');
        if (!container) return;

        const indicatorsHtml = indicators.map(indicator => `
            <div class="indicator-card" data-status="${indicator.status}">
                <div class="indicator-header">
                    <span class="indicator-name">${indicator.name}</span>
                    <span class="indicator-status">${indicator.status_icon}</span>
                </div>
                <div class="indicator-value">
                    <span class="value-number">${indicator.value}</span>
                    <span class="value-unit">${indicator.unit}</span>
                </div>
                <div class="indicator-trend">
                    <span class="trend-icon">${indicator.trend_icon}</span>
                    <span>${indicator.trend}</span>
                </div>
                <div class="indicator-footer">
                    <span class="footer-label">平均值</span>
                    <span class="footer-value">${indicator.avg}</span>
                </div>
            </div>
        `).join('');

        container.innerHTML = indicatorsHtml;
    }

    renderMockRealtimeIndicators() {
        const mockIndicators = [
            {
                name: 'WQI',
                status: 'good',
                status_icon: '✅',
                value: '42.3',
                unit: '',
                trend_icon: '↓',
                trend: '稳定',
                avg: '41.8'
            },
            {
                name: 'COD',
                status: 'warning',
                status_icon: '⚠️',
                value: '24.5',
                unit: 'mg/L',
                trend_icon: '↑',
                trend: '上升',
                avg: '22.1'
            },
            {
                name: 'pH',
                status: 'good',
                status_icon: '✅',
                value: '7.2',
                unit: '',
                trend_icon: '→',
                trend: '稳定',
                avg: '7.1'
            },
            {
                name: '溶解氧',
                status: 'good',
                status_icon: '✅',
                value: '8.1',
                unit: 'mg/L',
                trend_icon: '↑',
                trend: '上升',
                avg: '7.8'
            }
        ];

        this.renderRealtimeIndicators(mockIndicators);
    }

    initMonitoringCharts() {
        // Trend Chart
        this.createTrendChart();
        
        // Distribution Chart
        this.createDistributionChart();
    }

    createTrendChart() {
        const ctx = document.getElementById('trendChart');
        if (!ctx) return;

        this.charts.trend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                datasets: [{
                    label: 'WQI',
                    data: [42, 41, 43, 45, 43, 42],
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }

    createDistributionChart() {
        const ctx = document.getElementById('distributionChart');
        if (!ctx) return;

        this.charts.distribution = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['优秀', '良好', '一般', '较差', '很差'],
                datasets: [{
                    label: '站点数量',
                    data: [120, 36, 15, 6, 3],
                    backgroundColor: [
                        '#22c55e',
                        '#10b981',
                        '#f59e0b',
                        '#f97316',
                        '#ef4444'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initPredictionPage() {
        // Initialize prediction forms
        this.initPredictionForms();
        
        // Initialize prediction charts
        this.initPredictionCharts();
    }

    initPredictionForms() {
        // Pollution Level Form
        const pollutionForm = document.getElementById('pollutionLevelForm');
        if (pollutionForm) {
            pollutionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.predictPollutionLevel();
            });
        }

        // Water Quality Form
        const qualityForm = document.getElementById('waterQualityForm');
        if (qualityForm) {
            qualityForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.predictWaterQuality();
            });
        }

        // Day selector buttons
        document.querySelectorAll('.day-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.day-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
    }

    async predictPollutionLevel() {
        const formData = {
            Water_Quality_Index: parseFloat(document.getElementById('input_wqi').value),
            COD_mg_L: parseFloat(document.getElementById('input_cod').value),
            pH: parseFloat(document.getElementById('input_ph').value),
            Total_Nitrogen_mg_L: parseFloat(document.getElementById('input_tn').value),
            Total_Phosphorus_mg_L: parseFloat(document.getElementById('input_tp').value),
            Dissolved_Oxygen_mg_L: parseFloat(document.getElementById('input_do').value),
            Ammonia_Nitrogen_mg_L: parseFloat(document.getElementById('input_ammonia').value),
            Turbidity_NTU: parseFloat(document.getElementById('input_turbidity').value)
        };

        try {
            const response = await fetch(`${this.apiBaseUrl}/predict/pollution_level`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            this.displayPredictionResult(result);
        } catch (error) {
            console.error('Prediction error:', error);
            this.displayMockPredictionResult();
        }
    }

    displayPredictionResult(result) {
        const resultContainer = document.getElementById('predictionResult');
        if (!resultContainer) return;

        const resultHtml = `
            <div class="prediction-result-content">
                <div class="result-header">
                    <h4>预测结果</h4>
                    <div class="confidence-badge">
                        <i class="fas fa-check-circle"></i>
                        <span>置信度 ${result.confidence || '95%'}%</span>
                    </div>
                </div>
                <div class="result-main">
                    <div class="pollution-level ${result.level.toLowerCase()}">
                        <span class="level-text">${result.level || 'Good'}</span>
                        <span class="level-score">${result.score || '42.3'}</span>
                    </div>
                </div>
                <div class="result-details">
                    <div class="detail-item">
                        <span class="detail-label">预测概率</span>
                        <div class="probability-bars">
                            ${result.probabilities ? result.probabilities.map(prob => `
                                <div class="prob-item">
                                    <span class="prob-label">${prob.level}</span>
                                    <div class="prob-bar">
                                        <div class="prob-fill" style="width: ${prob.probability}%"></div>
                                    </div>
                                    <span class="prob-value">${prob.probability}%</span>
                                </div>
                            `).join('') : this.getMockProbabilities()}
                        </div>
                    </div>
                </div>
            </div>
        `;

        resultContainer.innerHTML = resultHtml;
    }

    displayMockPredictionResult() {
        const mockResult = {
            level: 'Good',
            score: '42.3',
            confidence: 95
        };

        this.displayPredictionResult(mockResult);
    }

    getMockProbabilities() {
        return `
            <div class="prob-item">
                <span class="prob-label">Excellent</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 15%"></div>
                </div>
                <span class="prob-value">15%</span>
            </div>
            <div class="prob-item">
                <span class="prob-label">Good</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 65%"></div>
                </div>
                <span class="prob-value">65%</span>
            </div>
            <div class="prob-item">
                <span class="prob-label">Fair</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 15%"></div>
                </div>
                <span class="prob-value">15%</span>
            </div>
            <div class="prob-item">
                <span class="prob-label">Poor</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 4%"></div>
                </div>
                <span class="prob-value">4%</span>
            </div>
            <div class="prob-item">
                <span class="prob-label">Very Poor</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 1%"></div>
                </div>
                <span class="prob-value">1%</span>
            </div>
        `;
    }

    async predictWaterQuality() {
        const formData = {
            province: document.getElementById('predict_province').value,
            date: document.getElementById('predict_date').value,
            days: document.querySelector('.day-btn.active')?.getAttribute('data-days') || '30'
        };

        // Mock implementation
        console.log('Water quality prediction:', formData);
        this.displayWaterQualityResult();
    }

    displayWaterQualityResult() {
        const resultContainer = document.getElementById('predictionResult');
        if (!resultContainer) return;

        const resultHtml = `
            <div class="water-quality-result">
                <div class="result-header">
                    <h4>水质指标预测</h4>
                    <div class="prediction-range">未来30天</div>
                </div>
                <div class="prediction-chart">
                    <canvas id="predictionTrendChart" height="200"></canvas>
                </div>
                <div class="prediction-summary">
                    <div class="summary-item">
                        <span class="summary-label">预测WQI</span>
                        <span class="summary-value">41.8 ± 2.3</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">趋势</span>
                        <span class="summary-value trend-up">↗ 上升趋势</span>
                    </div>
                </div>
            </div>
        `;

        resultContainer.innerHTML = resultHtml;

        // Create prediction trend chart
        this.createPredictionTrendChart();
    }

    createPredictionTrendChart() {
        const ctx = document.getElementById('predictionTrendChart');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 30}, (_, i) => `Day ${i + 1}`),
                datasets: [{
                    label: 'WQI预测',
                    data: Array.from({length: 30}, () => 40 + Math.random() * 8),
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: '置信区间上界',
                    data: Array.from({length: 30}, () => 44 + Math.random() * 4),
                    borderColor: '#f59e0b',
                    borderDash: [5, 5],
                    fill: false
                }, {
                    label: '置信区间下界',
                    data: Array.from({length: 30}, () => 38 + Math.random() * 4),
                    borderColor: '#f59e0b',
                    borderDash: [5, 5],
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    }
                }
            }
        });
    }

    initPredictionCharts() {
        // Confidence Chart
        this.createConfidenceChart();
        
        // Comparison Chart
        this.createComparisonChart();
    }

    createConfidenceChart() {
        const ctx = document.getElementById('confidenceChart');
        if (!ctx) return;

        this.charts.confidence = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['WQI', 'COD', 'pH', '总氮', '总磷', '溶解氧'],
                datasets: [{
                    label: '当前输入',
                    data: [85, 72, 90, 68, 75, 82],
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.2)'
                }, {
                    label: '历史平均',
                    data: [75, 75, 75, 75, 75, 75],
                    borderColor: '#94a3b8',
                    backgroundColor: 'rgba(148, 163, 184, 0.1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    createComparisonChart() {
        const ctx = document.getElementById('comparisonChart');
        if (!ctx) return;

        this.charts.comparison = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['本周', '上周', '上月同期', '去年同期'],
                datasets: [{
                    label: 'WQI对比',
                    data: [42.3, 43.1, 44.2, 46.8],
                    backgroundColor: '#0ea5e9'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    initAnalysisPage() {
        // Initialize analysis charts
        this.initAnalysisCharts();
        
        // Set up analysis controls
        this.setupAnalysisControls();
    }

    initAnalysisCharts() {
        // Destroy existing charts first
        this.destroyCharts(['multiTrend', 'correlation', 'distributionAnalysis']);
        
        // Multi-trend Chart
        this.createMultiTrendChart();
        
        // Correlation Chart
        this.createCorrelationChart();
        
        // Distribution Analysis Chart
        this.createDistributionAnalysisChart();
    }

    createMultiTrendChart() {
        const ctx = document.getElementById('multiTrendChart');
        if (!ctx) return;

        this.charts.multiTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
                datasets: [{
                    label: 'WQI',
                    data: [45, 43, 42, 41, 43, 42.3],
                    borderColor: '#0ea5e9',
                    tension: 0.4
                }, {
                    label: 'COD',
                    data: [22, 24, 23, 25, 24.5, 24],
                    borderColor: '#22c55e',
                    tension: 0.4
                }, {
                    label: '溶解氧',
                    data: [7.5, 7.8, 8.0, 7.9, 8.1, 8.0],
                    borderColor: '#f59e0b',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    }
                }
            }
        });
    }

    createCorrelationChart() {
        const ctx = document.getElementById('correlationChart');
        if (!ctx) return;

        this.charts.correlation = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'COD vs WQI',
                    data: Array.from({length: 50}, () => ({
                        x: 20 + Math.random() * 30,
                        y: 35 + Math.random() * 30
                    })),
                    backgroundColor: '#0ea5e9'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'COD (mg/L)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'WQI'
                        }
                    }
                }
            }
        });
    }

    createDistributionAnalysisChart() {
        const ctx = document.getElementById('distributionAnalysisChart');
        if (!ctx) return;

        this.charts.distributionAnalysis = new Chart(ctx, {
            type: 'boxPlot',
            data: {
                labels: ['WQI', 'COD', 'pH', '总氮', '总磷'],
                datasets: [{
                    label: '分布分析',
                    data: [
                        [35, 40, 42, 45, 50],
                        [18, 22, 24, 28, 32],
                        [6.5, 7.0, 7.2, 7.5, 8.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [0.1, 0.2, 0.3, 0.4, 0.5]
                    ],
                    backgroundColor: '#0ea5e9'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    setupAnalysisControls() {
        // Analysis type selector
        document.querySelector('.modern-select')?.addEventListener('change', (e) => {
            console.log('Analysis type changed:', e.target.value);
            // Update analysis based on type
        });

        // Time dimension buttons
        document.querySelectorAll('.dim-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.dim-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                console.log('Time dimension changed:', btn.textContent);
                // Update analysis based on time dimension
            });
        });

        // Indicator chips
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                chip.classList.toggle('active');
                console.log('Indicator toggled:', chip.textContent);
                // Update analysis based on selected indicators
            });
        });
    }

    initQualityPage() {
        // Initialize quality score ring
        this.createQualityScoreRing();
        
        // Load quality data
        this.loadQualityData();
    }

    createQualityScoreRing() {
        const canvas = document.getElementById('scoreRing');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 60;
        const score = 94.2;
        const angle = (score / 100) * 2 * Math.PI - Math.PI / 2;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw background circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 12;
        ctx.stroke();

        // Draw score arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, -Math.PI / 2, angle);
        ctx.strokeStyle = score >= 80 ? '#22c55e' : score >= 60 ? '#f59e0b' : '#ef4444';
        ctx.lineWidth = 12;
        ctx.stroke();
    }

    async loadQualityData() {
        // Mock quality data loading
        console.log('Loading quality data...');
    }

    initSpatialPage() {
        // Initialize spatial map
        this.initSpatialMap();
        
        // Set up spatial controls
        this.setupSpatialControls();
    }

    async initSpatialMap() {
        const mapContainer = document.getElementById('spatialMap');
        if (!mapContainer || this.maps.spatial) return;

        // Initialize Leaflet map
        this.maps.spatial = L.map('spatialMap').setView([35.8617, 104.1954], 5);

        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.maps.spatial);

        // Add drawing controls
        this.setupDrawingControls();
    }

    setupDrawingControls() {
        if (!this.maps.spatial) return;

        // Tool buttons
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const tool = btn.getAttribute('data-tool');
                this.activateSpatialTool(tool);
            });
        });
    }

    activateSpatialTool(tool) {
        console.log('Activating spatial tool:', tool);
        // Implement tool activation logic
    }

    setupSpatialControls() {
        // Layer toggles
        document.querySelectorAll('.toggle-item input').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const layer = e.target.nextElementSibling.nextElementSibling.textContent;
                console.log('Layer toggle:', layer, e.target.checked);
                // Toggle layer visibility
            });
        });

        // Parameter sliders
        document.querySelectorAll('.param-slider').forEach(slider => {
            slider.addEventListener('input', (e) => {
                const value = e.target.value;
                const valueDisplay = e.target.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = value + (valueDisplay.textContent.includes('km') ? ' km' : '%');
                }
            });
        });
    }

    initPageContent() {
        // Initialize all page content
        console.log('Initializing page content...');
    }

    initForms() {
        // Initialize all forms
        console.log('Initializing forms...');
    }

    initAnimations() {
        // Initialize animations and micro-interactions
        console.log('Initializing animations...');
    }

    async loadInitialData() {
        try {
            // Load stations data
            await this.loadStations();
            
            // Load statistics
            await this.loadStatistics();
            
            // Load alerts
            await this.loadAlerts();
            
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    async loadStations() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/stations`);
            const data = await response.json();
            
            if (data.success) {
                this.data.stations = data.data;
                this.populateProvinceFilter();
            }
        } catch (error) {
            console.error('Error loading stations:', error);
        }
    }

    async loadStatistics() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/statistics/summary`);
            const data = await response.json();
            
            if (data.success) {
                this.data.statistics = data.data;
            }
        } catch (error) {
            console.error('Error loading statistics:', error);
        }
    }

    async loadAlerts() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/monitoring/alerts`);
            const data = await response.json();
            
            if (data.success) {
                this.data.alerts = data.data;
            }
        } catch (error) {
            console.error('Error loading alerts:', error);
        }
    }

    populateProvinceFilter() {
        const provinceFilter = document.getElementById('provinceFilter');
        if (!provinceFilter || !this.data.stations.length) return;

        const provinces = [...new Set(this.data.stations.map(station => station.province))];
        
        provinces.forEach(province => {
            const option = document.createElement('option');
            option.value = province;
            option.textContent = province;
            provinceFilter.appendChild(option);
        });
    }

    // Utility functions
    showLoading() {
        document.getElementById('loadingOverlay').classList.add('show');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.remove('show');
    }

    formatNumber(num) {
        return num.toLocaleString();
    }

    formatDate(date) {
        return new Date(date).toLocaleDateString('zh-CN');
    }

    destroyCharts(chartNames) {
        chartNames.forEach(chartName => {
            if (this.charts[chartName]) {
                this.charts[chartName].destroy();
                delete this.charts[chartName];
            }
        });
    }

    // Chart initialization methods with cleanup
    initDashboardCharts() {
        // Destroy existing charts first
        this.destroyCharts(['waterLevel', 'pollutionDist']);
        
        // Water Level Chart
        this.createWaterLevelChart();
        
        // Pollution Distribution Chart
        this.createPollutionDistributionChart();
    }

    initMonitoringCharts() {
        // Destroy existing charts first
        this.destroyCharts(['trend', 'distribution']);
        
        // Trend Chart
        this.createTrendChart();
        
        // Distribution Chart
        this.createDistributionChart();
    }

    initPredictionCharts() {
        // Destroy existing charts first
        this.destroyCharts(['confidence', 'comparison']);
        
        // Confidence Chart
        this.createConfidenceChart();
        
        // Comparison Chart
        this.createComparisonChart();
    }
}

// Global utility functions
function applyMonitoringFilter() {
    console.log('Applying monitoring filter...');
    // Implement filter logic
}

function toggleMapView() {
    console.log('Toggling map view...');
}

function toggleFullscreen() {
    console.log('Toggling fullscreen...');
}

function exportMap() {
    console.log('Exporting map...');
}

function resetMapView() {
    console.log('Resetting map view...');
}

function zoomIn() {
    console.log('Zooming in...');
}

function zoomOut() {
    console.log('Zooming out...');
}

function toggleMeasure() {
    console.log('Toggling measure tool...');
}

function exportSpatialData() {
    console.log('Exporting spatial data...');
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, ready to initialize Modern Water Quality Platform');
});
