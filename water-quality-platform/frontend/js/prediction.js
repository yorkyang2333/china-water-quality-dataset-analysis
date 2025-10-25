/**
 * 预测模块
 * 处理污染等级预测和水质指标预测
 */

// 扩展WaterQualityApp类，添加预测功能
if (typeof WaterQualityApp !== 'undefined') {
    /**
     * 预测污染等级
     */
    WaterQualityApp.prototype.predictPollutionLevel = async function() {
        try {
            // 收集输入数据
            const inputData = {
                Water_Quality_Index: parseFloat(document.getElementById('input_wqi').value) || 35.0,
                COD_mg_L: parseFloat(document.getElementById('input_cod').value) || 20.0,
                BOD_mg_L: 4.0, // 默认值
                Total_Nitrogen_mg_L: parseFloat(document.getElementById('input_tn').value) || 3.0,
                Total_Phosphorus_mg_L: parseFloat(document.getElementById('input_tp').value) || 0.1,
                Dissolved_Oxygen_mg_L: parseFloat(document.getElementById('input_do').value) || 8.0,
                pH: parseFloat(document.getElementById('input_ph').value) || 7.0,
                Ammonia_N_mg_L: parseFloat(document.getElementById('input_ammonia').value) || 0.5,
                Turbidity_NTU: parseFloat(document.getElementById('input_turbidity').value) || 5.0,
                Temperature_C: 20.0, // 默认值
                Conductivity_uS_cm: 500.0, // 默认值
                Lead_ug_L: 1.0, // 默认值
                Cadmium_ug_L: 0.5, // 默认值
                Mercury_ug_L: 0.1, // 默认值
                Total_Heavy_Metals_ug_L: 1.6, // 默认值
                COD_BOD_Ratio: 5.0, // 默认值
                NP_Ratio: 30.0 // 默认值
            };

            this.showLoading();

            const response = await fetch(`${API_BASE_URL}/predict/pollution_level`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const result = await response.json();

            if (result.success) {
                this.displayPollutionLevelPrediction(result.data);
            } else {
                this.showError('预测失败: ' + result.error);
            }
        } catch (error) {
            console.error('❌ Prediction error:', error);
            this.showError('预测请求失败');
        } finally {
            this.hideLoading();
        }
    };

    /**
     * 显示污染等级预测结果
     */
    WaterQualityApp.prototype.displayPollutionLevelPrediction = function(data) {
        const color = this.getPollutionColor(data.prediction);
        const confidencePercent = (data.confidence * 100).toFixed(1);

        let probHtml = '<h4 style="margin-top: 15px; margin-bottom: 8px; font-size: 13px;">各等级概率：</h4>';

        // 按概率降序排序
        const sortedProbs = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1]);

        sortedProbs.forEach(([level, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            const levelColor = this.getPollutionColor(level);
            const barWidth = percentage;

            probHtml += `
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
                        <span style="color: ${levelColor}; font-weight: 500;">${level}</span>
                        <span style="font-weight: 600;">${percentage}%</span>
                    </div>
                    <div style="width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;">
                        <div style="width: ${barWidth}%; height: 100%; background: ${levelColor}; transition: width 0.3s;"></div>
                    </div>
                </div>
            `;
        });

        const resultHtml = `
            <h4>预测结果</h4>
            <div class="result-item" style="border: 2px solid ${color}; border-radius: 8px; padding: 12px; margin: 10px 0; background: ${color}15;">
                <div style="text-align: center;">
                    <div style="font-size: 18px; font-weight: 700; color: ${color}; margin-bottom: 5px;">
                        ${data.prediction}
                    </div>
                    <div style="font-size: 12px; color: #718096;">
                        置信度: ${confidencePercent}%
                    </div>
                </div>
            </div>
            ${probHtml}
            <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 8px; font-size: 11px; color: #856404;">
                💡 提示：预测结果基于输入的水质指标，实际情况可能受更多因素影响。
            </div>
        `;

        const resultDiv = document.getElementById('predictionResult');
        resultDiv.innerHTML = resultHtml;
        resultDiv.classList.add('show');

        console.log('✅ Pollution level prediction displayed');
    };

    /**
     * 预测水质指标
     */
    WaterQualityApp.prototype.predictWaterQuality = async function() {
        try {
            const inputData = {
                province: document.getElementById('predict_province').value,
                date: document.getElementById('predict_date').value
            };

            if (!inputData.date) {
                this.showError('请选择预测日期');
                return;
            }

            this.showLoading();

            const response = await fetch(`${API_BASE_URL}/predict/water_quality`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const result = await response.json();

            if (result.success) {
                this.displayWaterQualityPrediction(result.data);
            } else {
                this.showError('预测失败: ' + result.error);
            }
        } catch (error) {
            console.error('❌ Prediction error:', error);
            this.showError('预测请求失败');
        } finally {
            this.hideLoading();
        }
    };

    /**
     * 显示水质指标预测结果
     */
    WaterQualityApp.prototype.displayWaterQualityPrediction = function(data) {
        let predictionsHtml = '';

        const indicators = [
            { key: 'Water_Quality_Index', label: '水质指数', unit: '' },
            { key: 'COD_mg_L', label: 'COD', unit: 'mg/L' },
            { key: 'pH', label: 'pH', unit: '' },
            { key: 'Total_Nitrogen_mg_L', label: '总氮', unit: 'mg/L' },
            { key: 'Total_Phosphorus_mg_L', label: '总磷', unit: 'mg/L' },
            { key: 'Dissolved_Oxygen_mg_L', label: '溶解氧', unit: 'mg/L' },
            { key: 'Ammonia_N_mg_L', label: '氨氮', unit: 'mg/L' },
            { key: 'Turbidity_NTU', label: '浊度', unit: 'NTU' }
        ];

        indicators.forEach(ind => {
            const value = data.predictions[ind.key];
            if (value !== undefined) {
                predictionsHtml += `
                    <div class="result-item">
                        <span class="result-label">${ind.label}</span>
                        <span class="result-value">${value.toFixed(2)} ${ind.unit}</span>
                    </div>
                `;
            }
        });

        const resultHtml = `
            <h4>预测结果</h4>
            <div style="margin: 10px 0; padding: 10px; background: #f7fafc; border-radius: 8px;">
                <div class="result-item">
                    <span class="result-label">省份</span>
                    <span class="result-value">${data.province}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">日期</span>
                    <span class="result-value">${data.date}</span>
                </div>
            </div>
            ${predictionsHtml}
            ${data.note ? `
                <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 8px; font-size: 11px; color: #856404;">
                    💡 ${data.note}
                </div>
            ` : ''}
        `;

        const resultDiv = document.getElementById('predictionResult');
        resultDiv.innerHTML = resultHtml;
        resultDiv.classList.add('show');

        console.log('✅ Water quality prediction displayed');
    };

    /**
     * 填充示例数据（用于测试）
     */
    WaterQualityApp.prototype.fillExampleData = function() {
        document.getElementById('input_wqi').value = '35.5';
        document.getElementById('input_cod').value = '20.0';
        document.getElementById('input_ph').value = '7.2';
        document.getElementById('input_tn').value = '3.0';
    };

    /**
     * 清除预测表单
     */
    WaterQualityApp.prototype.clearPredictionForm = function() {
        document.getElementById('predictionResult').classList.remove('show');
        document.getElementById('predictionResult').innerHTML = '';
    };
}
