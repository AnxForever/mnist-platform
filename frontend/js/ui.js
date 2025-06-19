import * as ChartUtils from './chart_utils.js';
// UI 操作模块 - DOM 操作：渲染、更新、隐藏/显示元素

// --- DOM Element Selectors ---
const modelSelector = document.getElementById('trainedModelSelect');
const predictBtn = document.getElementById('predictBtn');
const resultContainer = document.getElementById('predictionResult');
const probabilityChartContainer = document.getElementById('probabilityChart');
const historyTableBody = document.getElementById('historyTableBody');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingOverlayText = document.getElementById('loading-overlay-text');

// --- Global Variables ---
let probabilityChart = null; // To hold the chart instance

// UI 操作模块 - DOM 操作：渲染、更新、隐藏/显示元素

// 渲染模型选择卡片
export function renderModelCards(models) {
    const container = document.getElementById('model-selection-grid');
    if (!container) {
        console.error('❌ 未找到模型选择容器');
        return;
    }
    
    // 清空现有内容
    container.innerHTML = '';
    
    models.forEach(model => {
        const card = createModelCard(model);
        container.appendChild(card);
    });
    
    console.log('✅ 已渲染模型卡片:', models.length);
}

// 创建单个模型卡片
function createModelCard(model) {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.dataset.modelId = model.id;
    
    const attentionBadge = model.has_attention ? 
        '<span class="attention-badge">⚡ Attention</span>' : '';
    
    card.innerHTML = `
        <div class="model-card-header">
            <h3 class="model-name">${model.name}</h3>
            ${attentionBadge}
        </div>
        <div class="model-description">
            ${model.description}
        </div>
        <div class="model-info">
            <div class="parameter-count">
                <span class="label">参数数量:</span>
                <span class="value">${formatNumber(model.parameter_count)}</span>
            </div>
        </div>
        <div class="model-card-footer">
            <label class="model-checkbox-container">
                <input type="checkbox" 
                       class="model-checkbox" 
                       value="${model.id}">
                <span class="checkmark"></span>
                <span class="checkbox-label">选择此模型</span>
            </label>
        </div>
    `;
    
    return card;
}

// 格式化数字显示
function formatNumber(num) {
    return num.toLocaleString();
}

// 显示错误消息
export function showErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <div class="error-content">
            <span class="error-icon">⚠️</span>
            <span class="error-text">${message}</span>
            <button class="error-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    document.body.insertBefore(errorDiv, document.body.firstChild);
    
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.remove();
        }
    }, 3000);
    
    console.error('❌ 错误消息:', message);
}

// 创建训练进度条
export function createTrainingProgressBars(jobs) {
    const container = document.getElementById('training-progress-container');
    if (!container) {
        console.error('❌ 未找到进度条容器');
        return;
    }
    
    container.innerHTML = '';
    
    jobs.forEach(job => {
        const progressBar = createProgressBar(job);
        container.appendChild(progressBar);
    });
    
    container.style.display = 'block';
    console.log('📊 已创建进度条:', jobs.length);
}

// 创建单个进度条
function createProgressBar(job) {
    const progressDiv = document.createElement('div');
    progressDiv.className = 'training-progress-item';
    progressDiv.dataset.jobId = job.job_id;
    
    progressDiv.innerHTML = `
        <div class="progress-header">
            <h4 class="progress-model-name">${getModelName(job.model_id)}</h4>
            <span class="progress-status status-queued">排队中</span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <span class="progress-percentage">0%</span>
        </div>
        <div class="progress-details">
            <div class="progress-metrics">
                <span class="metric">
                    <span class="metric-label">轮次:</span>
                    <span class="metric-value epoch">0/10</span>
                </span>
                <span class="metric">
                    <span class="metric-label">准确率:</span>
                    <span class="metric-value accuracy">0.0000</span>
                </span>
                <span class="metric">
                    <span class="metric-label">损失:</span>
                    <span class="metric-value loss">0.0000</span>
                </span>
                <span class="metric">
                    <span class="metric-label">最佳:</span>
                    <span class="metric-value best-accuracy">0.0000</span>
                </span>
                <span class="metric">
                    <span class="metric-label">速度:</span>
                    <span class="metric-value speed">0 samples/s</span>
                </span>
                <span class="metric">
                    <span class="metric-label">学习率:</span>
                    <span class="metric-value learning-rate">0.001</span>
                </span>
            </div>
        </div>
    `;
    
    return progressDiv;
}

// 获取模型显示名称
export function getModelName(modelId) {
    const modelMap = {
        'mlp': 'MLP',
        'cnn': 'CNN', 
        'rnn': 'RNN',
        'mlp_attention': 'MLP + Attention',
        'cnn_attention': 'CNN + Attention',
        'rnn_attention': 'RNN + Attention'
    };
    return modelMap[modelId] || modelId;
}

// 更新进度条状态
export function updateProgressBar(jobId, progressData) {
    const progressItem = document.querySelector(`[data-job-id="${jobId}"]`);
    if (!progressItem) {
        console.warn(`⚠️ 未找到job ${jobId} 的进度条`);
        return;
    }
    
    const status = progressData.status;
    const progress = progressData.progress;
    const isNewRecord = progressData.is_new_record;
    const finalAccuracy = progressData.final_accuracy;
    
    // 更新状态标签
    const statusElement = progressItem.querySelector('.progress-status');
    let statusText = getStatusText(status);
    
    // 如果是新纪录，添加新纪录徽章
    if (status === 'completed' && isNewRecord) {
        statusText += ' 🏆';
        progressItem.classList.add('new-record');
        
        // 添加新纪录徽章到标题
        const modelName = progressItem.querySelector('.progress-model-name');
        if (!modelName.querySelector('.new-record-badge')) {
            const badge = document.createElement('span');
            badge.className = 'new-record-badge';
            badge.textContent = '🏆 新纪录!';
            modelName.appendChild(badge);
        }
    }
    
    statusElement.textContent = statusText;
    statusElement.className = `progress-status status-${status}`;
    
    // 更新进度条
    const progressFill = progressItem.querySelector('.progress-fill');
    const progressPercentage = progressItem.querySelector('.progress-percentage');
    
    const percentage = status === 'completed' ? 100 : (progress.percentage || 0);
    progressFill.style.width = `${percentage}%`;
    progressPercentage.textContent = `${percentage}%`;
    
    // 如果是新纪录，添加特殊样式
    if (isNewRecord) {
        progressFill.classList.add('record');
    }
    
    // 更新指标
    const epochElement = progressItem.querySelector('.metric-value.epoch');
    const accuracyElement = progressItem.querySelector('.metric-value.accuracy');
    const lossElement = progressItem.querySelector('.metric-value.loss');
    const bestAccuracyElement = progressItem.querySelector('.metric-value.best-accuracy');
    const speedElement = progressItem.querySelector('.metric-value.speed');
    const learningRateElement = progressItem.querySelector('.metric-value.learning-rate');
    
    if (status === 'completed') {
        epochElement.textContent = `${progress.total_epochs}/${progress.total_epochs}`;
        accuracyElement.textContent = (finalAccuracy || progress.accuracy).toFixed(4);
        bestAccuracyElement.textContent = (finalAccuracy || progress.best_accuracy).toFixed(4);
    } else {
        epochElement.textContent = `${progress.current_epoch}/${progress.total_epochs}`;
        accuracyElement.textContent = progress.accuracy.toFixed(4);
        bestAccuracyElement.textContent = progress.best_accuracy.toFixed(4);
    }
    
    if (progress.loss !== undefined) {
        lossElement.textContent = progress.loss.toFixed(4);
    }
    
    if (progress.samples_per_second !== undefined) {
        speedElement.textContent = `${Math.round(progress.samples_per_second)} samples/s`;
    }

    if (progressData.config && progressData.config.learning_rate) {
        learningRateElement.textContent = progressData.config.learning_rate;
    }
}

// 获取状态显示文本
function getStatusText(status) {
    const statusMap = {
        'queued': '排队中',
        'running': '训练中',
        'completed': '已完成',
        'error': '错误',
        'cancelled': '已取消'
    };
    return statusMap[status] || '未知状态';
}

// --- Canvas Drawing ---
let canvas, ctx, isDrawing = false, lastX, lastY;

export function initializeCanvas() {
    canvas = document.getElementById('drawing-canvas');
    if (!canvas) {
        console.error('未找到 canvas 元素');
        return;
    }
    ctx = canvas.getContext('2d');
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    
    // 初始化画笔大小
    const brushSlider = document.getElementById('brush-size-slider');
    updateBrushSize(parseInt(brushSlider.value));

    // 绑定事件
    setupCanvasDrawing();
    
    return true; // 明确返回成功状态
}

export function setupCanvasDrawing() {
    if (!canvas) return;
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('touchstart', handleTouch, { passive: false });
    canvas.addEventListener('touchmove', handleTouch, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);
}

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getCoordinates(e);
}

function draw(e) {
    if (!isDrawing) return;
    const [x, y] = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return [
        (e.clientX - rect.left) * scaleX, 
        (e.clientY - rect.top) * scaleY
    ];
}

export function clearCanvas() {
    if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        console.log('画布已清除');
    }
}

export function updateBrushSize(size) {
    if (ctx) {
        ctx.lineWidth = size;
        document.getElementById('brush-size-value').textContent = `${size}px`;
    }
}

export function getCanvasImageData() {
    if (canvas) {
        return canvas.toDataURL('image/png');
    }
    return null;
}

export function isCanvasEmpty() {
    if (!ctx || !canvas) return true;
    const pixelBuffer = new Uint32Array(
        ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer
    );
    // 检查是否有非黑色像素
    return !pixelBuffer.some(color => color !== 0xFF000000);
}


// --- Prediction UI ---
export function renderPredictionResult(result) {
    if (!result) {
        showEmptyResult();
        return;
    }
    const container = document.getElementById('prediction-result');
    container.innerHTML = `
        <div class="prediction-number">
            预测结果: <span class="predicted-digit">${result.prediction}</span>
        </div>
        <div class="prediction-confidence">
            置信度: <span class="confidence-value">${(result.confidence * 100).toFixed(2)}%</span>
        </div>
        <div class="prediction-chart-container">
            <canvas id="prediction-chart"></canvas>
        </div>
    `;
    
    // 渲染概率图表
    renderProbabilityChart(result.probabilities);
    console.log('✅ 已渲染预测结果:', result);
}

export function showPredictionLoading() {
    const container = document.getElementById('prediction-result');
    container.innerHTML = '<div class="loading-spinner"></div><p>正在识别中...</p>';
}

export function showEmptyResult() {
    const container = document.getElementById('prediction-result');
    container.innerHTML = '<p class="empty-state">请在左侧绘制数字，然后点击"识别"按钮</p>';
}

// --- Trained Model Selector for Prediction ---
export function renderTrainedModels(models) {
    const selectElement = document.getElementById('prediction-model-select');
    if (!selectElement) return;

    // 保存当前选中的值
    const currentValue = selectElement.value;

    selectElement.innerHTML = '<option value="">请选择已训练的模型</option>';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.name;
        selectElement.appendChild(option);
    });
    
    // 恢复之前的选中状态
    if (currentValue && models.some(m => m.id === currentValue)) {
        selectElement.value = currentValue;
    }

    updatePredictButtonState();
}

export function updatePredictButtonState() {
    const selectElement = document.getElementById('prediction-model-select');
    const predictBtn = document.getElementById('predict-btn');
    if (selectElement && predictBtn) {
        const selectedModel = selectElement.value;
        if (selectedModel) {
            predictBtn.disabled = false;
            predictBtn.textContent = '🧠 开始识别';
        } else {
            predictBtn.disabled = true;
            predictBtn.textContent = '请先选择模型';
        }
    }
}


// --- History Table ---
let currentSort = { column: 'date', direction: 'desc' };

export function renderHistoryTable(historyData) {
    const container = document.getElementById('history-table-container');
    if (!container) return;

    if (!historyData || historyData.length === 0) {
        container.innerHTML = '<p class="empty-state">暂无训练历史记录</p>';
        return;
    }

    const sortedData = sortHistoryData([...historyData], currentSort.column, currentSort.direction);

    let tableHTML = '<table class="history-table"><thead><tr>';
    const headers = {
        completion_time_iso: '训练日期',
        model_name: '模型名称',
        final_accuracy: '最终准确率',
        training_duration_sec: '训练耗时(秒)',
        actions: '操作' // 新增操作列
    };

    for (const key in headers) {
        // 'actions' 列不可排序
        tableHTML += createHeaderCell(key, headers[key], key !== 'actions');
    }
    tableHTML += '</tr></thead><tbody>';

    sortedData.forEach(record => {
        const epochs = record.hyperparameters ? record.hyperparameters.epochs : 'N/A';
        const lr = record.hyperparameters ? record.hyperparameters.learning_rate : 'N/A';
        const batchSize = record.hyperparameters ? record.hyperparameters.batch_size : 'N/A';
        const accuracy = record.metrics && typeof record.metrics.final_accuracy === 'number'
            ? (record.metrics.final_accuracy * 100).toFixed(2) + '%'
            : 'N/A';
        const duration = record.metrics && typeof record.metrics.training_duration_sec === 'number'
            ? record.metrics.training_duration_sec.toFixed(2) + 's'
            : 'N/A';
        const trainingDate = record.completion_time_iso ? formatDate(record.completion_time_iso) : 'N/A';

        // 主摘要行
        tableHTML += `
            <tr class="history-main-row" data-job-id="${record.job_id}">
                <td>${trainingDate}</td>
                <td>${record.model_name || '未知模型'}</td>
                <td class="accuracy-cell">${accuracy}</td>
                <td>${duration}</td>
                <td>
                    <button class="btn-icon btn-details" onclick="Module.toggleHistoryDetails('${record.job_id}')">
                        <i class="fas fa-chevron-down"></i>
                    </button>
                </td>
            </tr>
        `;

        // 隐藏的详情行
        tableHTML += `
            <tr class="history-details-row" data-job-id="${record.job_id}">
                <td colspan="5">
                    <div class="details-card">
                        <h4>训练超参数详情</h4>
                        <div class="details-grid">
                            <div><strong>训练轮数:</strong><span>${epochs}</span></div>
                            <div><strong>学习率:</strong><span>${lr}</span></div>
                            <div><strong>批次大小:</strong><span>${batchSize}</span></div>
                            <div><strong>Job ID:</strong><span class="job-id-span">${record.job_id}</span></div>
                        </div>
                    </div>
                </td>
            </tr>
        `;
    });

    tableHTML += '</tbody></table>';
    container.innerHTML = tableHTML;
    
    container.querySelectorAll('.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const sortKey = th.dataset.sort;
            if (currentSort.column === sortKey) {
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            } else {
                currentSort.column = sortKey;
                currentSort.direction = 'desc';
            }
            renderHistoryTable(historyData);
        });
    });
}

/**
 * 切换训练历史详情行的可见性
 * @param {string} jobId - 任务ID
 */
export function toggleHistoryDetails(jobId) {
    const detailsRow = document.querySelector(`.history-details-row[data-job-id="${jobId}"]`);
    const mainRow = document.querySelector(`.history-main-row[data-job-id="${jobId}"]`);
    const buttonIcon = mainRow.querySelector('.btn-details i');

    if (detailsRow && mainRow && buttonIcon) {
        const isVisible = detailsRow.classList.toggle('visible');
        mainRow.classList.toggle('is-expanded', isVisible);
        buttonIcon.classList.toggle('fa-chevron-down', !isVisible);
        buttonIcon.classList.toggle('fa-chevron-up', isVisible);
    }
}

function createHeaderCell(key, title, sortable = true) {
    let cellClass = sortable ? 'sortable' : '';
    let cell = `<th data-sort="${key}" class="${cellClass}">`;
    cell += title;
    if (sortable && currentSort.column === key) {
        cell += currentSort.direction === 'asc' ? ' ▲' : ' ▼';
    }
    cell += '</th>';
    return cell;
}

function handleSort(sortKey, historyData) {
    if (currentSort.column === sortKey) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.column = sortKey;
        currentSort.direction = 'desc';
    }
    renderHistoryTable(historyData);
}

function sortHistoryData(data, column, direction) {
    return data.sort((a, b) => {
        let valA, valB;

        // 根据列名从正确嵌套的对象中提取值
        switch (column) {
            case 'completion_time_iso':
            case 'model_name':
                valA = a[column];
                valB = b[column];
                break;
            case 'final_accuracy':
            case 'training_duration_sec':
                valA = a.metrics ? a.metrics[column] : null;
                valB = b.metrics ? b.metrics[column] : null;
                break;
            case 'epochs':
            case 'learning_rate':
            case 'batch_size':
                valA = a.hyperparameters ? a.hyperparameters[column] : null;
                valB = b.hyperparameters ? b.hyperparameters[column] : null;
                break;
            default:
                valA = a[column];
                valB = b[column];
        }

        // 处理null或undefined值，将它们排在最后
        if (valA == null) return 1;
        if (valB == null) return -1;

        if (column === 'completion_time_iso') {
            valA = new Date(valA);
            valB = new Date(valB);
        }

        if (valA < valB) {
            return direction === 'asc' ? -1 : 1;
        }
        if (valA > valB) {
            return direction === 'asc' ? 1 : -1;
        }
        return 0;
    });
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    const pad = (num) => num.toString().padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

// --- Comparison Charts ---
export function renderComparisonCharts(processedData) {
    if (!processedData) {
        console.warn("没有可用于对比的有效数据");
        // 这里可以添加隐藏或清空旧图表的逻辑
        return;
    }

    const { barData, radarData, lineChartData } = processedData;

    // 1. 渲染三张柱状图
    if (barData) {
        ChartUtils.createBarChart('accuracyBarChart', barData.accuracies, '最高准确率对比');
        ChartUtils.createBarChart('speedBarChart', barData.speeds, '训练耗时对比 (秒)');
        
        const paramsData = JSON.parse(JSON.stringify(barData.params));
        paramsData.datasets[0].data = paramsData.datasets[0].data.map(p => (p / 10000).toFixed(2));
        ChartUtils.createBarChart('paramsBarChart', paramsData, '模型参数量对比 (万)');
    }

    // 2. 渲染雷达图
    if (radarData) {
        ChartUtils.createRadarChart('radarChart', radarData);
    }

    // 3. 渲染学习曲线图
    if (lineChartData) {
        ChartUtils.createLineChart('learningCurveChart', lineChartData, '学习曲线对比 (验证集准确率)');
    }
}


// --- Misc UI Helpers ---
export function populateModelSelector(models) {
    const selector = document.getElementById('prediction-model-select');
    if (!selector) return;
    const selectedValue = selector.value;
    selector.innerHTML = '<option value="">请选择模型</option>';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.name;
        selector.appendChild(option);
    });
    selector.value = selectedValue;
}

export function handleCanvasUpdate(isEmpty) {
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        const modelSelected = document.getElementById('prediction-model-select').value !== '';
        predictBtn.disabled = isEmpty || !modelSelected;
    }
}

export function renderPrediction(prediction, probabilities) {
    // ...
}

export function clearPrediction() {
    // ...
}

export function navigateTo(pageId) {
    document.querySelectorAll('.page').forEach(page => page.classList.add('hidden'));
    document.getElementById(pageId).classList.remove('hidden');
}

export function getSelectedModels() {
    return Array.from(document.querySelectorAll('.model-checkbox:checked')).map(cb => cb.value);
}

export function getTrainingConfig() {
    return {
        epochs: document.getElementById('epochs-slider').value,
        lr: document.getElementById('learning-rate-input').value,
        batch_size: document.getElementById('batch-size-input').value,
    };
}

export function showTrainingModal(selectedModels, config) {
    let modelListHTML = selectedModels.map(id => `<li>${getModelName(id)}</li>`).join('');
    
    const modalHTML = `
        <div class="modal-backdrop">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>确认训练任务</h3>
                    <button class="modal-close-btn">&times;</button>
                </div>
                <div class="modal-body">
                    <p>即将为以下模型启动训练：</p>
                    <ul class="model-confirm-list">${modelListHTML}</ul>
                    <hr>
                    <p><strong>训练参数:</strong></p>
                    <ul class="param-confirm-list">
                        <li><strong>训练轮数:</strong> ${config.epochs}</li>
                        <li><strong>学习率:</strong> ${config.lr}</li>
                        <li><strong>批次大小:</strong> ${config.batch_size}</li>
                    </ul>
                </div>
                <div class="modal-footer">
                    <button class="btn-secondary modal-cancel-btn">取消</button>
                    <button class="btn-primary modal-confirm-btn">启动训练</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    // Add event listeners for modal buttons
    // ...
}


export function updateTrainingProgress(jobId, modelId, progress) {
    const progressItem = document.querySelector(`.progress-item[data-job-id="${jobId}"]`);
    if (!progressItem) return;

    // ... Update progress bar, percentages, etc.
    const statusEl = progressItem.querySelector('.status-text');
    statusEl.textContent = getStatusMessage(progress.status);
    progressItem.className = `progress-item status-${progress.status}`;
}

export function showLoadingOverlay(text = '加载中...') {
    if (loadingOverlay) {
        loadingOverlayText.textContent = text;
        loadingOverlay.classList.remove('hidden');
    }
    // 安全超时
    setTimeout(() => {
        hideLoadingOverlay();
    }, 10000); // 10秒后自动隐藏，防止卡死
}

export function hideLoadingOverlay() {
    if (loadingOverlay) {
        loadingOverlay.classList.add('hidden');
    }
}

export function showError(title, message) {
    // Implement a more robust error display, e.g., a toast notification
    alert(`${title}\n\n${message}`);
}

// 渲染概率分布图表
function renderProbabilityChart(probabilities) {
    const chartContainer = document.getElementById('prediction-chart');
    if (!chartContainer) return;
    
    const ctx = chartContainer.getContext('2d');
    
    if (window.predictionChart instanceof Chart) {
        window.predictionChart.destroy();
    }

    window.predictionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from(Array(10).keys()).map(String),
            datasets: [{
                label: '模型预测概率',
                data: probabilities,
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    max: 1,
                    ticks: { color: '#fff' }
                },
                y: {
                    ticks: { color: '#fff' }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function getStatusMessage(status) {
    switch (status) {
        case 'running': return '训练中...';
        case 'queued': return '排队中';
        case 'completed': return '完成';
        case 'error': return '错误';
        default: return '未知';
    }
}