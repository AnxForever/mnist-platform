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
    
    // 直接使用传入的 model_name，不再调用那个垃圾 getModelName
    progressDiv.innerHTML = `
        <div class="progress-header">
            <h4 class="progress-model-name">${job.model_name || job.model_id}</h4>
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

// 获取选中的模型ID列表
export function getSelectedModelIds() {
    // 这个函数已经没用了，但暂时保留以免其他地方意外调用
    console.warn("getSelectedModelIds is deprecated and will be removed.");
    return [];
}

// 更新对比状态栏
export function updateComparisonStatusBar(count) {
    const statusBar = document.getElementById('comparison-status-bar');
    const statusText = document.getElementById('comparison-status-text');
    if (!statusBar || !statusText) return;

    if (count > 0) {
        statusText.textContent = `已选择 ${count} 项进行对比`;
        statusBar.classList.add('visible');
    } else {
        statusBar.classList.remove('visible');
    }
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
    
    if (status === 'completed') {
        statusText = isNewRecord ? `🎉 新纪录: ${finalAccuracy.toFixed(4)}` : `完成: ${finalAccuracy.toFixed(4)}`;
        statusElement.className = isNewRecord ? 'progress-status status-new-record' : 'progress-status status-completed';
    } else {
        statusElement.className = `progress-status status-${status}`;
    }
    statusElement.textContent = statusText;

    // 更新进度条填充和百分比
    const fill = progressItem.querySelector('.progress-fill');
    const percentage = progressItem.querySelector('.progress-percentage');
    fill.style.width = `${progress.percentage}%`;
    percentage.textContent = `${Math.round(progress.percentage)}%`;

    // 更新详细指标
    progressItem.querySelector('.epoch').textContent = `${progress.current_epoch}/${progress.total_epochs}`;
    progressItem.querySelector('.accuracy').textContent = progress.accuracy.toFixed(4);
    progressItem.querySelector('.loss').textContent = progress.loss.toFixed(4);
    progressItem.querySelector('.best-accuracy').textContent = progress.best_accuracy.toFixed(4);
    progressItem.querySelector('.speed').textContent = `${Math.round(progress.samples_per_second)} samples/s`;
    
    const lrElement = progressItem.querySelector('.learning-rate');
    if (lrElement && progressData.config) {
        lrElement.textContent = progressData.config.learning_rate;
    }
}

function getStatusText(status) {
    const statusMap = {
        queued: '排队中',
        running: '训练中',
        completed: '已完成',
        error: '错误'
    };
    return statusMap[status] || status;
}

// =================================================================
// 6. 手写识别 Canvas 相关
// =================================================================

let canvas, ctx, isDrawing = false, lastX, lastY;

export function initializeCanvas() {
    canvas = document.getElementById('drawing-canvas');
    if (!canvas) {
        console.error('❌ 未找到 Canvas 元素');
        return;
    }
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // 初始化画笔样式
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    updateBrushSize(document.getElementById('brush-size-slider').value);
    
    setupCanvasDrawing();
}

export function setupCanvasDrawing() {
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
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
    updatePredictButtonState();
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
    return [e.clientX - rect.left, e.clientY - rect.top];
}

export function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    console.log('画布已清除');
    updatePredictButtonState();
    showEmptyResult();
}

export function updateBrushSize(size) {
    ctx.lineWidth = size;
    document.getElementById('brush-size-value').textContent = `${size}px`;
}

export function getCanvasImageData() {
    return canvas.toDataURL('image/png');
}

export function isCanvasEmpty() {
    const pixelBuffer = new Uint32Array(
        ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer
    );
    // 检查是否有非黑色像素
    return !pixelBuffer.some(color => color !== 0xff000000);
}


// =================================================================
// 7. 预测结果渲染
// =================================================================

export function renderPredictionResult(result) {
    const container = document.getElementById('prediction-result');
    if (!container) return;

    if (result.error) {
        container.innerHTML = `<div class="prediction-error">错误: ${result.error}</div>`;
        return;
    }

    const { prediction, probabilities } = result;
    const confidence = Math.max(...probabilities) * 100;

    container.innerHTML = `
        <div class="prediction-header">预测结果</div>
        <div class="prediction-number">${prediction}</div>
        <div class="prediction-confidence">置信度: ${confidence.toFixed(2)}%</div>
        <div class="probability-chart-container">
            <canvas id="probability-chart"></canvas>
        </div>
    `;

    ChartUtils.createBarChart('probability-chart', { labels: Array.from(Array(10).keys()).map(String), datasets: [{ label: '概率', data: probabilities, backgroundColor: 'rgba(75, 192, 192, 0.6)' }] }, '数字概率分布');
}

export function showPredictionLoading() {
    const container = document.getElementById('prediction-result');
    if (container) {
        container.innerHTML = `<div class="loading-spinner"></div><p>识别中...</p>`;
    }
}
export function showEmptyResult() {
    const container = document.getElementById('prediction-result');
    if (container) {
        container.innerHTML = `<div class="empty-state"><i class="fas fa-paint-brush"></i><p>请在左侧面板中绘制一个数字</p></div>`;
    }
}

// =================================================================
// 8. 已训练模型列表（用于预测）
// =================================================================

export function renderTrainedModels(models) {
    const select = document.getElementById('prediction-model-select');
    if (!select) return;

    select.innerHTML = ''; // 清空

    if (!models || models.length === 0) {
        select.innerHTML = '<option value="">无可用模型</option>';
        return;
    }

    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.model_id;
        option.textContent = model.display_name;
        select.appendChild(option);
    });
}

// =================================================================
// 9. 训练历史 & 对比
// =================================================================

export function updatePredictButtonState() {
    const select = document.getElementById('prediction-model-select');
    const predictBtn = document.getElementById('predict-btn');
    if (!select || !predictBtn) return;

    const modelSelected = select.value !== '';
    const canvasEmpty = isCanvasEmpty();
    
    predictBtn.disabled = !modelSelected || canvasEmpty;
    
    if (!modelSelected) {
        predictBtn.textContent = '请选择模型';
    } else if (canvasEmpty) {
        predictBtn.textContent = '请绘制数字';
    } else {
        predictBtn.textContent = '开始识别';
    }
}

export function renderHistoryTable(historyData) {
    const tableBody = document.getElementById('history-table-body');
    if (!tableBody) {
        console.error('❌ 未找到历史表格容器 #history-table-body');
        return;
    }
    tableBody.innerHTML = ''; // 清空

    if (historyData.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="10" class="empty-row">暂无训练历史记录</td></tr>';
        return;
    }

    const sortedData = historyData.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    const headerRow = document.createElement('tr');
    headerRow.innerHTML = `
        <th></th>
        <th>模型名称</th>
        <th>状态</th>
        <th>最高准确率</th>
        <th>最终损失</th>
        <th>轮数</th>
        <th>学习率</th>
        <th>批次大小</th>
        <th>完成时间</th>
        <th>操作</th>
    `;
    const a = tableBody.parentElement.querySelector('thead');
    if (a) {
        a.innerHTML = '';
        a.appendChild(headerRow);
    }
    

    sortedData.forEach(record => {
        const row = document.createElement('tr');
        row.dataset.jobId = record.job_id;

        const statusClass = `status-${record.status.toLowerCase()}`;
        const statusText = getStatusText(record.status);

        // 状态同步：根据全局状态决定按钮的初始样式和文本
        const isAddedToCompare = window.AppState.comparisonSet.has(record.job_id);
        const compareButtonText = isAddedToCompare ? '已添加' : '加入对比';
        const compareButtonClass = isAddedToCompare ? 'btn-compare added' : 'btn-compare';

        row.innerHTML = `
            <td>
                <span class="details-toggle" onclick="window.Module.showDetailsModal('${record.job_id}')">
                    <i class="fas fa-search-plus"></i>
                </span>
            </td>
            <td>${record.model_name}</td>
            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
            <td>${((record.best_accuracy || 0) * 100).toFixed(2)}%</td>
            <td>${(record.final_loss || 0).toFixed(4)}</td>
            <td>${record.config ? record.config.epochs : 'N/A'}</td>
            <td>${record.config ? record.config.learning_rate : 'N/A'}</td>
            <td>${record.config ? record.config.batch_size : 'N/A'}</td>
            <td>${formatDate(record.timestamp)}</td>
            <td>
                <button class="${compareButtonClass}" data-job-id="${record.job_id}">${compareButtonText}</button>
            </td>
        `;

        tableBody.appendChild(row);
    });
}

// 彻底替换旧的 toggleHistoryDetails
export function showDetailsModal(jobId) {
    const modal = document.getElementById('details-modal');
    const modalBody = document.getElementById('modal-body');
    if (!modal || !modalBody) {
        console.error('❌ 模态框元素未找到!');
        return;
    }

    const record = window.AppState.trainingHistory.find(r => r.job_id === jobId);
    if (!record) {
        showErrorMessage('未找到该条记录的详细数据。');
        return;
    }
    
    // 填充模态框内容
    modalBody.innerHTML = `
        <div class="details-content">
            <div class="details-chart">
                <canvas id="modal-chart-${jobId}"></canvas>
            </div>
            <div class="details-info">
                <h4>训练详情</h4>
                <p><strong>Job ID:</strong> ${jobId}</p>
                <p><strong>模型参数:</strong> ${(record.model_params || 0).toLocaleString()}</p>
                <p><strong>训练时长:</strong> ${formatDuration(record.duration_seconds || 0)}</p>
                <h4>逐轮数据</h4>
                <div class="epoch-table-container">
                    <table>
                       <thead>
                           <tr><th>轮次</th><th>准确率</th><th>损失</th></tr>
                       </thead>
                       <tbody>
                           ${(record.epoch_metrics && record.epoch_metrics.length > 0) ? record.epoch_metrics.map(metric => `
                               <tr>
                                   <td>${metric.epoch}</td>
                                   <td>${(metric.accuracy * 100).toFixed(2)}%</td>
                                   <td>${(metric.loss || 0).toFixed(4)}</td>
                               </tr>
                           `).join('') : '<tr><td colspan="3">无逐轮数据</td></tr>'}
                       </tbody>
                    </table>
                </div>
            </div>
        </div>`;
    
    // 显示模态框
    modal.style.display = 'flex';
    setTimeout(() => {
        modal.style.opacity = '1';
        modal.querySelector('.modal-content').style.transform = 'scale(1)';
    }, 10);

    // 渲染图表
    try {
        if (record.epoch_metrics && record.epoch_metrics.length > 0) {
            const lineChartData = {
                labels: record.epoch_metrics.map(m => `Epoch ${m.epoch}`),
                datasets: [
                    {
                        label: '准确率',
                        data: record.epoch_metrics.map(m => m.accuracy),
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        yAxisID: 'y',
                        fill: true,
                    },
                    {
                        label: '损失',
                        data: record.epoch_metrics.map(m => m.loss),
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        yAxisID: 'y1',
                        fill: true,
                    }
                ]
            };
            ChartUtils.createLineChart(`modal-chart-${jobId}`, lineChartData, `训练曲线`);
        } else {
            document.getElementById(`modal-chart-${jobId}`).parentElement.innerHTML = '<p class="empty-state">无图表数据</p>';
        }
    } catch (error) {
        console.error(`❌ 渲染 Job ${jobId} 的图表失败:`, error);
        document.getElementById(`modal-chart-${jobId}`).parentElement.innerHTML = '<p class="error-text">图表加载失败</p>';
    }
}

export function hideDetailsModal() {
    const modal = document.getElementById('details-modal');
    if (modal) {
        modal.style.opacity = '0';
        modal.querySelector('.modal-content').style.transform = 'scale(0.95)';
        setTimeout(() => {
            modal.style.display = 'none';
            document.getElementById('modal-body').innerHTML = ''; // 清空内容
        }, 300);
    }
}

function createHeaderCell(key, title, isSortable = true) {
    // Deprecated
}

function handleSort(sortKey, historyData) {
    // Deprecated
}

function sortHistoryData(data, column, direction) {
    // Deprecated
}


function formatDuration(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}秒`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}分 ${remainingSeconds}秒`;
}

function formatDate(dateString) {
    if (!dateString) return 'N/A'; // 卫语句：如果输入无效，直接返回
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return '无效日期'; // 卫语句：如果日期解析失败，返回提示
    
    const pad = (num) => num.toString().padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}
// =================================================================
// 10. 对比图表渲染
// =================================================================
export function renderComparisonCharts(processedData) {
    if (!processedData) {
        console.warn('⚠️ 没有可供渲染的对比数据');
        ChartUtils.clearAllCharts();
        return;
    }
    
    try {
        ChartUtils.createRadarChart('comparison-radar-chart', processedData.radar);
        ChartUtils.createBarChart('comparison-bar-chart', processedData.bar, '模型最高准确率对比');
        ChartUtils.createLineChart('comparison-line-chart', processedData.line, '模型学习曲线对比');
        console.log('✅ 已渲染所有对比图表');
    } catch (e) {
        console.error("❌ 渲染对比图表时出错:", e);
        showErrorMessage("渲染对比图表失败，请检查数据或控制台日志。");
    }
}

// =================================================================
// 11. 杂项 & 辅助函数
// =================================================================

export function populateModelSelector(models) {
    // Deprecated
}

export function handleCanvasUpdate(isEmpty) {
    // Deprecated
}

export function renderPrediction(prediction, probabilities) {
    // Deprecated in favor of renderPredictionResult
}

export function clearPrediction() {
     // Deprecated in favor of showEmptyResult
}

export function navigateTo(pageId) {
    // Deprecated - handled by tab navigation
}

export function getSelectedModels() {
     // Deprecated
    return [];
}

export function getTrainingConfig() {
    // Deprecated
    return {};
}

export function showTrainingModal(selectedModels, config) {
   // Deprecated
}

export function updateTrainingProgress(jobId, modelId, progress) {
    // Deprecated, use updateProgressBar instead
}

// --- Loading Overlay ---
export function showLoadingOverlay(text = '加载中...') {
    if (loadingOverlay && loadingOverlayText) {
        loadingOverlayText.textContent = text;
        loadingOverlay.style.display = 'flex';
    }
}

export function hideLoadingOverlay() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

export function showError(title, message) {
    // A more generic error modal if needed.
    // For now, using the banner-style showErrorMessage.
    console.error(`ERROR: ${title} - ${message}`);
    showErrorMessage(`${title}: ${message}`);
}


function renderProbabilityChart(probabilities) {
    // Deprecated, use ChartUtils
}
// ... The rest of your utility functions ...
function getStatusMessage(status) {
    // Deprecated
    return '';
}