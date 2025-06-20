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

// --- Global Variables & State ---
let probabilityChart = null; // To hold the chart instance
let tableSortState = { column: 'timestamp', direction: 'desc' }; // 唯一状态源

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
    progressDiv.id = `progress-item-${job.job_id}`;
    
    // 直接使用传入的 model_name
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
    const progressItem = document.getElementById(`progress-item-${jobId}`);
    if (!progressItem) {
        console.warn(`⚠️ 未找到ID为 progress-item-${jobId} 的进度条元素`);
        return;
    }
    
    const status = progressData.status;
    const finalAccuracy = progressData.final_accuracy;
    const isNewRecord = progressData.is_new_record;
    
    // 【关键修正】从 progressData.progress 读取嵌套的进度对象，并做空值检查
    const progressDetails = progressData.progress || {};
    
    // 更新状态标签
    const statusElement = progressItem.querySelector('.progress-status');
    let statusText = getStatusText(status);
    
    if (status === 'completed') {
        const acc = finalAccuracy || progressDetails.best_accuracy || 0;
        statusText = isNewRecord ? `🎉 新纪录: ${acc.toFixed(4)}` : `完成: ${acc.toFixed(4)}`;
        statusElement.className = isNewRecord ? 'progress-status status-new-record' : 'progress-status status-completed';
    } else {
        statusElement.className = `progress-status status-${status}`;
    }
    statusElement.textContent = statusText;

    // 更新进度条填充和百分比
    const fill = progressItem.querySelector('.progress-fill');
    const percentageElement = progressItem.querySelector('.progress-percentage');
    const percentage = progressDetails.percentage || 0;
    fill.style.width = `${percentage}%`;
    percentageElement.textContent = `${Math.round(percentage)}%`;

    // 更新详细指标
    progressItem.querySelector('.epoch').textContent = `${progressDetails.current_epoch || 0}/${progressDetails.total_epochs || 'N/A'}`;
    progressItem.querySelector('.accuracy').textContent = (progressDetails.accuracy || 0).toFixed(4);
    progressItem.querySelector('.loss').textContent = (progressDetails.loss || 0).toFixed(4);
    progressItem.querySelector('.best-accuracy').textContent = (progressDetails.best_accuracy || 0).toFixed(4);
    progressItem.querySelector('.speed').textContent = `${Math.round(progressDetails.samples_per_second || 0)} samples/s`;
    
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

// 【删除】重复的Canvas函数 - 这些函数已在canvas.js中正确实现
// clearCanvas, updateBrushSize, getCanvasImageData 都应该从canvas.js导入使用

// 【删除】有问题的isCanvasEmpty函数，改用canvas.js模块的isEmpty()实现
// 此函数使用了未定义的ctx和canvas变量，已被canvas.js模块的正确实现替代


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
    // 【修正】画布状态检查现在由Canvas模块回调触发，这里只检查模型选择
    
    predictBtn.disabled = !modelSelected;
    
    if (!modelSelected) {
        predictBtn.textContent = '请选择模型';
        } else {
        predictBtn.textContent = '开始识别';
    }
}

// =================================================================
// 历史记录表格 (History Table) - 新架构
// =================================================================

/**
 * 【步骤3】初始化历史记录表的静态部分（表头和事件监听）。
 * 此函数在页面加载时只调用一次。
 */
export function initializeHistoryTable() {
    const tableHeader = document.querySelector('.history-table thead');
    if (!tableHeader) {
        console.error('❌ 找不到历史表格的表头 an element');
        return;
    }

    tableHeader.innerHTML = `
        <tr>
            <th data-sort-key="model_name">模型名称</th>
            <th data-sort-key="timestamp">完成时间</th>
            <th data-sort-key="best_accuracy">最佳准确率</th>
            <th data-sort-key="duration_seconds">训练时长</th>
            <th data-sort-key="status">状态</th>
            <th>操作</th>
            </tr>
        `;

    // 【关键】使用事件委托，在父元素上监听一次点击事件
    tableHeader.addEventListener('click', (event) => {
        const headerCell = event.target.closest('th');
        if (headerCell && headerCell.dataset.sortKey) {
            handleSortClick(headerCell.dataset.sortKey);
        }
    });
}

/**
 * 【步骤4】处理表头点击事件，更新排序状态并重新渲染。
 * @param {string} sortKey - 被点击的列的key。
 */
function handleSortClick(sortKey) {
    if (tableSortState.column === sortKey) {
        // 如果点击的是当前排序列，则切换排序方向
        tableSortState.direction = tableSortState.direction === 'asc' ? 'desc' : 'asc';
            } else {
        // 否则，切换到新列并默认降序
        tableSortState.column = sortKey;
        tableSortState.direction = 'desc';
    }
    // 根据新状态重新渲染表格
    renderHistoryTable();
}

/**
 * 【步骤2】独立的排序函数。
 * @param {Array} historyData - 原始训练历史数据。
 * @returns {Array} - 返回一个已排序的新的数据副本。
 */
function getSortedHistory(historyData) {
    // 创建一个副本以避免修改原始数据
    const dataCopy = [...historyData];
    const { column, direction } = tableSortState;

    dataCopy.sort((a, b) => {
        let valA = a[column];
        let valB = b[column];

        // 对特定列进行特殊处理
        if (column === 'timestamp') {
            valA = new Date(valA).getTime();
            valB = new Date(valB).getTime();
        }

        if (valA < valB) {
            return direction === 'asc' ? -1 : 1;
        }
        if (valA > valB) {
            return direction === 'asc' ? 1 : -1;
        }
        return 0;
    });

    return dataCopy;
}

/**
 * 【步骤5】核心渲染函数，只负责根据全局状态渲染UI。
 */
export function renderHistoryTable() {
    const historyData = window.AppState.trainingHistory || [];
    const tableBody = document.getElementById('history-table-body');
    const tableHeader = document.querySelector('.history-table thead');

    if (!tableBody || !tableHeader) return;

    // 1. 获取排序后的数据
    const sortedData = getSortedHistory(historyData);

    // 2. 更新表头视觉指示器
    tableHeader.querySelectorAll('th').forEach(th => {
        th.classList.remove('sorted-asc', 'sorted-desc');
        if (th.dataset.sortKey === tableSortState.column) {
            th.classList.add(tableSortState.direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
        }
    });

    // 3. 渲染表体
    tableBody.innerHTML = ''; // 清空现有内容
    if (sortedData.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="6" class="empty-state">还没有任何训练记录。</td></tr>`;
        return;
    }
    
    sortedData.forEach(record => {
        const row = document.createElement('tr');
        row.className = 'history-main-row';
        row.innerHTML = `
            <td>${record.model_name || 'N/A'}</td>
            <td>${formatDate(record.timestamp)}</td>
            <td class="accuracy-cell">${(record.best_accuracy * 100).toFixed(2)}%</td>
            <td>${formatDuration(record.duration_seconds)}</td>
            <td><span class="status-badge status-${record.status}">${record.status}</span></td>
            <td>
                <button class="btn-details" data-job-id="${record.job_id}">详情</button>
                <button class="btn-compare ${window.AppState.comparisonSet.has(record.job_id) ? 'added' : ''}" data-job-id="${record.job_id}">
                    ${window.AppState.comparisonSet.has(record.job_id) ? '✓ 已添加' : '添加到对比'}
                </button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

export function showDetailsModal(jobId) {
    const modal = document.getElementById('details-modal');
    const modalBody = document.getElementById('modal-body');

    // 从全局状态中查找完整的历史记录
    const record = window.AppState.trainingHistory.find(r => r.job_id === jobId);

    if (!record) {
        console.error(`❌ 未找到 Job ID 为 ${jobId} 的历史记录`);
        showError("无法加载详情", `未找到任务ID为 ${jobId} 的记录。`);
        return;
    }

    // 格式化函数
    const formatValue = (value, fallback = 'N/A') => value !== undefined && value !== null ? value : fallback;
    const formatPercent = (value) => typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : 'N/A';
    const formatInt = (value) => typeof value === 'number' ? Math.round(value) : 'N/A';
    const formatParams = (params) => typeof params === 'number' ? params.toLocaleString() : 'N/A';
    
    // 构建弹窗内容HTML
    modalBody.innerHTML = `
        <div class="modal-header">
            <h3 class="modal-title">训练详情: ${record.model_name}</h3>
            <button id="modal-close-btn-inner" class="modal-close-btn">&times;</button>
        </div>
        <!-- 核心指标卡片横向排列 -->
        <div class="metrics-cards-container">
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-content">
                    <div class="metric-label">最佳准确率</div>
                    <div class="metric-value highlight">${formatPercent(record.best_accuracy)}</div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-content">
                    <div class="metric-label">最终损失</div>
                    <div class="metric-value">${formatValue(record.final_val_loss?.toFixed(4), 'N/A')}</div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">⏱️</div>
                <div class="metric-content">
                    <div class="metric-label">训练时长</div>
                    <div class="metric-value">${formatDuration(record.duration_seconds)}</div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">⚡</div>
                <div class="metric-content">
                    <div class="metric-label">训练速度</div>
                    <div class="metric-value">${formatInt(record.samples_per_second)} samples/s</div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">🔧</div>
                <div class="metric-content">
                    <div class="metric-label">模型参数</div>
                    <div class="metric-value">${formatParams(record.hyperparameters_extended?.model_architecture?.total_parameters)}</div>
                </div>
            </div>
        </div>

        <!-- 中间部分：图表和信息 -->
        <div class="details-main-content">
            <div class="chart-section">
                <div class="chart-tabs">
                    <button class="chart-tab-btn active" data-tab="loss">📉 训练损失</button>
                    <button class="chart-tab-btn" data-tab="accuracy">📈 准确率</button>
                </div>
                <div class="chart-tab-content active" id="loss-tab">
                    <div class="chart-container">
                        <canvas id="lossChart"></canvas>
                    </div>
                </div>
                <div class="chart-tab-content" id="accuracy-tab">
                    <div class="chart-container">
                        <canvas id="accuracyChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="info-sidebar">
                <div class="info-section">
                    <h4>⚙️ 超参数配置</h4>
                    <div class="info-grid">
                        <div><strong>Epochs:</strong> <span>${formatValue(record.config?.epochs)}</span></div>
                        <div><strong>学习率:</strong> <span>${formatValue(record.config?.learning_rate)}</span></div>
                        <div><strong>批次大小:</strong> <span>${formatValue(record.config?.batch_size)}</span></div>
                    </div>
                </div>
                
                <div class="info-section">
                    <h4>📋 任务信息</h4>
                    <div class="info-grid">
                        <div><strong>任务ID:</strong> <span class="job-id-span" title="${record.job_id}">${record.job_id.substring(0, 15)}...</span></div>
                        <div><strong>完成时间:</strong> <span>${formatDate(record.timestamp)}</span></div>
                        <div><strong>状态:</strong> <span class="status-badge status-${record.status}">${record.status}</span></div>
                    </div>
                </div>
            </div>
        </div>
        <!-- 详细数据表格 - 可折叠 -->
        <div class="epoch-table-section">
            <div class="section-header" onclick="toggleEpochTable()">
                <h4>📈 各轮次详细数据</h4>
                <button class="collapse-btn" id="epoch-collapse-btn">
                    <span class="collapse-icon">▼</span>
                </button>
            </div>
            <div class="epoch-table-container" id="epoch-table-container">
                <table class="details-epoch-table">
                    <thead>
                        <tr>
                            <th>Epoch</th>
                            <th>训练损失</th>
                            <th>训练准确率</th>
                            <th>验证损失</th>
                            <th>验证准确率</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${(record.epoch_metrics || []).map(epoch => `
                            <tr>
                                <td>${epoch.epoch}</td>
                                <td>${formatValue(epoch.loss?.toFixed(4))}</td>
                                <td>${formatPercent(epoch.accuracy)}</td>
                                <td>${formatValue(epoch.val_loss?.toFixed(4), 'N/A')}</td>
                                <td>${formatPercent(epoch.val_accuracy)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;

    // 显示弹窗
    modal.style.display = 'flex';
    modal.style.opacity = '1';
    modal.classList.add('visible');

    // 渲染图表 (在DOM元素可见后渲染)
    if (record.epoch_metrics && record.epoch_metrics.length > 0) {
        // 默认只渲染损失图表
        ChartUtils.renderLossChart('lossChart', record.epoch_metrics);
        
        // 标记准确率图表为未渲染状态
        modal.setAttribute('data-accuracy-rendered', 'false');
    }

    // 添加标签页切换事件监听
    const tabButtons = modal.querySelectorAll('.chart-tab-btn');
    tabButtons.forEach(button => {
        button.addEventListener('click', (e) => switchChartTab(e.target.dataset.tab, record));
    });

    // 添加关闭事件监听
    document.getElementById('modal-close-btn-inner').addEventListener('click', hideDetailsModal);
    
    // 添加全局折叠函数
    window.toggleEpochTable = function() {
        const container = document.getElementById('epoch-table-container');
        const icon = document.querySelector('.collapse-icon');
        
        if (container.style.display === 'none') {
            container.style.display = 'block';
            icon.textContent = '▼';
        } else {
            container.style.display = 'none';
            icon.textContent = '▶';
        }
    };
}

// 标签页切换函数
function switchChartTab(tabName, record) {
    const modal = document.getElementById('details-modal');
    
    // 切换标签按钮状态
    const tabButtons = modal.querySelectorAll('.chart-tab-btn');
    const tabContents = modal.querySelectorAll('.chart-tab-content');
    
    tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });
    
    // 延迟渲染准确率图表（只在第一次切换时渲染）
    if (tabName === 'accuracy' && modal.getAttribute('data-accuracy-rendered') === 'false') {
        if (record.epoch_metrics && record.epoch_metrics.length > 0) {
            // 使用 setTimeout 确保DOM已完全更新
            setTimeout(() => {
                ChartUtils.renderAccuracyChart('accuracyChart', record.epoch_metrics);
                modal.setAttribute('data-accuracy-rendered', 'true');
            }, 50);
        }
    }
}

export function hideDetailsModal() {
    const modal = document.getElementById('details-modal');
    if (modal) {
        modal.style.opacity = '0';
        modal.classList.remove('visible');
        const modalContent = modal.querySelector('.modal-content');
        if (modalContent) {
            modalContent.style.transform = 'scale(0.95)';
        }
        setTimeout(() => {
            modal.style.display = 'none';
            document.getElementById('modal-body').innerHTML = ''; // 清空内容
        }, 300);
    }
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

export function renderHyperparameterComparison(modelsData) {
    const container = document.getElementById('hyperparameter-comparison');
    if (!container) {
        console.warn('超参数对比容器不存在');
        return;
    }

    const html = `
        <div class="comparison-section">
            <h3>📊 超参数对比</h3>
            <div class="hyperparameter-table-container">
                ${generateHyperparameterTable(modelsData)}
            </div>
        </div>
        <div class="comparison-section">
            <h3>📈 性能指标对比</h3>
            <div class="performance-table-container">
                ${generatePerformanceTable(modelsData)}
            </div>
        </div>
        <div class="comparison-section">
            <h3>🔧 环境信息对比</h3>
            <div class="environment-table-container">
                ${generateEnvironmentTable(modelsData)}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function generateHyperparameterTable(modelsData) {
    const headers = ['模型', '学习率', '批次大小', '训练轮数', '优化器', '总参数', '可训练参数'];
    
    const rows = modelsData.map(model => {
        const config = model.config || {};
        const hyperparams = model.hyperparameters_extended || {};
        const basic = hyperparams.basic || config;
        const optimizer = hyperparams.optimizer || {};
        const architecture = hyperparams.model_architecture || {};
        
        return [
            model.model_name || model.model_id,
            basic.learning_rate || '未知',
            basic.batch_size || '未知',
            basic.epochs || '未知',
            optimizer.type || 'Adam',
            formatNumber(architecture.total_parameters || model.model_params || 0),
            formatNumber(architecture.trainable_parameters || model.trainable_params || model.model_params || 0)
        ];
    });
    
    return generateComparisonTable(headers, rows);
}

function generatePerformanceTable(modelsData) {
    const headers = ['模型', '最佳准确率', '最终训练损失', '最终验证损失', '训练时间(秒)', '样本/秒', '收敛轮数'];
    
    const rows = modelsData.map(model => {
        const stability = model.stability_metrics || {};
        
        return [
            model.model_name || model.model_id,
            `${(model.best_accuracy * 100).toFixed(2)}%`,
            (model.final_train_loss || model.final_loss || 0).toFixed(4),
            (model.final_val_loss || model.final_train_loss || model.final_loss || 0).toFixed(4),
            (model.duration_seconds || 0).toFixed(1),
            Math.round(model.samples_per_second || 0),
            stability.convergence_epoch || '未知'
        ];
    });
    
    return generateComparisonTable(headers, rows);
}

function generateEnvironmentTable(modelsData) {
    const headers = ['模型', '设备', 'PyTorch版本', 'GPU型号', 'GPU内存(GB)', '训练稳定性'];
    
    const rows = modelsData.map(model => {
        const env = model.environment_info || {};
        const stability = model.stability_metrics || {};
        const valAccStd = stability.val_accuracy_std;
        const stabilityText = valAccStd !== undefined ? 
            `${(valAccStd * 100).toFixed(2)}% (${valAccStd < 0.01 ? '很稳定' : valAccStd < 0.05 ? '稳定' : '一般'})` : 
            '未知';
        
        return [
            model.model_name || model.model_id,
            env.device || '未知',
            env.pytorch_version || '未知',
            env.gpu_name || 'CPU',
            env.gpu_memory_gb || '0',
            stabilityText
        ];
    });
    
    return generateComparisonTable(headers, rows);
}

function generateComparisonTable(headers, rows) {
    const headerRow = headers.map(h => `<th>${h}</th>`).join('');
    const bodyRows = rows.map(row => 
        `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`
    ).join('');
    
    return `
        <table class="comparison-table">
            <thead>
                <tr>${headerRow}</tr>
            </thead>
            <tbody>
                ${bodyRows}
            </tbody>
        </table>
    `;
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

/**
 * 根据从后端获取的已训练模型列表，动态更新预测页面的下拉选择框
 * @param {Array<Object>} trainedModels - 已训练模型信息数组
 */
export function updatePredictionModelDropdown(trainedModels) {
    const select = document.getElementById('prediction-model-select');
    if (!select) return;

    select.innerHTML = ''; // 清空现有选项

    if (!trainedModels || trainedModels.length === 0) {
        const option = document.createElement('option');
        option.textContent = '暂无可用模型，请先训练';
        option.disabled = true;
        select.appendChild(option);
    } else {
        trainedModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.filename;
            // 将 model_id 存储在 dataset 中，方便预测时获取
            option.dataset.modelId = model.model_id; 
            option.textContent = `${model.display_name} (准确率: ${(model.accuracy * 100).toFixed(2)}%)`;
            select.appendChild(option);
        });
    }
    // 更新预测按钮的状态
    updatePredictButtonState();
}


/**
 * 根据模型ID查找其显示名称
 */
export function getModelDisplayName(modelId) {
    const select = document.getElementById('prediction-model-select');
    if (!select) return '未知模型';

    const option = select.querySelector(`option[value="${modelId}"]`);
    return option ? option.textContent : '未知模型';
}