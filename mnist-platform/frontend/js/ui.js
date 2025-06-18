import * as ChartUtils from './chart_utils.js';
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
function getModelName(modelId) {
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
    
    // 更新性能指标
    if (progress.samples_per_sec !== undefined && speedElement) {
        speedElement.textContent = `${Math.round(progress.samples_per_sec)} samples/s`;
    }
    
    if (progress.learning_rate !== undefined && learningRateElement) {
        learningRateElement.textContent = progress.learning_rate.toFixed(6);
    }
}

// 获取状态显示文本
function getStatusText(status) {
    const statusMap = {
        'queued': '排队中',
        'running': '训练中',
        'completed': '已完成',
        'failed': '训练失败'
    };
    return statusMap[status] || status;
}

console.log('📱 UI 模块已加载');

// ==================== 手写识别 Canvas 绘制功能 ====================

// Canvas 绘制状态
let canvasState = {
    isDrawing: false,
    lastX: 0,
    lastY: 0,
    brushSize: 14,
    canvas: null,
    ctx: null,
    updateTimer: null  // 添加防抖计时器
};

// 初始化 Canvas
export function initializeCanvas() {
    const canvas = document.getElementById('drawing-canvas');
    if (!canvas) {
        console.error('❌ 未找到绘制画布');
        return false;
    }
    
    canvasState.canvas = canvas;
    canvasState.ctx = canvas.getContext('2d');
    
    // 设置 Canvas 绘制样式
    canvasState.ctx.strokeStyle = '#ffffff';
    canvasState.ctx.lineWidth = canvasState.brushSize;
    canvasState.ctx.lineCap = 'round';
    canvasState.ctx.lineJoin = 'round';
    
    // 清空画布
    clearCanvas();
    
    // 设置事件监听器
    setupCanvasDrawing();
    
    console.log('🎨 Canvas 初始化完成');
    return true;
}

// 设置 Canvas 绘制事件
export function setupCanvasDrawing() {
    const canvas = canvasState.canvas;
    if (!canvas) return;
    
    // 鼠标事件
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // 触摸事件（移动端支持）
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // 防止页面滚动
    canvas.addEventListener('touchstart', e => e.preventDefault());
    canvas.addEventListener('touchmove', e => e.preventDefault());
}

// 开始绘制
function startDrawing(e) {
    canvasState.isDrawing = true;
    const coords = getCoordinates(e);
    canvasState.lastX = coords.x;
    canvasState.lastY = coords.y;
    
    // 立即更新按钮状态（开始绘制时）
    setTimeout(() => updatePredictButtonState(), 10);
}

// 绘制过程
function draw(e) {
    if (!canvasState.isDrawing) return;
    
    const coords = getCoordinates(e);
    const ctx = canvasState.ctx;
    
    ctx.beginPath();
    ctx.moveTo(canvasState.lastX, canvasState.lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    
    canvasState.lastX = coords.x;
    canvasState.lastY = coords.y;
    
    // 绘制过程中更新按钮状态（防抖）
    if (!canvasState.updateTimer) {
        canvasState.updateTimer = setTimeout(() => {
            updatePredictButtonState();
            canvasState.updateTimer = null;
        }, 50);
    }
}

// 停止绘制
function stopDrawing() {
    if (canvasState.isDrawing) {
        canvasState.isDrawing = false;
        
        // 绘制完成后立即更新按钮状态
        setTimeout(() => updatePredictButtonState(), 10);
    }
}

// 处理触摸事件
function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                     e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvasState.canvas.dispatchEvent(mouseEvent);
}

// 获取鼠标/触摸坐标
function getCoordinates(e) {
    const rect = canvasState.canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

// 清除画布
export function clearCanvas() {
    if (!canvasState.ctx) return;
    
    canvasState.ctx.fillStyle = '#000000';
    canvasState.ctx.fillRect(0, 0, canvasState.canvas.width, canvasState.canvas.height);
    
    // 清除预测结果
    const resultContainer = document.getElementById('prediction-result');
    if (resultContainer) {
        showEmptyResult();
    }
    
    console.log('🧹 画布已清除');
}

// 更新画笔大小
export function updateBrushSize(size) {
    canvasState.brushSize = size;
    if (canvasState.ctx) {
        canvasState.ctx.lineWidth = size;
    }
    
    // 更新显示值
    const valueElement = document.getElementById('brush-size-value');
    if (valueElement) {
        valueElement.textContent = size + 'px';
    }
}

// 获取 Canvas 图像数据
export function getCanvasImageData() {
    if (!canvasState.canvas) {
        console.error('❌ Canvas 未初始化');
        return null;
    }
    
    try {
        // 获取 Canvas 的 base64 数据
        const imageData = canvasState.canvas.toDataURL('image/png');
        console.log('📷 已获取 Canvas 图像数据');
        return imageData;
    } catch (error) {
        console.error('❌ 获取图像数据失败:', error);
        return null;
    }
}

// 检查画布是否为空
export function isCanvasEmpty() {
    if (!canvasState.canvas) return true;
    
    const ctx = canvasState.ctx;
    const imageData = ctx.getImageData(0, 0, canvasState.canvas.width, canvasState.canvas.height);
    
    // 检查是否所有像素都是黑色 (RGB = 0,0,0)
    for (let i = 0; i < imageData.data.length; i += 4) {
        // 检查 RGB 值，如果任何一个不是 0，说明有绘制内容
        if (imageData.data[i] > 0 || imageData.data[i + 1] > 0 || imageData.data[i + 2] > 0) {
            return false;
        }
    }
    return true;
}

// 渲染预测结果
export function renderPredictionResult(result) {
    const container = document.getElementById('prediction-result');
    if (!container) {
        console.error('❌ 未找到预测结果容器');
        return;
    }
    
    const prediction = result.prediction;
    const probabilities = result.probabilities;
    const confidence = Math.max(...probabilities);
    
    container.innerHTML = `
        <div class="prediction-display">
            <div class="predicted-digit">${prediction}</div>
            <div class="confidence-score">置信度: ${(confidence * 100).toFixed(1)}%</div>
        </div>
        
        <div class="probabilities-container">
            <div class="probabilities-title">各数字概率分布</div>
            <div class="probability-bars">
                ${probabilities.map((prob, index) => `
                    <div class="probability-item">
                        <div class="probability-digit">${index}</div>
                        <div class="probability-bar">
                            <div class="probability-fill ${prob === confidence ? 'highest' : ''}" 
                                 style="height: ${prob * 100}%"></div>
                        </div>
                        <div class="probability-value">${(prob * 100).toFixed(1)}%</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    console.log(`🎯 预测结果已显示: ${prediction} (置信度: ${(confidence * 100).toFixed(1)}%)`);
}

// 显示加载状态
export function showPredictionLoading() {
    const container = document.getElementById('prediction-result');
    if (!container) return;
    
    container.innerHTML = `
        <div class="empty-result">
            <div class="loading-spinner"></div>
            <div class="empty-result-text">正在识别中...</div>
            <div class="empty-result-hint">请稍候</div>
        </div>
    `;
}

// 显示空结果状态
export function showEmptyResult() {
    const container = document.getElementById('prediction-result');
    if (!container) return;
    
    container.innerHTML = `
        <div class="empty-result">
            <div class="empty-result-icon">✏️</div>
            <div class="empty-result-text">请在左侧画布上绘制数字</div>
            <div class="empty-result-hint">画完后点击"识别"按钮</div>
        </div>
    `;
}

// 渲染已训练模型选择器
export function renderTrainedModels(models) {
    const select = document.getElementById('prediction-model-select');
    if (!select) {
        console.error('❌ 未找到模型选择器');
        return;
    }
    
    // 清空现有选项
    select.innerHTML = '<option value="">请选择已训练的模型</option>';
    
    if (models.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = '暂无已训练的模型';
        option.disabled = true;
        select.appendChild(option);
        return;
    }
    
    // 添加模型选项
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = `${model.name} (准确率: ${(model.accuracy * 100).toFixed(2)}%)`;
        select.appendChild(option);
    });
    
    console.log('📋 已更新模型选择器:', models.length, '个模型');
}

// 更新预测按钮状态
export function updatePredictButtonState() {
    const predictBtn = document.getElementById('predict-btn');
    const modelSelect = document.getElementById('prediction-model-select');
    
    if (!predictBtn || !modelSelect) return;
    
    const hasModel = modelSelect.value !== '';
    const hasDrawing = !isCanvasEmpty();
    
    predictBtn.disabled = !hasModel || !hasDrawing;
    
    // 更新按钮文本
    if (!hasModel) {
        predictBtn.textContent = '请选择模型';
    } else if (!hasDrawing) {
        predictBtn.textContent = '请先绘制数字';
    } else {
        predictBtn.textContent = '🔍 识别';
    }
}

console.log('🎨 Canvas 绘制模块已加载');

// ==================== 训练结果页面 ====================

// 存储当前排序状态
let historySortState = {
    column: 'completion_time',
    direction: 'desc'
};

/**
 * 渲染训练历史记录表格
 * @param {Array} historyData - 从API获取的训练历史数组
 */
export function renderHistoryTable(historyData) {
    const container = document.getElementById('history-table-container');
    if (!container) {
        console.error('❌ 未找到历史记录表格容器');
        return;
    }

    if (!historyData || historyData.length === 0) {
        container.innerHTML = `<div class="empty-state">暂无训练历史记录</div>`;
        return;
    }
    
    // 根据当前状态排序数据
    const sortedData = sortHistoryData(historyData, historySortState.column, historySortState.direction);

    // 创建表格结构
    const table = document.createElement('table');
    table.className = 'history-table';
    
    // 创建表头
    table.innerHTML = `
        <thead>
            <tr>
                ${createHeaderCell('model_name', '模型名称')}
                ${createHeaderCell('final_accuracy', '最终准确率')}
                ${createHeaderCell('training_duration_sec', '训练耗时(秒)')}
                ${createHeaderCell('epochs', '轮数')}
                ${createHeaderCell('learning_rate', '学习率')}
                ${createHeaderCell('batch_size', '批次大小')}
                ${createHeaderCell('completion_time', '完成时间')}
            </tr>
        </thead>
    `;

    // 创建表格内容
    const tbody = document.createElement('tbody');
    sortedData.forEach(record => {
        const tr = document.createElement('tr');
        const attentionBadge = record.has_attention ? '<span class="attention-badge-small">⚡</span>' : '';
        tr.innerHTML = `
            <td>${record.model_name || getModelName(record.model_id)} ${attentionBadge}</td>
            <td>${(record.metrics.final_accuracy * 100).toFixed(2)}%</td>
            <td>${record.metrics.training_duration_sec.toFixed(1)}</td>
            <td>${record.hyperparameters.epochs}</td>
            <td>${record.hyperparameters.learning_rate}</td>
            <td>${record.hyperparameters.batch_size}</td>
            <td>${formatDate(record.completion_time)}</td>
        `;
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    // 渲染表格
    container.innerHTML = '';
    container.appendChild(table);
    
    // 绑定表头点击事件
    const headers = container.querySelectorAll('th[data-sort-key]');
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const sortKey = header.dataset.sortKey;
            handleSort(sortKey, historyData);
        });
    });

    console.log('📈 已渲染训练历史表格');
}

// 创建可排序的表头单元格
function createHeaderCell(key, title) {
    const isSorted = historySortState.column === key;
    const sortIcon = isSorted ? (historySortState.direction === 'asc' ? '▲' : '▼') : '↕';
    return `<th data-sort-key="${key}" class="${isSorted ? 'sorted' : ''}">${title} <span class="sort-icon">${sortIcon}</span></th>`;
}

// 处理排序逻辑
function handleSort(sortKey, historyData) {
    if (historySortState.column === sortKey) {
        // 切换排序方向
        historySortState.direction = historySortState.direction === 'asc' ? 'desc' : 'asc';
    } else {
        // 新的排序列，默认降序
        historySortState.column = sortKey;
        historySortState.direction = 'desc';
    }
    // 重新渲染表格
    renderHistoryTable(historyData);
}

// 排序数据
function sortHistoryData(data, column, direction) {
    return [...data].sort((a, b) => {
        let valA, valB;

        // 根据不同列获取值
        switch (column) {
            case 'model_name':
                valA = a.model_name || getModelName(a.model_id);
                valB = b.model_name || getModelName(b.model_id);
                break;
            case 'final_accuracy':
                valA = a.metrics.final_accuracy;
                valB = b.metrics.final_accuracy;
                break;
            case 'training_duration_sec':
                valA = a.metrics.training_duration_sec;
                valB = b.metrics.training_duration_sec;
                break;
            case 'epochs':
                valA = a.hyperparameters.epochs;
                valB = b.hyperparameters.epochs;
                break;
            case 'learning_rate':
                valA = a.hyperparameters.learning_rate;
                valB = b.hyperparameters.learning_rate;
                break;
            case 'batch_size':
                valA = a.hyperparameters.batch_size;
                valB = b.hyperparameters.batch_size;
                break;
            case 'completion_time':
                valA = new Date(a.completion_time).getTime();
                valB = new Date(b.completion_time).getTime();
                break;
            default:
                return 0;
        }

        // 执行比较
        if (valA < valB) {
            return direction === 'asc' ? -1 : 1;
        }
        if (valA > valB) {
            return direction === 'asc' ? 1 : -1;
        }
        return 0;
    });
}

// 格式化日期
function formatDate(dateString) {
    const date = new Date(dateString);
    if (isNaN(date)) return 'N/A';
    
    // 补零函数
    const pad = (num) => num.toString().padStart(2, '0');
    
    const year = date.getFullYear();
    const month = pad(date.getMonth() + 1);
    const day = pad(date.getDate());
    const hours = pad(date.getHours());
    const minutes = pad(date.getMinutes());
    
    return `${year}-${month}-${day} ${hours}:${minutes}`;
}

// ==================== 模型对比页面 ====================
/**
 * 渲染模型对比图表
 * @param {object} processedData - 经过处理用于图表的数据
 */
export function renderComparisonCharts(processedData) {
    if (!processedData || !processedData.labels || processedData.labels.length === 0) {
        const container = document.getElementById('comparison-charts-container');
        if (container) {
            container.innerHTML = `<div class="empty-state">没有可用于对比的数据。请至少训练不同类型的模型各一次。</div>`;
        }
        return;
    }
    
    // 渲染雷达图
    ChartUtils.createRadarChart('radarChart', processedData.radarData);

    // 渲染准确率柱状图
    ChartUtils.createBarChart('accuracyBarChart', processedData.barData.accuracies, '最高准确率 (%)');

    // 渲染速度柱状图
    ChartUtils.createBarChart('speedBarChart', processedData.barData.speeds, '训练耗时 (秒)');

    // 渲染参数量柱状图
    ChartUtils.createBarChart('paramsBarChart', processedData.barData.params, '模型参数量');

    console.log('📊 已渲染所有对比图表');
}