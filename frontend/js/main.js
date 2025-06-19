// 主应用逻辑 - 状态管理、事件绑定、模块协调
import * as API from './api.js';
import * as UI from './ui.js';
import * as ChartUtils from './chart_utils.js';

// =================================================================
// 1. 全局状态和初始化
// =================================================================

const AppState = {
    availableModels: [],
    currentTrainingJobs: {},
    pollingIntervalId: null,
    trainedModels: [],
    trainingHistory: [],
    activeTab: 'model-training',
    selectedModels: new Set(),
    comparisonSet: new Set()
};

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 MNIST 智能分析平台启动');
    initTabNavigation();
    initModelTrainingPage();
    initEventListeners();
    initGlobalFunctions();
    initModalControls();
    console.log('✅ 应用初始化完成');
});

function initGlobalFunctions() {
    window.updateSliderValue = function(paramName, value) {
        const valueElement = document.getElementById(`${paramName}-value`);
        if (valueElement) {
            valueElement.textContent = value;
        }
    };

    // 将新的模态框函数暴露到全局
    window.Module = {
        showDetailsModal: UI.showDetailsModal
    };
}

// =================================================================
// 2. 页面导航和事件处理
// =================================================================

function initTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === targetTab) {
                    content.classList.add('active');
                }
            });
            
            AppState.activeTab = targetTab;
            handleTabSwitch(targetTab);
        });
    });
}

function initEventListeners() {
    // 统一的点击事件指挥中心
    document.addEventListener('click', (e) => {
        const target = e.target;
        const targetId = target.id;
        const targetClassList = target.classList;

        if (targetId === 'start-training-btn') {
            handleStartTraining();
        } else if (targetId === 'clear-canvas-btn') {
            UI.clearCanvas();
        } else if (targetId === 'predict-btn') {
            handlePrediction();
        } else if (targetId === 'refresh-comparison-btn') {
            displayComparisonCharts();
        } else if (targetId === 'compare-now-btn') {
            document.querySelector('.tab-btn[data-tab="model-comparison"]').click();
        } else if (targetClassList.contains('btn-compare')) {
            const jobId = target.dataset.jobId;
            handleToggleComparison(jobId, target);
        }
    });

    // 统一的变更事件监听器
    document.addEventListener('change', (e) => {
        const target = e.target;
        if (target.classList.contains('model-checkbox')) {
            handleModelSelection(target);
        }
        if (target.id === 'prediction-model-select') {
            UI.updatePredictButtonState();
        }
    });
}

async function handleTabSwitch(tabId) {
    switch (tabId) {
        case 'handwriting-recognition':
            await initHandwritingRecognition();
            break;
        case 'training-results':
            await loadTrainingHistory();
            break;
        case 'model-comparison':
            await initComparisonTab();
            break;
    }
}

function handleModelSelection(checkbox) {
    const modelId = checkbox.value;
    const startTrainingBtn = document.getElementById('start-training-btn');
    
    if (checkbox.checked) {
        AppState.selectedModels.add(modelId);
    } else {
        AppState.selectedModels.delete(modelId);
    }
    
    startTrainingBtn.disabled = AppState.selectedModels.size === 0;
    console.log('🎯 已选择模型:', Array.from(AppState.selectedModels));
}

function handleToggleComparison(jobId, button) {
    if (AppState.comparisonSet.has(jobId)) {
        AppState.comparisonSet.delete(jobId);
        button.classList.remove('added');
        button.textContent = '加入对比';
    } else {
        AppState.comparisonSet.add(jobId);
        button.classList.add('added');
        button.textContent = '已添加';
    }
    UI.updateComparisonStatusBar(AppState.comparisonSet.size);
}

// =================================================================
// 3. 模型训练页逻辑
// =================================================================

async function initModelTrainingPage() {
    try {
        AppState.availableModels = await API.getModels();
        UI.renderModelCards(AppState.availableModels);
        console.log('📋 已加载模型列表:', AppState.availableModels);
    } catch (error) {
        console.error('❌ 加载模型列表失败:', error);
        UI.showErrorMessage('加载模型列表失败，请检查后端服务是否正常运行');
    }
}

async function handleStartTraining() {
    if (AppState.selectedModels.size === 0) {
        UI.showErrorMessage('请至少选择一个模型进行训练');
        return;
    }
    
    try {
        const params = getTrainingParameters();
        const trainingConfigs = Array.from(AppState.selectedModels).map(modelId => ({
            id: modelId,
            epochs: params.epochs,
            lr: params.lr,
            batch_size: params.batch_size
        }));
        
        console.log('🔥 开始训练，配置:', trainingConfigs);
        
        const response = await API.startTraining({ models: trainingConfigs });
        
        const jobsWithNames = response.jobs.map(job => {
            const model = AppState.availableModels.find(m => m.id === job.model_id);
            return {
                ...job,
                model_name: model ? model.name : job.model_id
            };
        });
        
        jobsWithNames.forEach(job => {
            AppState.currentTrainingJobs[job.job_id] = {
                model_id: job.model_id,
                status: 'queued'
            };
        });
        
        UI.createTrainingProgressBars(jobsWithNames);
        startProgressPolling();
        document.getElementById('start-training-btn').disabled = true;
        
        console.log('✅ 训练任务已启动:', response.jobs);
        
    } catch (error) {
        console.error('❌ 启动训练失败:', error);
        UI.showErrorMessage('启动训练失败: ' + error.message);
    }
}

function startProgressPolling() {
    if (AppState.pollingIntervalId) {
        clearInterval(AppState.pollingIntervalId);
    }
    
    AppState.pollingIntervalId = setInterval(updateTrainingProgress, 2000);
    console.log('⏰ 已开始进度轮询');
}

async function updateTrainingProgress() {
    const runningJobIds = Object.keys(AppState.currentTrainingJobs).filter(
        jobId => ['queued', 'running'].includes(AppState.currentTrainingJobs[jobId].status)
    );
    
    if (runningJobIds.length === 0) {
        clearInterval(AppState.pollingIntervalId);
        AppState.pollingIntervalId = null;
        document.getElementById('start-training-btn').disabled = AppState.selectedModels.size === 0;
        console.log('🏁 所有训练任务已完成');
        return;
    }
    
    try {
        const progress = await API.getTrainingProgress();
        runningJobIds.forEach(jobId => {
            if (progress.jobs[jobId]) {
                const jobProgress = progress.jobs[jobId];
                UI.updateProgressBar(jobId, jobProgress);
                AppState.currentTrainingJobs[jobId].status = jobProgress.status;
            }
        });
    } catch (error) {
        console.error('❌ 获取训练进度失败:', error);
    }
}

function getTrainingParameters() {
    const epochs = parseInt(document.getElementById('epochs-slider').value, 10);
    const lr = parseFloat(document.getElementById('lr-slider').value);
    const batch_size = parseInt(document.getElementById('batch-size-slider').value, 10);
    return { epochs, lr, batch_size };
}

// =================================================================
// 4. 手写识别页逻辑
// =================================================================

async function initHandwritingRecognition() {
    UI.showEmptyResult();
    UI.initializeCanvas();
    if (AppState.trainedModels.length === 0) {
        await loadTrainedModelsForPrediction();
    } else {
        UI.renderTrainedModels(AppState.trainedModels);
        UI.updatePredictButtonState();
    }
}

async function handlePrediction() {
    if (UI.isCanvasEmpty()) {
        UI.showErrorMessage('请先绘制一个数字');
        return;
    }
    try {
        UI.showPredictionLoading();
        const model_id = document.getElementById('prediction-model-select').value;
        const imageData = UI.getCanvasImageData();
        const result = await API.predict({ model_id, image_base64: imageData });
        UI.renderPredictionResult(result);
    } catch (error) {
        console.error('❌ 预测失败:', error);
        UI.showErrorMessage('预测失败: ' + (error.error || error.message));
        UI.showEmptyResult();
    }
}

// =================================================================
// 5. 数据加载与处理 (结果页 & 对比页)
// =================================================================

async function loadTrainedModelsForPrediction() {
    try {
        UI.showPredictionLoading();
        const models = await API.getTrainedModels();
        AppState.trainedModels = models;
        UI.renderTrainedModels(models);
        UI.updatePredictButtonState();
    } catch (error) {
        console.error('❌ 加载已训练模型失败:', error);
        UI.showErrorMessage('加载已训练模型列表失败，请检查后端服务。');
        UI.showEmptyResult();
    }
}

async function loadTrainingHistory() {
    if (AppState.trainingHistory.length > 0 && !window.forceRefreshHistory) {
        UI.renderHistoryTable(AppState.trainingHistory);
        return;
    }
    try {
        const historyData = await API.getTrainingHistory();
        AppState.trainingHistory = historyData.map(normalizeHistoryRecord);
        UI.renderHistoryTable(AppState.trainingHistory);
        console.log(`📚 已加载并归一化 ${AppState.trainingHistory.length} 条训练历史`);
        window.forceRefreshHistory = false;
    } catch (error) {
        console.error('❌ 加载训练历史失败:', error);
        UI.showErrorMessage('加载训练历史记录失败。');
    }
}

/**
 * 古文翻译机 (Data Normalizer)
 * 将旧版嵌套格式的训练历史记录，转换为新版扁平化格式。
 * @param {object} record - 一条训练记录。
 * @returns {object} - 格式统一的训练记录。
 */
function normalizeHistoryRecord(record) {
    if ('best_accuracy' in record && 'final_loss' in record) {
        return record;
    }

    const newRecord = {};
    const oldMetrics = record.metrics || {};
    const oldHyperparams = record.hyperparameters || {};

    newRecord.job_id = record.job_id;
    newRecord.model_id = record.model_id;
    newRecord.model_name = record.model_name;
    newRecord.status = record.status;
    newRecord.error_message = record.error_message || null;
    
    newRecord.timestamp = record.completion_time || record.start_time;

    newRecord.config = {
        epochs: oldHyperparams.epochs,
        learning_rate: oldHyperparams.learning_rate,
        batch_size: oldHyperparams.batch_size
    };
    
    newRecord.best_accuracy = oldMetrics.final_accuracy || 0;
    newRecord.final_loss = oldMetrics.final_loss || 0;
    newRecord.model_params = oldMetrics.total_params || 0;
    newRecord.duration_seconds = oldMetrics.training_duration_sec || 0;
    
    newRecord.samples_per_second = oldMetrics.samples_per_second || 0;
    newRecord.epoch_metrics = record.epoch_metrics || [];

    console.log(`📜 翻译古文记录: ${record.job_id}`);
    return newRecord;
}

async function initComparisonTab() {
    UI.showLoadingOverlay('正在准备对比数据...');
    try {
        if (AppState.trainingHistory.length === 0) {
            console.log('对比页：首次加载，正在获取训练历史...');
            AppState.trainingHistory = await API.getTrainingHistory();
        }

        if (AppState.comparisonSet.size === 0) {
            console.log('对比页：对比集合为空，显示提示信息。');
            UI.showErrorMessage('请先从"训练结果"页面选择至少一项加入对比。');
            ChartUtils.clearAllCharts();
        } else {
            console.log(`对比页：集合中有 ${AppState.comparisonSet.size} 项，开始渲染图表。`);
            await displayComparisonCharts();
        }
    } catch (error) {
        console.error('❌ 加载对比数据失败:', error);
        UI.showErrorMessage('加载模型对比数据失败。');
    } finally {
        UI.hideLoadingOverlay();
    }
}

async function displayComparisonCharts() {
    if (AppState.comparisonSet.size === 0) {
        UI.showErrorMessage('请先从"训练结果"页面选择要对比的训练记录。');
        ChartUtils.clearAllCharts();
        return;
    }
    
    const filteredHistory = AppState.trainingHistory.filter(record => 
        AppState.comparisonSet.has(record.job_id)
    );

    if (filteredHistory.length === 0) {
        UI.showErrorMessage('所选的训练记录未找到，可能数据已过期，请刷新页面。');
        ChartUtils.clearAllCharts();
        return;
    }

    const processedData = processDataForComparison(filteredHistory);
    UI.renderComparisonCharts(processedData);
}

function normalizeInverseMetric(value, maxValue) {
    if (maxValue === 0) return 1;
    return 1 - (value / maxValue);
}

function prepareRadarData(modelsData) {
    const radarLabels = ['准确率', '速度', '参数量', '稳定性(损失)'];
    const maxAccuracy = Math.max(...modelsData.map(m => m.best_accuracy)) || 1;
    const maxSpeed = Math.max(...modelsData.map(m => m.samples_per_second || 0)) || 1;
    const maxParams = Math.max(...modelsData.map(m => m.model_params)) || 1;
    const maxLoss = Math.max(...modelsData.map(m => m.final_loss)) || 1;
    
    const datasets = modelsData.map(model => {
        const color = ChartUtils.getColorForModel(model.model_id);
        return {
            label: model.model_name || model.model_id,
            data: [
                (model.best_accuracy / maxAccuracy),
                normalizeInverseMetric(model.model_params, maxParams),
                (model.samples_per_second || 0) / maxSpeed,
                normalizeInverseMetric(model.final_loss, maxLoss)
            ].map(v => isNaN(v) ? 0 : v),
            backgroundColor: color.replace('1)', '0.2)'),
            borderColor: color,
        };
    });
    return { labels: radarLabels, datasets };
}

function prepareBarData(modelsData) {
    const labels = modelsData.map(m => m.model_name || m.model_id);
    const data = modelsData.map(m => m.best_accuracy);
    const colors = modelsData.map(m => ChartUtils.getColorForModel(m.model_id));
    return { 
        labels, 
        datasets: [{
            label: '最高准确率',
            data: data,
            backgroundColor: colors
        }]
    };
}

function prepareLineData(modelsData, fullHistory) {
    const allEpochs = [...new Set(fullHistory.flatMap(r => r.epoch_metrics ? r.epoch_metrics.map(m => m.epoch) : []))].sort((a, b) => a - b);
    
    const datasets = modelsData.map(model => {
        const bestRun = model;
        const color = ChartUtils.getColorForModel(model.model_id);

        if (!bestRun.epoch_metrics || bestRun.epoch_metrics.length === 0) {
            return {
                label: model.model_name || model.model_id,
                data: [],
                borderColor: color,
                fill: false,
            };
        }
        
        const dataPoints = allEpochs.map(epoch => {
            const metric = bestRun.epoch_metrics.find(m => m.epoch === epoch);
            return metric ? metric.accuracy : null;
        });

        return {
            label: model.model_name || model.model_id,
            data: dataPoints,
            borderColor: color,
            backgroundColor: color.replace('1)', '0.1)'),
            fill: false,
            tension: 0.1
        };
    });
    return { labels: allEpochs, datasets };
}

function processDataForComparison(historyData) {
    if (!historyData || historyData.length === 0) {
        return null;
    }

    const modelsData = historyData; 

    return {
        radar: prepareRadarData(modelsData),
        bar: prepareBarData(modelsData),
        line: prepareLineData(modelsData, historyData)
    };
}

// 帮助渲染的颜色
const CHART_COLORS = [
    '75, 192, 192', '255, 99, 132', '255, 205, 86', 
    '54, 162, 235', '153, 102, 255', '255, 159, 64'
];

// 新增：模态框关闭事件处理
function initModalControls() {
    const modal = document.getElementById('details-modal');
    const closeBtn = document.getElementById('modal-close-btn');

    if (modal && closeBtn) {
        closeBtn.addEventListener('click', UI.hideDetailsModal);
        modal.addEventListener('click', (e) => {
            if (e.target.id === 'details-modal') {
                UI.hideDetailsModal();
            }
        });
    }
}

// 导出状态（用于调试）
window.AppState = AppState; 