// 主应用逻辑 - 状态管理、事件绑定、模块协调
import * as API from './api.js';
import * as UI from './ui.js';
import * as ChartUtils from './chart_utils.js';
import * as Canvas from './canvas.js';

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
    UI.initializeHistoryTable();
    Canvas.init(() => {
        UI.updatePredictButtonState();
    });
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
            Canvas.clearCanvas();
        } else if (targetId === 'predict-btn') {
            handlePrediction();
        } else if (targetId === 'refresh-comparison-btn') {
            displayComparisonCharts();
        } else if (targetId === 'compare-now-btn') {
            document.querySelector('.tab-btn[data-tab="model-comparison"]').click();
        } else if (targetClassList.contains('btn-compare')) {
            const jobId = target.dataset.jobId;
            handleToggleComparison(jobId, target);
        } else if (targetClassList.contains('btn-details')) {
            const jobId = target.dataset.jobId;
            UI.showDetailsModal(jobId);
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
        
        const response = await API.startTraining({ models: trainingConfigs });
        
        // 把新创建的任务ID注册到全局状态中，以便轮询
        response.jobs.forEach(job => {
            AppState.currentTrainingJobs[job.job_id] = { status: 'queued', model_id: job.model_id };
        });

        const jobsWithNames = response.jobs.map(job => {
            const model = AppState.availableModels.find(m => m.id === job.model_id);
            return {
                ...job,
                model_name: model ? model.name : job.model_id
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
        const response = await API.getTrainingProgress(runningJobIds);
        console.log('📈 收到训练进度:', JSON.stringify(response, null, 2));
        
        // 后端返回 {progress: Array}，需要转换为按job_id索引的对象
        const progressByJobId = {};
        if (response.progress && Array.isArray(response.progress)) {
            response.progress.forEach(jobProgress => {
                if (jobProgress.job_id) {
                    progressByJobId[jobProgress.job_id] = jobProgress;
                }
            });
        }
        
        runningJobIds.forEach(jobId => {
            if (progressByJobId[jobId]) {
                const jobProgress = progressByJobId[jobId];
                UI.updateProgressBar(jobId, jobProgress);
                AppState.currentTrainingJobs[jobId].status = jobProgress.status;
            }
        });
    } catch (error) {
        console.error('❌ 获取训练进度失败:', error);
    }
}

function getTrainingParameters() {
    const epochsElement = document.getElementById('epochs-slider');
    const lrElement = document.getElementById('lr-slider');
    const batchSizeElement = document.getElementById('batch-size-slider');
    
    // 安全检查：如果元素不存在或值为空，使用默认值
    const epochs = epochsElement && epochsElement.value ? parseInt(epochsElement.value, 10) : 10;
    const lr = lrElement && lrElement.value ? parseFloat(lrElement.value) : 0.001;
    const batch_size = batchSizeElement && batchSizeElement.value ? parseInt(batchSizeElement.value, 10) : 64;
    
    return { epochs, lr, batch_size };
}

// =================================================================
// 4. 手写识别页逻辑 (Handwriting Recognition)
// =================================================================

async function initHandwritingRecognition() {
    try {
        const trainedModels = await API.getTrainedModels();
        UI.updatePredictionModelDropdown(trainedModels);
    } catch (error) {
        console.error('❌ 加载已训练模型列表失败:', error);
        UI.showErrorMessage('加载可用模型列表失败，请检查后端服务。');
        UI.updatePredictionModelDropdown([]); // 传入空数组以显示提示信息
    }
}

async function handlePrediction() {
    if (Canvas.isEmpty()) {
        UI.showErrorMessage('请先在画板上写一个数字');
        return;
    }

    const select = document.getElementById('prediction-model-select');
    const selectedOption = select.options[select.selectedIndex];
    
    if (!selectedOption || !selectedOption.value) {
        UI.showErrorMessage('请选择一个有效的模型进行预测');
        return;
    }

    const filename = selectedOption.value;
    const modelId = selectedOption.dataset.modelId;
    const imageBase64 = Canvas.getImageData();

    UI.showLoadingOverlay('正在识别手写数字...');

    try {
        const result = await API.predict(modelId, filename, imageBase64);
        UI.renderPredictionResult({
            prediction: result.predicted_class,
            probabilities: result.probabilities
        });
        console.log('🔍 预测结果:', result);
    } catch (error) {
        console.error('❌ 预测失败:', error);
        UI.showErrorMessage('预测失败: ' + error.message);
    } finally {
        UI.hideLoadingOverlay();
    }
}

// =================================================================
// 5. 训练结果与对比 (Results & Comparison)
// =================================================================

async function loadTrainingHistory() {
    try {
        UI.showLoadingOverlay('正在加载训练历史...');
        const history = await API.getTrainingHistory();
        
        AppState.trainingHistory = history.map(normalizeHistoryRecord);
        
        console.log('📚 已加载训练历史:', AppState.trainingHistory.length, '条记录');
        
        UI.renderHistoryTable();

    } catch (error) {
        console.error('❌ 加载训练历史失败:', error);
        UI.showErrorMessage('加载训练历史记录失败。');
    }
}

function normalizeHistoryRecord(record) {
    // 检查是否为新的增强数据格式
    if ('best_accuracy' in record && 'hyperparameters_extended' in record) {
        return record; // 已经是新格式，直接返回
    }
    
    // 检查是否为旧的标准化格式
    if ('best_accuracy' in record && 'final_loss' in record) {
        return record; // 已经标准化过，直接返回
    }

    const newRecord = {};
    const oldMetrics = record.metrics || {};
    const oldHyperparams = record.hyperparameters || {};

    newRecord.job_id = record.job_id;
    newRecord.model_id = record.model_id;
    newRecord.model_name = record.model_name;
    newRecord.status = record.status;
    newRecord.error_message = record.error_message || null;
    
    // 🔥 修复时间戳bug：Unix秒级时间戳需要转换为毫秒级，然后转为ISO字符串
    const rawTimestamp = record.completion_time || record.start_time || record.timestamp;
    if (rawTimestamp) {
        if (typeof rawTimestamp === 'string') {
            newRecord.timestamp = rawTimestamp; // 已经是ISO字符串
        } else {
            // 如果时间戳小于某个阈值，说明是秒级时间戳，需要乘以1000
            const timestamp = rawTimestamp < 10000000000 ? rawTimestamp * 1000 : rawTimestamp;
            newRecord.timestamp = new Date(timestamp).toISOString();
        }
    } else {
        newRecord.timestamp = new Date().toISOString();
    }

    newRecord.config = {
        epochs: oldHyperparams.epochs,
        learning_rate: oldHyperparams.learning_rate,
        batch_size: oldHyperparams.batch_size
    };
    
    newRecord.best_accuracy = oldMetrics.final_accuracy || record.best_accuracy || 0;
    newRecord.final_train_loss = oldMetrics.final_loss || record.final_train_loss || 0;
    newRecord.final_val_loss = record.final_val_loss || newRecord.final_train_loss;
    newRecord.model_params = oldMetrics.total_params || record.model_params || 0;
    newRecord.trainable_params = record.trainable_params || newRecord.model_params;
    newRecord.duration_seconds = oldMetrics.training_duration_sec || record.duration_seconds || 0;
    
    newRecord.samples_per_second = oldMetrics.samples_per_second || record.samples_per_second || 0;
    newRecord.epoch_metrics = record.epoch_metrics || [];
    
    // 增强数据（如果存在）
    newRecord.stability_metrics = record.stability_metrics || {};
    newRecord.environment_info = record.environment_info || {};
    newRecord.hyperparameters_extended = record.hyperparameters_extended || {
        basic: newRecord.config
    };

    console.log(`✅ 标准化历史记录: ${record.job_id}`);
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
    UI.renderHyperparameterComparison(filteredHistory);
}

function normalizeInverseMetric(value, maxValue) {
    if (maxValue === 0) return 1;
    return 1 - (value / maxValue);
}

function prepareRadarData(modelsData) {
    const radarLabels = ['准确率', '训练效率', '模型效率', '训练稳定性'];
    
    // 计算各维度的统计数据
    const accuracies = modelsData.map(m => m.best_accuracy);
    const speeds = modelsData.map(m => m.samples_per_second || 0);
    const params = modelsData.map(m => m.model_params || 0);
    const stabilities = modelsData.map(m => {
        // 使用验证集准确率的标准差来衡量稳定性（越小越稳定）
        const valAccStd = m.stability_metrics?.val_accuracy_std || 0.1;
        return 1 / (1 + valAccStd * 10); // 转换为0-1之间，越稳定越接近1
    });
    
    const maxAccuracy = Math.max(...accuracies) || 1;
    const maxSpeed = Math.max(...speeds) || 1;
    const maxParams = Math.max(...params) || 1;
    
    const datasets = modelsData.map(model => {
        const color = ChartUtils.getColorForModel(model.model_id);
        
        // 计算各维度的标准化分数 (0-1)
        const accuracyScore = model.best_accuracy / maxAccuracy;
        const speedScore = (model.samples_per_second || 0) / maxSpeed;
        const efficiencyScore = maxParams > 0 ? (1 - (model.model_params || 0) / maxParams) : 1; // 参数越少效率越高
        const stabilityScore = model.stability_metrics?.val_accuracy_std !== undefined ? 
            1 / (1 + model.stability_metrics.val_accuracy_std * 10) : 0.8; // 默认稳定性

    return {
            label: model.model_name || model.model_id,
                data: [
                Math.max(0, Math.min(1, accuracyScore)),
                Math.max(0, Math.min(1, speedScore)),
                Math.max(0, Math.min(1, efficiencyScore)),
                Math.max(0, Math.min(1, stabilityScore))
            ],
            backgroundColor: color.replace('1)', '0.2)'),
            borderColor: color,
            borderWidth: 2,
            pointBackgroundColor: color,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: color
        };
    });
    return { labels: radarLabels, datasets };
}

function prepareBarData(modelsData) {
    const labels = modelsData.map(m => m.model_name || m.model_id);
    const accuracyData = modelsData.map(m => m.best_accuracy);

    return {
        labels: labels,
        datasets: [{
            label: '最高准确率',
            data: accuracyData,
            backgroundColor: modelsData.map(m => ChartUtils.getColorForModel(m.model_id)),
        }]
    };
}

function prepareLineData(modelsData) {
    // 1. 确定所有训练轮次的最大值，以统一X轴
    const allEpochs = modelsData.flatMap(m => m.epoch_metrics.map(e => e.epoch));
    const maxEpoch = Math.max(...allEpochs, 0);
    const labels = Array.from({ length: maxEpoch }, (_, i) => i + 1);

    // 2. 为每个模型创建一个数据集
    const datasets = modelsData.map(model => {
        const color = ChartUtils.getColorForModel(model.model_id);
        const data = new Array(maxEpoch).fill(null); // 用null填充，以便对齐

        model.epoch_metrics.forEach(epoch => {
            // epoch号从1开始，所以索引是 epoch - 1
            if(epoch.epoch - 1 < maxEpoch) {
                data[epoch.epoch - 1] = epoch.val_accuracy;
            }
        });
        
        return {
            label: model.model_name || model.model_id,
            data: data,
            borderColor: color,
            backgroundColor: color.replace('1)', '0.1)'),
            fill: false,
            tension: 0.1
        };
    });

    return { labels, datasets };
}

function processDataForComparison(historyData) {
    if (!historyData || historyData.length === 0) {
        return null;
    }

    const modelsData = historyData; 

    return {
        radar: prepareRadarData(modelsData),
        bar: prepareBarData(modelsData),
        line: prepareLineData(modelsData)
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