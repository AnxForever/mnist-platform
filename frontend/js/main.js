// 主应用逻辑 - 状态管理、事件绑定、模块协调
import * as API from './api.js';
import { init as initCanvas, getImageData, clearCanvas } from './canvas.js';
import * as UI from './ui.js';
import * as ChartUtils from './chart_utils.js';
import { getModelName } from './ui.js';

// 全局应用状态
const AppState = {
    // 静态数据
    availableModels: [],
    
    // 动态数据
    currentTrainingJobs: {},
    pollingIntervalId: null,
    trainedModels: [],
    trainingHistory: [],
    
    // UI 状态
    activeTab: 'model-training',
    selectedModels: new Set()
};

// 应用初始化
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 MNIST 智能分析平台启动');
    
    // 初始化标签页切换
    initTabNavigation();
    
    // 初始化模型训练页
    initModelTrainingPage();
    
    // 初始化事件监听器
    initEventListeners();
    
    // 初始化全局函数
    initGlobalFunctions();
    
    console.log('✅ 应用初始化完成');
});

// 初始化全局函数
function initGlobalFunctions() {
    // 滑块值更新函数
    window.updateSliderValue = function(paramName, value) {
        const valueElement = document.getElementById(`${paramName}-value`);
        if (valueElement) {
            valueElement.textContent = value;
        }
    };
}

// 标签页导航初始化
function initTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            
            // 更新按钮状态
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // 更新内容显示
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === targetTab) {
                    content.classList.add('active');
                }
            });
            
            // 更新应用状态
            AppState.activeTab = targetTab;
            
            // 根据切换的标签页执行特定逻辑
            handleTabSwitch(targetTab);
        });
    });
}

// 模型训练页初始化
async function initModelTrainingPage() {
    try {
        // 获取可用模型列表
        AppState.availableModels = await API.getModels();
        
        // 渲染模型卡片
        UI.renderModelCards(AppState.availableModels);
        
        console.log('📋 已加载模型列表:', AppState.availableModels);
    } catch (error) {
        console.error('❌ 加载模型列表失败:', error);
        UI.showErrorMessage('加载模型列表失败，请检查后端服务是否正常运行');
    }
}

// 事件监听器初始化
function initEventListeners() {
    // 开始训练按钮
    const startTrainingBtn = document.getElementById('start-training-btn');
    startTrainingBtn.addEventListener('click', handleStartTraining);
    
    // 模型选择变化监听（委托事件）
    document.addEventListener('change', (e) => {
        if (e.target.classList.contains('model-checkbox')) {
            handleModelSelection(e.target);
        }
        
        // 手写识别模型选择变化
        if (e.target.id === 'prediction-model-select') {
            UI.updatePredictButtonState();
        }
        
        // 画笔大小变化
        if (e.target.id === 'brush-size-slider') {
            UI.updateBrushSize(parseInt(e.target.value));
        }
    });
    
    // 手写识别相关事件
    const clearCanvasBtn = document.getElementById('clear-canvas-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    if (clearCanvasBtn) {
        clearCanvasBtn.addEventListener('click', handleCanvasClear);
    }
    
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
}

// 处理标签页切换
async function handleTabSwitch(tabId) {
    switch (tabId) {
        case 'handwriting-recognition':
            await initHandwritingRecognition();
            break;
        case 'training-results':
            await loadTrainingHistory();
            break;
        case 'model-comparison':
            await loadComparisonData();
            break;
    }
}

// 处理模型选择
function handleModelSelection(checkbox) {
    const modelId = checkbox.value;
    const startTrainingBtn = document.getElementById('start-training-btn');
    
    if (checkbox.checked) {
        AppState.selectedModels.add(modelId);
    } else {
        AppState.selectedModels.delete(modelId);
    }
    
    // 更新开始训练按钮状态
    startTrainingBtn.disabled = AppState.selectedModels.size === 0;
    
    console.log('🎯 已选择模型:', Array.from(AppState.selectedModels));
}

// 处理开始训练
async function handleStartTraining() {
    if (AppState.selectedModels.size === 0) {
        UI.showErrorMessage('请至少选择一个模型进行训练');
        return;
    }
    
    try {
        // 获取训练参数
        const params = getTrainingParameters();
        
        // 构建训练配置
        const trainingConfigs = Array.from(AppState.selectedModels).map(modelId => ({
            id: modelId,
            epochs: params.epochs,
            lr: params.lr,
            batch_size: params.batch_size
        }));
        
        console.log('📊 训练参数:', params);
        
        console.log('🔥 开始训练，配置:', trainingConfigs);
        
        // 调用后端API启动训练
        const response = await API.startTraining({ models: trainingConfigs });
        
        // 保存任务信息
        response.jobs.forEach(job => {
            AppState.currentTrainingJobs[job.job_id] = {
                model_id: job.model_id,
                status: 'queued'
            };
        });
        
        // 创建初始进度条
        UI.createTrainingProgressBars(response.jobs);
        
        // 开始轮询进度
        startProgressPolling();
        
        // 禁用开始训练按钮
        document.getElementById('start-training-btn').disabled = true;
        
        console.log('✅ 训练任务已启动:', response.jobs);
        
    } catch (error) {
        console.error('❌ 启动训练失败:', error);
        UI.showErrorMessage('启动训练失败: ' + error.message);
    }
}

// 开始进度轮询
function startProgressPolling() {
    if (AppState.pollingIntervalId) {
        clearInterval(AppState.pollingIntervalId);
    }
    
    AppState.pollingIntervalId = setInterval(async () => {
        await updateTrainingProgress();
    }, 2000); // 每2秒轮询一次
    
    console.log('⏰ 已开始进度轮询');
}

// 更新训练进度
async function updateTrainingProgress() {
    const runningJobIds = Object.keys(AppState.currentTrainingJobs).filter(
        jobId => ['queued', 'running'].includes(AppState.currentTrainingJobs[jobId].status)
    );
    
    if (runningJobIds.length === 0) {
        // 所有任务都已完成，停止轮询
        clearInterval(AppState.pollingIntervalId);
        AppState.pollingIntervalId = null;
        
        // 重新启用开始训练按钮
        document.getElementById('start-training-btn').disabled = AppState.selectedModels.size === 0;
        
        console.log('🏁 所有训练任务已完成');
        return;
    }
    
    try {
        const progressData = await API.getTrainingProgress(runningJobIds);
        
        // 更新本地状态和UI
        progressData.progress.forEach(jobProgress => {
            if (AppState.currentTrainingJobs[jobProgress.job_id]) {
                AppState.currentTrainingJobs[jobProgress.job_id] = jobProgress;
                UI.updateProgressBar(jobProgress.job_id, jobProgress);
            }
        });
        
    } catch (error) {
        console.error('❌ 更新训练进度失败:', error);
    }
}

// 加载已训练模型（用于预测）
async function loadTrainedModelsForPrediction() {
    try {
        console.log('📋 加载已训练模型列表');
        AppState.trainedModels = await API.getTrainedModels();
        UI.renderTrainedModels(AppState.trainedModels);
        
        // 更新预测按钮状态
        setTimeout(() => {
            UI.updatePredictButtonState();
        }, 100);
        
        console.log('✅ 已加载', AppState.trainedModels.length, '个已训练模型');
    } catch (error) {
        console.error('❌ 加载已训练模型失败:', error);
        UI.showErrorMessage('加载模型列表失败: ' + error.message);
    }
}

// 加载训练历史
async function loadTrainingHistory() {
    console.log("开始加载训练历史...");
    try {
        const historyData = await API.getTrainingHistory();
        console.log("获取到训练历史数据:", historyData);
        UI.renderHistoryTable(historyData);
    } catch (error) {
        console.error("加载训练历史失败:", error);
        // 使用正确的错误提示函数
        UI.showErrorMessage(`加载训练历史失败: ${error.message}`);
    }
}

// 加载并显示模型对比数据
async function loadComparisonData() {
    console.log("开始加载模型对比数据...");
    try {
        // 1. 获取正确的原材料：训练历史
        const historyData = await API.getTrainingHistory();
        console.log("获取到训练历史用于对比:", historyData);

        // 2. 调用配菜师进行数据加工
        const processedData = processDataForComparison(historyData);
        console.log("处理后的对比数据:", processedData);

        // 3. 把配好的菜交给厨师
        UI.renderComparisonCharts(processedData);
    } catch (error) {
        console.error("加载模型对比数据失败:", error);
        // 使用正确的错误提示函数
        UI.showErrorMessage(`加载模型对比数据失败: ${error.message}`);
    }
}

// 获取训练参数
function getTrainingParameters() {
    const epochs = document.getElementById('epochs-slider')?.value || 10;
    const learningRate = document.getElementById('learning-rate-input')?.value || 0.001;
    const batchSize = document.getElementById('batch-size-input')?.value || 64;
    
    return {
        epochs: parseInt(epochs),
        lr: parseFloat(learningRate),
        batch_size: parseInt(batchSize)
    };
}

// ==================== 手写识别页面功能 ====================

// 初始化手写识别页面
async function initHandwritingRecognition() {
    console.log('🎨 初始化手写识别页面');
    
    // 初始化Canvas
    const canvasInitialized = initCanvas(UI.updatePredictButtonState);
    if (!canvasInitialized) {
        UI.showErrorMessage("Canvas 初始化失败");
        return;
    }

    // 加载已训练模型列表
    await loadTrainedModelsForPrediction();
}

function handleCanvasClear() {
    clearCanvas();
}

async function handlePrediction() {
    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 识别中...';

    try {
        const modelSelect = document.getElementById('prediction-model-select');
        const modelId = modelSelect.value;
        const imageBase64 = getImageData();

        if (!modelId) {
            UI.showErrorMessage('请先选择一个识别模型');
            return;
        }

        if (!imageBase64) {
            UI.showErrorMessage('画板为空，请先绘制一个数字');
            return;
        }

        // 开始预测
        console.log(`🔍 开始预测，使用模型: ${modelId}`);
        const result = await API.predict(modelId, imageBase64);

        // 显示结果
        UI.renderPredictionResult(result);
        
        console.log('✅ 预测完成:', result);

    } catch (error) {
        console.error('❌ 预测失败:', error);
        UI.showErrorMessage('预测失败: ' + error.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '识别';
    }
}

/**
 * 归一化处理"值越小越好"的指标（如：耗时、参数量）
 * @param {number} value - 当前值
 * @param {number} maxValue - 所有值中的最大值
 * @returns {number} 0到1之间的归一化分数
 */
function normalizeInverseMetric(value, maxValue) {
    if (maxValue === 0) {
        return 1; // 如果最大值是0，说明已经达到最佳，给满分
    }
    return 1 - (value / maxValue);
}

// 为模型对比图表处理数据
// 从所有历史记录中，为每种模型找出最佳准确率的记录，并格式化为图表所需数据
function processDataForComparison(historyData) {
    const bestRecords = {};

    // 1. 找出每个模型类型的最佳记录
    historyData.forEach(record => {
        const modelId = record.model_id;
        if (!bestRecords[modelId] || record.metrics.final_accuracy > bestRecords[modelId].metrics.final_accuracy) {
            bestRecords[modelId] = record;
        }
    });

    const models = Object.values(bestRecords);
    if (models.length === 0) {
        return null;
    }
    
    // 2. 提取并格式化用于柱状图和雷达图的数据
    const labels = models.map(m => getModelName(m.model_id));
    const accuracies = models.map(m => m.metrics.final_accuracy);
    const speeds = models.map(m => m.metrics.training_duration_sec);
    const params = models.map(m => m.metrics.total_params);

    const barData = {
        accuracies: {
            labels,
            datasets: [{ label: '最高准确率', data: accuracies.map(a => a * 100), backgroundColor: 'rgba(75, 192, 192, 0.6)' }]
        },
        speeds: {
            labels,
            datasets: [{ label: '训练耗时 (秒)', data: speeds, backgroundColor: 'rgba(255, 159, 64, 0.6)' }]
        },
        params: {
            labels,
            datasets: [{ label: '模型参数量', data: params, backgroundColor: 'rgba(153, 102, 255, 0.6)' }]
        }
    };
    
    // 3. 为雷达图归一化数据
    const radarLabels = ['准确性', '效率 (速度)', '简洁性 (参数)'];
    const maxSpeed = Math.max(...speeds);
    const maxParams = Math.max(...params);

    const radarData = {
        labels: radarLabels,
        datasets: models.map((model, index) => {
            const color = CHART_COLORS[index % CHART_COLORS.length];
            const normalizedAcc = model.metrics.final_accuracy;
            const normalizedSpeed = normalizeInverseMetric(model.metrics.training_duration_sec, maxSpeed);
            const normalizedParams = normalizeInverseMetric(model.metrics.total_params, maxParams);
            return {
                label: getModelName(model.model_id),
                data: [normalizedAcc, normalizedSpeed, normalizedParams],
                fill: true,
                backgroundColor: `rgba(${color}, 0.2)`,
                borderColor: `rgb(${color})`,
                pointBackgroundColor: `rgb(${color})`,
            };
        })
    };

    // 4. 提取并格式化用于折线图的数据
    const modelsForLineChart = models.filter(
        m => m.metrics && Array.isArray(m.metrics.epoch_metrics) && m.metrics.epoch_metrics.length > 0
    );

    let lineChartData = { labels: [], datasets: [] }; // 默认空结构，确保安全

    if (modelsForLineChart.length > 0) {
        const maxEpochs = Math.max(...modelsForLineChart.map(m => m.metrics.epoch_metrics.length));
        const epochLabels = Array.from({ length: maxEpochs }, (_, i) => `Epoch ${i + 1}`);
        
        lineChartData = {
            labels: epochLabels,
            datasets: modelsForLineChart.map((model, index) => {
                const color = CHART_COLORS[index % CHART_COLORS.length];
                return {
                    label: `${getModelName(model.model_id)} 验证准确率`,
                    data: model.metrics.epoch_metrics.map(e => e.val_accuracy),
                    borderColor: `rgb(${color})`,
                    backgroundColor: `rgba(${color}, 0.5)`,
                    tension: 0.1
                };
            })
        };
    }

    return { barData, radarData, lineChartData };
}

// 帮助渲染的颜色
const CHART_COLORS = [
    '75, 192, 192', '255, 99, 132', '255, 205, 86', 
    '54, 162, 235', '153, 102, 255', '255, 159, 64'
];

// 导出状态（用于调试）
window.AppState = AppState;

window.Module = {
    // 只保留这一个由 HTML onclick 调用的函数
    toggleHistoryDetails: UI.toggleHistoryDetails
}; 