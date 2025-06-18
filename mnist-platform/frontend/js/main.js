// 主应用逻辑 - 状态管理、事件绑定、模块协调
import * as API from './api.js';
import * as Canvas from './canvas.js';
import * as UI from './ui.js';
import * as ChartUtils from './chart_utils.js';

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
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 MNIST 智能分析平台启动');
    
    // 初始化标签页切换
    initTabNavigation();
    
    // 初始化模型训练页
    await initModelTrainingPage();
    
    // 初始化事件监听器
    initEventListeners();
    
    // 初始化全局函数
    initGlobalFunctions();
    
    // 页面加载时，根据初始显示的页面加载相应数据
    const initialPage = document.querySelector('.page:not(.hidden)').id;
    if (initialPage === 'page-history') {
        loadTrainingHistory();
    } else if (initialPage === 'page-comparison') {
        loadComparisonData();
    } else if (initialPage === 'page-training') {
        // 首次加载训练页面时，刷新模型列表
        UI.updateAvailableModelsList();
    }
    
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
    try {
        const historyData = await API.getTrainingHistory();
        UI.renderHistoryTable(historyData);
    } catch (error) {
        console.error('加载训练历史失败:', error);
        UI.showToast('加载训练历史失败，请查看控制台');
    }
}

// 加载并显示模型对比数据
async function loadComparisonData() {
    try {
        const historyData = await API.getTrainingHistory();
        if (!historyData || historyData.length === 0) {
            console.log('没有足够的历史数据进行对比。');
            const container = document.getElementById('comparison-charts-container');
            if(container) {
                container.innerHTML = `<div class="empty-state">暂无训练历史，无法进行模型对比</div>`;
            }
            return;
        }
        const processedData = processDataForComparison(historyData);
        UI.renderComparisonCharts(processedData);
    } catch (error) {
        console.error('加载模型对比数据失败:', error);
        UI.showToast('加载模型对比数据失败，请查看控制台');
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
    
    try {
        // 初始化 Canvas
        if (!UI.initializeCanvas()) {
            UI.showErrorMessage('Canvas 初始化失败');
            return;
        }
        
        // 加载已训练模型
        await loadTrainedModelsForPrediction();
        
        // 显示空结果状态
        UI.showEmptyResult();
        
        // 更新预测按钮状态
        UI.updatePredictButtonState();
        
        console.log('✅ 手写识别页面初始化完成');
        
    } catch (error) {
        console.error('❌ 手写识别页面初始化失败:', error);
        UI.showErrorMessage('初始化手写识别功能失败: ' + error.message);
    }
}

// 处理画布清除
function handleCanvasClear() {
    UI.clearCanvas();
    UI.updatePredictButtonState();
    console.log('🧹 用户清除了画布');
}

// 处理预测请求
async function handlePrediction() {
    const modelSelect = document.getElementById('prediction-model-select');
    const selectedModel = modelSelect.value;
    
    if (!selectedModel) {
        UI.showErrorMessage('请先选择一个模型');
        return;
    }
    
    if (UI.isCanvasEmpty()) {
        UI.showErrorMessage('请先在画布上绘制数字');
        return;
    }
    
    try {
        console.log('🔍 开始预测，使用模型:', selectedModel);
        
        // 显示加载状态
        UI.showPredictionLoading();
        
        // 获取画布图像数据
        const imageData = UI.getCanvasImageData();
        if (!imageData) {
            throw new Error('无法获取图像数据');
        }
        
        // 调用预测 API
        const result = await API.predict({
            model_id: selectedModel,
            image_base64: imageData
        });
        
        // 显示预测结果
        UI.renderPredictionResult(result);
        
        console.log('✅ 预测完成:', result);
        
    } catch (error) {
        console.error('❌ 预测失败:', error);
        UI.showErrorMessage('预测失败: ' + error.message);
        UI.showEmptyResult();
    }
}

// 为模型对比图表处理数据
// 从所有历史记录中，为每种模型找出最佳准确率的记录
function processDataForComparison(historyData) {
    const bestRecords = {};

    historyData.forEach(record => {
        const modelType = record.model_name.split(' ')[0]; // e.g., "CNN" from "CNN with Attention"
        const recordAccuracy = record.metrics.final_accuracy;

        if (!bestRecords[modelType] || recordAccuracy > bestRecords[modelType].metrics.final_accuracy) {
            bestRecords[modelType] = record;
        }
    });

    const models = Object.values(bestRecords);
    const labels = models.map(m => {
        const modelName = m.model_name || UI.getModelName(m.model_id);
        return m.has_attention ? `${modelName} ⚡` : modelName;
    });

    // 为雷达图归一化数据
    const accuracies = models.map(m => m.metrics.final_accuracy);
    const speeds = models.map(m => m.metrics.training_duration_sec);
    const params = models.map(m => m.metrics.total_params);

    const maxSpeed = Math.max(...speeds);
    const maxParams = Math.max(...params);

    // 速度越快越好（值越小越好），所以用 1 - normalized
    const normalizedSpeeds = speeds.map(s => 1 - (s / maxSpeed));
    // 参数越少越好（值越小越好），所以用 1 - normalized
    const normalizedParams = params.map(p => 1 - (p / maxParams));

    // 简单的平均值计算函数
    const simpleMean = (arr) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    return {
        labels: labels,
        radarData: {
            labels: ['准确性', '效率 (速度)', '简洁性 (参数)'],
            datasets: [{
                label: '综合性能',
                data: [
                    simpleMean(accuracies), 
                    simpleMean(normalizedSpeeds),
                    simpleMean(normalizedParams)
                ],
                fill: true,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }]
        },
        barData: {
            accuracies: {
                labels,
                datasets: [{
                    label: '最高准确率',
                    data: accuracies.map(a => a * 100), // 转为百分比
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                }]
            },
            speeds: {
                labels,
                datasets: [{
                    label: '训练耗时 (秒)',
                    data: speeds,
                    backgroundColor: 'rgba(255, 159, 64, 0.6)',
                }]
            },
            params: {
                labels,
                datasets: [{
                    label: '模型参数量',
                    data: params,
                    backgroundColor: 'rgba(153, 102, 255, 0.6)',
                }]
            }
        }
    };
}

// 导出状态（用于调试）
window.AppState = AppState; 