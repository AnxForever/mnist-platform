// API 模块 - 封装所有后端API请求
const API_BASE_URL = 'http://localhost:5000/api';

// 网络请求的通用错误处理
async function handleResponse(response) {
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: '未知错误' }));
        throw new Error(errorData.error || '请求失败');
    }
    return response.json();
}

// 网络请求的重试机制
async function fetchWithRetry(url, options = {}, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            return await handleResponse(response);
        } catch (error) {
            if (i === retries - 1) throw error;
            console.warn(`API请求重试 ${i + 1}/${retries}:`, error.message);
            await new Promise(resolve => setTimeout(resolve, 1000)); // 等待1秒后重试
        }
    }
}

// 获取可选模型列表
export async function getModels() {
    return fetch(`${API_BASE_URL}/models`).then(handleResponse);
}

// 启动模型训练 - 修复参数格式
export async function startTraining(trainingData) {
    return fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingData),
    }).then(handleResponse);
}

// 查询训练进度
export async function getTrainingProgress(jobIds) {
    const jobIdParams = Array.isArray(jobIds) ? jobIds.join(',') : jobIds;
    return fetch(`${API_BASE_URL}/training_progress?job_ids=${jobIdParams}`).then(handleResponse);
}

// 获取已训练模型列表
export async function getTrainedModels() {
    return fetch(`${API_BASE_URL}/trained_models`).then(handleResponse);
}

// 执行手写识别预测
export async function predict(modelId, filename, imageBase64) {
    const body = {
        model_id: modelId,
        filename: filename,
        image_base64: imageBase64
    };
    return fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }).then(handleResponse);
}

// 获取训练历史记录
export async function getTrainingHistory() {
    return fetch(`${API_BASE_URL}/history`).then(handleResponse);
}

export async function cancelTraining(jobId) {
    return fetch(`${API_BASE_URL}/cancel_training?job_id=${jobId}`, { method: 'POST' }).then(handleResponse);
} 