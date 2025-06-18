// API 模块 - 封装所有后端API请求
const API_BASE_URL = 'http://localhost:5000/api';

// 网络请求的通用错误处理
async function handleResponse(response) {
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API请求失败 (${response.status}): ${errorText}`);
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
    return fetchWithRetry(`${API_BASE_URL}/models`);
}

// 启动模型训练
export async function startTraining(payload) {
    return fetchWithRetry(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });
}

// 查询训练进度
export async function getTrainingProgress(jobIds) {
    const jobIdsParam = Array.isArray(jobIds) ? jobIds.join(',') : jobIds;
    return fetchWithRetry(`${API_BASE_URL}/training_progress?job_ids=${jobIdsParam}`);
}

// 获取已训练模型列表
export async function getTrainedModels() {
    return fetchWithRetry(`${API_BASE_URL}/trained_models`);
}

// 执行手写识别预测
export async function predict(payload) {
    // 验证输入参数
    if (!payload || !payload.model_id || !payload.image_base64) {
        throw new Error('预测请求参数不完整：需要model_id和image_base64');
    }
    
    // 验证base64图像数据
    if (!payload.image_base64.startsWith('data:image/')) {
        throw new Error('图像数据格式错误：必须是有效的base64编码');
    }
    
    try {
        console.log('📡 发送预测请求，模型:', payload.model_id);
        
        const result = await fetchWithRetry(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        }, 2); // 预测请求减少重试次数，提高响应速度
        
        // 验证返回结果
        if (!result || typeof result.prediction === 'undefined' || !Array.isArray(result.probabilities)) {
            throw new Error('服务器返回的预测结果格式错误');
        }
        
        if (result.probabilities.length !== 10) {
            throw new Error('概率分布数据不完整：应包含10个数字的概率');
        }
        
        console.log('✅ 预测请求成功');
        return result;
        
    } catch (error) {
        console.error('❌ 预测请求失败:', error.message);
        throw new Error(`预测失败: ${error.message}`);
    }
}

// 获取训练历史记录
export async function getHistory() {
    return fetchWithRetry(`${API_BASE_URL}/history`);
} 