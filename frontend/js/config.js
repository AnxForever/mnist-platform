// 前端配置文件
// 支持本地开发和云端部署的环境配置

const CONFIG = {
    // API基础URL配置
    API_BASE_URL: 'https://mnist-platform-backend.onrender.com',
    // API_BASE_URL: 'http://127.0.0.1:5000', // 本地测试时使用
    DEFAULT_MODEL: 'cnn',
    UPDATE_INTERVAL: 2000, // ms
    
    // API端点配置
    ENDPOINTS: {
        STATUS: '/api/status',
        MODELS: '/api/models',
        PRETRAINED_MODELS: '/api/pretrained_models',
        TRAIN: '/api/train',
        TRAINING_PROGRESS: '/api/training_progress',
        TRAINED_MODELS: '/api/trained_models',
        PREDICT: '/api/predict',
        HISTORY: '/api/history'
    },
    
    // 应用配置
    APP: {
        NAME: 'MNIST智能分析平台',
        VERSION: '2.0.0',
        DESCRIPTION: '深度学习模型训练与对比分析系统'
    },
    
    // 请求配置
    REQUEST: {
        TIMEOUT: 30000, // 30秒超时
        RETRY_ATTEMPTS: 3,
        RETRY_DELAY: 1000 // 1秒重试延迟
    },
    
    // UI配置
    UI: {
        POLLING_INTERVAL: 2000, // 训练进度轮询间隔(毫秒)
        MAX_CONCURRENT_TRAINING: 3,
        DEFAULT_EPOCHS: 10,
        DEFAULT_LEARNING_RATE: 0.001,
        DEFAULT_BATCH_SIZE: 64
    }
};

// 构建完整的API URL
function getApiUrl(endpoint) {
    const baseUrl = CONFIG.API_BASE_URL.replace(/\/$/, ''); // 移除末尾斜杠
    const endpointPath = CONFIG.ENDPOINTS[endpoint] || endpoint;
    return `${baseUrl}${endpointPath}`;
}

// 输出配置信息（调试用）
function logConfig() {
    console.log('🔧 前端配置信息:');
    console.log('  API基础URL:', CONFIG.API_BASE_URL);
    console.log('  当前环境:', window.location.hostname);
    console.log('  应用版本:', CONFIG.APP.VERSION);
}

// 导出配置
if (typeof module !== 'undefined' && module.exports) {
    // Node.js环境
    module.exports = { CONFIG, getApiUrl, logConfig };
} else {
    // 浏览器环境
    window.CONFIG = CONFIG;
    window.getApiUrl = getApiUrl;
    window.logConfig = logConfig;
    
    // 在控制台显示配置信息
    logConfig();
}