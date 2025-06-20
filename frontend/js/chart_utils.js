/**
 * chart_utils.js
 * 
 * 这个模块封装了所有与图表创建相关的功能，主要使用 Chart.js 库。
 * 它提供了创建雷达图和分组柱状图的便捷函数，用于模型性能的可视化对比。
 */

// 存储已创建的图表实例，防止重复渲染
const chartInstances = {};

/**
 * 销毁已存在的图表实例
 * @param {string} chartId - canvas元素的ID
 */
function destroyChart(chartId) {
    if (chartInstances[chartId]) {
        chartInstances[chartId].destroy();
        delete chartInstances[chartId];
    }
}

/**
 * 创建或更新雷达图
 * @param {string} chartId - canvas元素的ID
 * @param {object} data - 图表数据，包含 labels 和 datasets
 */
export function createRadarChart(chartId, data) {
    const ctx = document.getElementById(chartId);
    if (!ctx) {
        console.error(`❌ 未找到ID为 '${chartId}' 的Canvas元素`);
        return;
    }

    destroyChart(chartId); // 先销毁旧图表

    chartInstances[chartId] = new Chart(ctx, {
        type: 'radar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '模型综合性能雷达图',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
    console.log('📊 已创建雷达图');
}

/**
 * 创建或更新分组柱状图
 * @param {string} chartId - canvas元素的ID
 * @param {object} data - 图表数据，包含 labels 和 datasets
 * @param {string} title - 图表标题
 */
export function createBarChart(chartId, data, title) {
    const ctx = document.getElementById(chartId);
    if (!ctx) {
        console.error(`❌ 未找到ID为 '${chartId}' 的Canvas元素`);
        return;
    }
    
    destroyChart(chartId); // 先销毁旧图表

    // 智能Y轴范围调整
    const allDataPoints = data.datasets.flatMap(dataset => dataset.data);
    const minValue = Math.min(...allDataPoints);
    const suggestedMin = minValue > 0.9 ? minValue * 0.99 : minValue * 0.9;

    chartInstances[chartId] = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    suggestedMin: suggestedMin,
                    suggestedMax: 1.0,
                    ticks: {
                        // 格式化为百分比
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
    console.log(`📊 已创建柱状图: ${title}`);
}

/**
 * 渲染损失对比折线图 (训练 vs 验证)
 * @param {string} chartId - canvas元素的ID
 * @param {Array} history - 包含epoch指标的数组
 */
export function renderLossChart(chartId, history) {
    const ctx = document.getElementById(chartId);
    if (!ctx) {
        console.error(`❌ 未找到ID为 '${chartId}' 的Canvas元素`);
        return;
    }
    
    destroyChart(chartId); // 先销毁旧图表

    const labels = history.map(h => `Epoch ${h.epoch}`);
    const trainLossData = history.map(h => h.loss);
    // 关键：如果旧数据没有val_loss，则传递null，Chart.js会优雅地处理断点
    const valLossData = history.map(h => h.val_loss ?? null);

    const data = {
        labels: labels,
        datasets: [
            {
                label: '训练损失 (Training Loss)',
                data: trainLossData,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: false,
                tension: 0.1
            },
            {
                label: '验证损失 (Validation Loss)',
                data: valLossData,
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: false,
                tension: 0.1,
                borderDash: [5, 5] // 使用虚线以区分
            }
        ]
    };

    chartInstances[chartId] = new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: '训练 vs 验证 损失', font: { size: 16 } },
                legend: { position: 'top' }
            },
            scales: {
                x: { title: { display: true, text: '训练轮次 (Epoch)' } },
                y: { beginAtZero: true, title: { display: true, text: '损失值 (Loss)' } }
            }
        }
    });
    console.log(`📊 已创建损失对比图: ${chartId}`);
}

/**
 * 渲染准确率对比折线图 (训练 vs 验证)
 * @param {string} chartId - canvas元素的ID
 * @param {Array} history - 包含epoch指标的数组
 */
export function renderAccuracyChart(chartId, history) {
    const ctx = document.getElementById(chartId);
    if (!ctx) {
        console.error(`❌ 未找到ID为 '${chartId}' 的Canvas元素`);
        return;
    }
    
    destroyChart(chartId); // 先销毁旧图表

    const labels = history.map(h => `Epoch ${h.epoch}`);
    const trainAccData = history.map(h => h.accuracy);
    // 关键：如果旧数据没有val_accuracy，则传递null
    const valAccData = history.map(h => h.val_accuracy ?? null);

    const data = {
        labels: labels,
        datasets: [
            {
                label: '训练准确率 (Training Accuracy)',
                data: trainAccData,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: false,
                tension: 0.1
            },
            {
                label: '验证准确率 (Validation Accuracy)',
                data: valAccData,
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                fill: false,
                tension: 0.1,
                borderDash: [5, 5] // 使用虚线以区分
            }
        ]
    };

    chartInstances[chartId] = new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: '训练 vs 验证 准确率', font: { size: 16 } },
                legend: { position: 'top' }
            },
            scales: {
                x: { title: { display: true, text: '训练轮次 (Epoch)' } },
                y: {
                    beginAtZero: false, // 准确率可能从较高值开始
                    suggestedMin: Math.min(...trainAccData, ...valAccData.filter(v => v !== null)) * 0.95,
                    suggestedMax: 1,
                    title: { display: true, text: '准确率 (Accuracy)' }
                }
            }
        }
    });
    console.log(`📊 已创建准确率对比图: ${chartId}`);
}

/**
 * 创建或更新折线图
 * @param {string} chartId - canvas元素的ID
 * @param {object} data - 图表数据，包含 labels 和 datasets
 * @param {string} title - 图表标题
 */
export function createLineChart(chartId, data, title) {
    const ctx = document.getElementById(chartId);
    if (!ctx) {
        console.error(`❌ 未找到ID为 '${chartId}' 的Canvas元素`);
        return;
    }
    
    destroyChart(chartId); // 先销毁旧图表

    // 智能Y轴范围调整
    const allDataPoints = data.datasets.flatMap(dataset => dataset.data.filter(d => d !== null));
    const minValue = allDataPoints.length > 0 ? Math.min(...allDataPoints) : 0.8;
    // 如果最低值已经很高了，就把起点设置得更接近一些，以放大差异
    const suggestedMin = minValue > 0.9 ? minValue * 0.995 : minValue * 0.98;

    chartInstances[chartId] = new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '训练轮次 (Epoch)'
                    }
                },
                y: {
                    beginAtZero: false,
                    suggestedMin: suggestedMin,
                    suggestedMax: 1.0,
                    title: {
                        display: true,
                        text: '验证准确率'
                    },
                    ticks: {
                        // 格式化为百分比
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
    console.log(`📊 已创建折线图: ${title}`);
}

/**
 * 根据模型ID返回一个固定的颜色，确保图表颜色一致性
 * @param {string} modelId - 模型ID (e.g., 'cnn', 'mlp_attention')
 * @returns {string} - RGBA格式的颜色字符串
 */
export function getColorForModel(modelId) {
    const modelBase = modelId.replace('_attention', '');
    const hasAttention = modelId.includes('_attention');

    // 基础颜色映射
    const colors = {
        'mlp': 'rgba(54, 162, 235, 1)',   // 蓝色
        'cnn': 'rgba(255, 99, 132, 1)',   // 红色
        'rnn': 'rgba(75, 192, 192, 1)'    // 绿色
    };

    let color = colors[modelBase] || 'rgba(201, 203, 207, 1)'; // 默认为灰色

    // 如果有Attention，微调颜色（例如，增加透明度或亮度, 这里我们只简单返回相同颜色，但可以扩展）
    // 这里为了区分，我们稍微改变一下颜色, 但保持主色调
    if (hasAttention) {
        switch (modelBase) {
            case 'mlp':
                return 'rgba(30, 136, 229, 1)'; // 深蓝
            case 'cnn':
                return 'rgba(239, 83, 80, 1)'; // 深红
            case 'rnn':
                return 'rgba(0, 150, 136, 1)'; // 深绿
        }
    }

    return color;
}

/**
 * 清除所有已创建的图表实例
 */
export function clearAllCharts() {
    Object.keys(chartInstances).forEach(id => {
        destroyChart(id);
    });
    console.log('🧹 已清除所有图表');
}
