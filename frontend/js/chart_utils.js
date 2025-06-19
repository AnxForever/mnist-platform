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
                    beginAtZero: true
                }
            }
        }
    });
    console.log(`📊 已创建柱状图: ${title}`);
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
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '准确率'
                    }
                }
            }
        }
    });
    console.log(`📊 已创建折线图: ${title}`);
}
