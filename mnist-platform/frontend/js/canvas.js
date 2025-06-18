/**
 * canvas.js
 * 
 * 封装所有与手写画板相关的逻辑。
 * 包括：初始化、事件监听、绘图、清空、获取图像数据等。
 */

// 模块内部状态
let canvas, context;
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// 回调函数，用于通知外部（如 main.js）状态变化
let onCanvasChangeCallback = () => {};

// 防抖函数，用于减少事件处理频率
function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, delay);
    };
}

// 延迟执行的回调，提升性能
const debouncedCanvasChange = debounce(() => onCanvasChangeCallback(), 150);

/**
 * 初始化Canvas模块
 * @param {Function} onCanvasChange - 当Canvas内容变化时调用的回调函数
 */
export function init(onCanvasChange) {
    canvas = document.getElementById('drawing-canvas');
    const clearButton = document.getElementById('clear-canvas-btn');
    const brushSizeSlider = document.getElementById('brush-size-slider');
    const brushSizeValue = document.getElementById('brush-size-value');

    if (!canvas || !clearButton || !brushSizeSlider) {
        console.error('❌ Canvas或其控制元素未找到，手写识别功能可能无法初始化。');
        return;
    }

    context = canvas.getContext('2d');
    context.lineJoin = 'round';
    context.lineCap = 'round';

    // 设置初始画笔粗细
    context.lineWidth = brushSizeSlider.value;
    brushSizeValue.textContent = `${brushSizeSlider.value}px`;

    // 存储回调函数
    onCanvasChangeCallback = onCanvasChange;

    // --- 事件监听 ---
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // 触摸事件支持 (用于移动设备)
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }, { passive: false });
    
    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        const mouseEvent = new MouseEvent('mouseup', {});
        canvas.dispatchEvent(mouseEvent);
    }, { passive: false });

    // 清空按钮
    clearButton.addEventListener('click', clearCanvas);

    // 画笔粗细滑块
    brushSizeSlider.addEventListener('input', (e) => {
        context.lineWidth = e.target.value;
        brushSizeValue.textContent = `${e.target.value}px`;
    });

    // 初始清空并设置背景
    clearCanvas(true); // 传入true表示不触发回调
    console.log('🎨 Canvas 绘制模块已加载');
}

/**
 * 开始绘制
 * @param {MouseEvent} e 
 */
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
    onCanvasChangeCallback(); // 开始绘制即认为内容已改变
}

/**
 * 绘制过程
 * @param {MouseEvent} e 
 */
function draw(e) {
    if (!isDrawing) return;
    context.beginPath();
    context.moveTo(lastX, lastY);
    context.lineTo(e.offsetX, e.offsetY);
    context.stroke();
    [lastX, lastY] = [e.offsetX, e.offsetY];
    debouncedCanvasChange(); // 绘制过程中使用防抖回调
}

/**
 * 停止绘制
 */
function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        onCanvasChangeCallback(); // 停止绘制时立即触发回调
    }
}

/**
 * 清空画板
 * @param {boolean} silent - 如果为true，则不触发onCanvasChange回调
 */
export function clearCanvas(silent = false) {
    context.fillStyle = '#000000'; // 设置为纯黑色背景
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = '#FFFFFF'; // 设置画笔为纯白色
    if (!silent) {
        onCanvasChangeCallback();
    }
    console.log('🧹 画板已清空');
}

/**
 * 获取画板的图像数据 (Base64)
 * @returns {string} - Base64编码的图像数据
 */
export function getImageData() {
    if (isEmpty()) return null;
    return canvas.toDataURL('image/png');
}

/**
 * 检查画板是否为空
 * @returns {boolean} - 如果为空则返回true
 */
export function isEmpty() {
    // 获取整个Canvas的图像数据
    const pixelBuffer = new Uint32Array(
        context.getImageData(0, 0, canvas.width, canvas.height).data.buffer
    );
    // 检查是否有任何非背景色（黑色）的像素
    // 0xFF000000 是纯黑色的像素值 (A=255, R=0, G=0, B=0)
    return !pixelBuffer.some(color => color !== 0xFF000000);
} 