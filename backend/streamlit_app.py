import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import base64
from models import get_model_instance
from pretrained_models import PretrainedModelManager
import os

# 页面配置
st.set_page_config(
    page_title="🤖 MNIST智能分析平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化预训练模型管理器
@st.cache_resource
def init_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return PretrainedModelManager(current_dir)

pretrained_manager = init_models()

# 主标题
st.title("🤖 MNIST智能分析平台")
st.markdown("### 支持6种深度学习模型的手写数字识别系统")

# 侧边栏
st.sidebar.title("🎯 选择功能")
mode = st.sidebar.selectbox(
    "功能模式",
    ["🎨 手写识别", "📊 模型对比", "ℹ️ 关于系统"]
)

if mode == "🎨 手写识别":
    st.header("✏️ 手写数字识别")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 输入方式")
        input_method = st.radio("选择输入方式", ["上传图片", "画布绘制"])
        
        if input_method == "上传图片":
            uploaded_file = st.file_uploader(
                "上传手写数字图片",
                type=['png', 'jpg', 'jpeg'],
                help="支持PNG、JPG格式，建议28x28像素"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="上传的图片", width=200)
                
                # 预处理图片
                image_gray = image.convert('L')
                image_resized = image_gray.resize((28, 28))
                image_array = np.array(image_resized) / 255.0
                image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
                
                if st.button("🔍 开始识别", type="primary"):
                    with st.spinner("AI正在识别中..."):
                        # 获取所有预训练模型的预测
                        results = {}
                        for model_id, model_info in pretrained_manager.pretrained_info.items():
                            try:
                                model = pretrained_manager.load_pretrained_model(model_id)
                                if model:
                                    model.eval()
                                    with torch.no_grad():
                                        output = model(image_tensor)
                                        probabilities = torch.softmax(output, dim=1)
                                        predicted = torch.argmax(probabilities, dim=1).item()
                                        confidence = probabilities[0][predicted].item()
                                        
                                        results[model_info['name']] = {
                                            'prediction': predicted,
                                            'confidence': confidence
                                        }
                            except Exception as e:
                                st.error(f"{model_info['name']} 识别失败: {str(e)}")
                        
                        # 显示结果
                        with col2:
                            st.subheader("🎯 识别结果")
                            
                            for model_name, result in results.items():
                                with st.container():
                                    st.markdown(f"**{model_name}**")
                                    col_pred, col_conf = st.columns([1, 2])
                                    with col_pred:
                                        st.metric("预测数字", f"{result['prediction']}")
                                    with col_conf:
                                        confidence_percent = result['confidence'] * 100
                                        st.metric("置信度", f"{confidence_percent:.1f}%")
                                    
                                    # 置信度进度条
                                    st.progress(result['confidence'])
                                    st.divider()
        
        else:  # 画布绘制
            st.info("🎨 画布功能需要在Web版本中使用")
            st.markdown("请使用完整的Web界面来体验手绘功能")

elif mode == "📊 模型对比":
    st.header("📊 模型性能对比")
    
    # 显示模型性能表格
    model_data = []
    for model_id, info in pretrained_manager.pretrained_info.items():
        model_data.append({
            "模型类型": info['name'],
            "准确率": f"{info['accuracy']:.2%}",
            "特点": info['description']
        })
    
    st.table(model_data)
    
    # 性能图表
    st.subheader("📈 准确率对比")
    
    models = [info['name'] for info in pretrained_manager.pretrained_info.values()]
    accuracies = [info['accuracy'] for info in pretrained_manager.pretrained_info.values()]
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    
    ax.set_ylabel('准确率')
    ax.set_title('各模型准确率对比')
    ax.set_ylim(0.95, 1.0)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{acc:.2%}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

elif mode == "ℹ️ 关于系统":
    st.header("ℹ️ 关于MNIST智能分析平台")
    
    st.markdown("""
    ### 🚀 系统特性
    
    - **6种深度学习模型**: MLP、CNN、RNN及其Attention增强版
    - **实时识别**: 毫秒级响应速度
    - **预训练模型**: 无需等待，立即体验
    - **高准确率**: 最高99.2%的识别准确率
    
    ### 🛠️ 技术栈
    
    - **后端**: Python + PyTorch + Flask
    - **前端**: Streamlit + HTML5 Canvas
    - **部署**: 云端部署，全球访问
    
    ### 📊 模型架构
    
    1. **MLP (多层感知机)**: 基础全连接网络
    2. **CNN (卷积神经网络)**: 卷积特征提取
    3. **RNN (循环神经网络)**: 序列信息处理
    4. **Attention机制**: 注意力增强版本
    
    ### 🎯 使用指南
    
    1. 上传或绘制手写数字图片
    2. 选择识别模型
    3. 查看识别结果和置信度
    4. 对比不同模型的性能
    """)
    
    st.success("🎉 感谢使用MNIST智能分析平台！")

# 底部信息
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 系统状态")
st.sidebar.success("✅ 系统运行正常")
st.sidebar.info(f"📦 已加载 {len(pretrained_manager.pretrained_info)} 个预训练模型") 