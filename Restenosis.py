# In[21]:
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# In[22]:


# Load the model
model = joblib.load('XGBoost.pkl')


# In[17]:


print(type(X_train), X_train.shape)
print(type(y_train), y_train.shape)


# In[23]:


# 定义特征选项
high_altitude_options = {1: 'High altitude (1)', 2: 'Low altitude (2)'}
aptt_options = {1: 'Low (1)', 2: 'Normal (2)', 3: 'High (3)'}
wbc_options = {1: 'Low (1)', 2: 'Normal (2)', 3: 'High (3)'}
monocyte_options = {2: 'Normal (2)', 3: 'High (3)'}
neutrophil_options = {1: 'Low (1)', 2: 'Normal (2)', 3: 'High (3)'}
lymphocyte_options = {1: 'Low (1)', 2: 'Normal (2)', 3: 'High (3)'}
cancer_options = {0: 'No (0)', 1: 'Yes (1)'}
chd_options = {0: 'No (0)', 1: 'Yes (1)'}
surgical_method_options = {
    1: 'Balloon Angioplasty (1)',
    2: 'Drug Balloon Angioplasty (2)',
    3: 'Balloon Angioplasty + Stent Insertion (3)',
    4: 'Balloon Angioplasty + Drug-eluting Stent (4)',
    5: 'Drug Balloon Angioplasty + Stent Insertion (5)',
    6: 'Drug Balloon Angioplasty + Drug-eluting Stent (6)',
    7: 'Any Surgery + Plaque Rotablator (7)'
}
diabetes_options = {0: 'No (0)', 1: 'Yes (1)'}
preoperative_dependence_options = {0: 'No (0)', 1: 'Yes (1)'}

# 定义特征名
feature_names = [
    "High Altitude", "Femoral Endarterectomy", "Activated Partial Thromboplastin Time", 
    "White Blood Cell", "Monocyte", "Neutrophil", "Lymphocyte", "Cancer", "CHD", 
    "Surgical Method", "Diabetes Mellitus", "Preoperative Complete Dependence"
]

# Streamlit 用户界面
st.title("Surgical Risk Predictor")

# 高海拔地区选择
high_altitude = st.selectbox("High-altitude region:", options=list(high_altitude_options.keys()), format_func=lambda x: high_altitude_options[x])

# Femoral Endarterectomy 手术选择
femoral_endarterectomy = st.selectbox("Femoral Endarterectomy (YES: 1, NO: 0):", options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (0)')

# APTT 选择
aptt = st.selectbox("Activated Partial Thromboplastin Time:", options=list(aptt_options.keys()), format_func=lambda x: aptt_options[x])

# 白细胞数量
wbc = st.selectbox("White Blood Cell:", options=list(wbc_options.keys()), format_func=lambda x: wbc_options[x])

# 单核细胞数量
monocyte = st.selectbox("Monocyte:", options=list(monocyte_options.keys()), format_func=lambda x: monocyte_options[x])

# 中性粒细胞数量
neutrophil = st.selectbox("Neutrophil:", options=list(neutrophil_options.keys()), format_func=lambda x: neutrophil_options[x])

# 淋巴细胞数量
lymphocyte = st.selectbox("Lymphocyte:", options=list(lymphocyte_options.keys()), format_func=lambda x: lymphocyte_options[x])

# 癌症病史
cancer = st.selectbox("Cancer (YES: 1, NO: 0):", options=list(cancer_options.keys()), format_func=lambda x: cancer_options[x])

# 冠心病
chd = st.selectbox("CHD (YES: 1, NO: 0):", options=list(chd_options.keys()), format_func=lambda x: chd_options[x])

# 手术方法选择
surgical_method = st.selectbox("Surgical Method:", options=list(surgical_method_options.keys()), format_func=lambda x: surgical_method_options[x])

# 糖尿病
diabetes = st.selectbox("Diabetes Mellitus (YES: 1, NO: 0):", options=list(diabetes_options.keys()), format_func=lambda x: diabetes_options[x])

# 术前完全依赖选择
preoperative_dependence = st.selectbox("Preoperative Complete Dependence (YES: 1, NO: 0):", options=list(preoperative_dependence_options.keys()), format_func=lambda x: preoperative_dependence_options[x])

# 处理用户输入并进行预测
feature_values = [high_altitude, femoral_endarterectomy, aptt, wbc, monocyte, neutrophil, lymphocyte, cancer, chd, surgical_method, diabetes, preoperative_dependence]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (f"According to our model, you have a high surgical risk. "
                  f"The model predicts that your probability of having a high risk is {probability:.1f}%. "
                  "While this is just an estimate, it suggests that you may face significant risks. "
                  "I recommend that you consult with your surgical team for further evaluation.")
    else:
        advice = (f"According to our model, you have a low surgical risk. "
                  f"The model predicts that your probability of having a low risk is {probability:.1f}%. "
                  "However, regular monitoring and medical consultations are still important.")
    st.write(advice)
    
    # 计算SHAP值并显示force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


# In[ ]:




