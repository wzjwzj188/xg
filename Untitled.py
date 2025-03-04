#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[7]:


df = pd.read_excel("C:/Users/wenzhongjian/Desktop/DATA/data.xlsx")


# In[8]:


# 划分特征和目标变量
X = df.drop(['class'], axis=1)
y = df['class']


# In[9]:


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,                                                    
                                                    random_state=42, stratify=df['class'])
df.head()


# In[1]:


get_ipython().system('pip install xgboost')
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[10]:


# XGBoost模型参数
params_xgb = {
    'learning_rate': 0.02, 
    'booster': 'gbtree', 
    'objective': 'binary:logistic', 
    'max_leaves': 127, 
    'verbosity': 1,
    'seed': 42, 
    'nthread': -1, 
    'colsample_bytree': 0.6,
    'subsample': 0.7,
    'eval_metric': 'logloss'
}


# In[11]:


# 初始化XGBoost分类模型
model_xgb = xgb.XGBClassifier(**params_xgb)


# In[12]:


# 定义参数网格，用于网格搜索
param_grid = {    
    'n_estimators': [100, 200, 300, 400, 500],  # 树的数量    
    'max_depth': [3, 4, 5, 6, 7],               # 树的深度    
    'learning_rate': [0.01, 0.02, 0.05, 0.1],   # 学习率
}


# In[13]:


# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='neg_log_loss',  # 评价指标为负对数损失
    cv=5,                    # 5折交叉验证
     n_jobs=-1,               # 并行计算
    verbose=1                # 输出详细进度信息
)
# 训练模型
grid_search.fit(X_train, y_train)
# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)
# 使用最优参数训练模型
best_model = grid_search.best_estimator_


# In[14]:


#模型评价
from sklearn.metrics import classification_report
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[15]:


import joblib
# 保存模型
joblib.dump(best_model , 'XGBoost.pkl')


# In[20]:


get_ipython().system('pip install streamlit')


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


# In[ ]:


# Define feature options
cp_options = {1: 'Typical angina (1)',    
              2: 'Atypical angina (2)',    
              3: 'Non-anginal pain (3)',    
              4: 'Asymptomatic (4)'


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




