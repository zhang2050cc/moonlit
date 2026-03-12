import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.set_page_config(page_title="鸢尾花数据集探索", layout="wide")
st.title("🌼 鸢尾花数据集探索")

# 加载数据（带缓存）
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    return df

df = load_data()

# 侧边栏选择特征
st.sidebar.header("选择特征")
x_feature = st.sidebar.selectbox("X轴", df.columns[:-1])
y_feature = st.sidebar.selectbox("Y轴", df.columns[:-1], index=1)

# 主区域分成两列
col1, col2 = st.columns(2)

with col1:
    st.subheader("散点图")
    fig, ax = plt.subplots()
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        ax.scatter(subset[x_feature], subset[y_feature], label=species, alpha=0.7)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("数据预览")
    st.dataframe(df)

st.markdown("---")
st.write("该数据集包含150个样本，每个样本有4个特征。")