"""
Day 4: Streamlit UI
启动: streamlit run ui/app.py
"""
import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="LightRAG Demo", page_icon="🔍")
st.title("🔍 LightRAG 知识库问答")

with st.sidebar:
    st.header("上传文档")
    f = st.file_uploader("选择 .txt 文件", type=["txt"])
    if f and st.button("导入"):
        r = requests.post(f"{API}/upload", files={"file": (f.name, f, "text/plain")})
        st.success("✓ 导入成功" if r.ok else f"✗ {r.text}")

st.header("提问")
mode = st.selectbox("检索模式", ["hybrid", "local", "global", "naive"])
question = st.text_input("输入问题")

if st.button("查询") and question:
    with st.spinner("检索中..."):
        r = requests.post(f"{API}/query", json={"question": question, "mode": mode})
    if r.ok:
        st.markdown(r.json()["answer"])
    else:
        st.error(r.text)
