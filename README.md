# LightRAG Demo

基于 [LightRAG](https://github.com/HKUDS/LightRAG) 的本地知识库问答系统，支持图谱增强检索。

## 项目结构

```
lightrag-demo/
├── day1_ingest.py        # Day1: 跑通 LightRAG，导入文档
├── api/main.py           # Day3: FastAPI 上传/查询接口
├── ui/app.py             # Day4: Streamlit 前端
├── experiments/          # Day5: 与普通 RAG 对比实验
├── docs/                 # 放你要导入的 .txt 文档
├── data/                 # LightRAG 索引存储（自动生成）
├── assets/               # 架构图、截图
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# Day1: 导入文档
python day1_ingest.py

# Day3: 启动 API
uvicorn api.main:app --reload

# Day4: 启动 UI
streamlit run ui/app.py
```

## 检索模式对比

| 模式 | 说明 |
|------|------|
| naive | 普通向量检索 |
| local | 局部图谱检索 |
| global | 全局图谱检索 |
| hybrid | 混合（推荐） |

## 技术栈

- **LightRAG** — 图谱增强 RAG 框架
- **FastAPI** — 后端接口
- **Streamlit** — 前端 UI
- **OpenAI** — LLM + Embedding
