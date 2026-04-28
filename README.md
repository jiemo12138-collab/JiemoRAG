# JiemoRAG

基于 [LightRAG](https://github.com/HKUDS/LightRAG) 的本地知识库问答系统，支持图谱增强检索，上传文档后通过自然语言提问，快速获取相关内容。

## 项目结构

```
lightrag-demo/
├── api/main.py       # FastAPI 后端，提供上传/查询接口
├── frontend/         # 前端页面
├── config.py         # 配置文件
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 配置 .env
DEEPSEEK_API_KEY=your_key

# 启动后端
uvicorn api.main:app --reload
```

然后打开 `frontend/index.html` 即可使用。

## 检索模式

| 模式 | 说明 |
|------|------|
| naive | 普通向量检索 |
| local | 局部图谱检索 |
| global | 全局图谱检索 |
| hybrid | 混合（推荐） |

## 技术栈

- **LightRAG** — 图谱增强 RAG 框架
- **FastAPI** — 后端接口
- **DeepSeek** — LLM
- **BAAI/BGE-M3** — 本地 Embedding 模型
