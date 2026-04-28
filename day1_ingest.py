"""
Day 1: 跑通 LightRAG，本地导入文档
用法: python day1_ingest.py
"""
import asyncio
import os
from lightrag import LightRAG, QueryParam
from config import deepseek_llm, EMBEDDING_FUNC

WORKING_DIR = "./data/lightrag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=deepseek_llm,
    embedding_func=EMBEDDING_FUNC,
)

async def main():
    await rag.initialize_storages()

    # 导入 docs/ 下所有 .txt 文件
    docs_dir = "./docs"
    inserted = 0
    for fname in os.listdir(docs_dir):
        if fname.endswith(".txt"):
            path = os.path.join(docs_dir, fname)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            await rag.ainsert(text)
            print(f"✓ 导入: {fname}")
            inserted += 1

    if inserted == 0:
        print("docs/ 里没有 .txt 文件，先放一个进去再跑")
        return

    # 测试四种检索模式
    q = "这些文档的核心主题是什么？"
    for mode in ["naive", "local", "global", "hybrid"]:
        print(f"\n--- [{mode}] ---")
        result = await rag.aquery(q, param=QueryParam(mode=mode))
        print(result[:300] if result else "(无结果)")

if __name__ == "__main__":
    asyncio.run(main())
