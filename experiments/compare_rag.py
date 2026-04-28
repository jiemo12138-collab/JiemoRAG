"""
Day 5: LightRAG vs 普通 RAG 对比实验
"""
import asyncio
import time
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

WORKING_DIR = "./data/lightrag_storage"

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
)

TEST_QUESTIONS = [
    "这些文档的核心主题是什么？",
    "文档中提到了哪些关键概念之间的关系？",
    "请总结文档中最重要的结论。",
]

MODES = ["naive", "local", "global", "hybrid"]

async def run_experiment():
    results = []
    for q in TEST_QUESTIONS:
        row = {"question": q}
        for mode in MODES:
            t0 = time.time()
            ans = await rag.aquery(q, param=QueryParam(mode=mode))
            elapsed = round(time.time() - t0, 2)
            row[mode] = {"answer": ans[:120] + "...", "time": elapsed}
        results.append(row)

    for r in results:
        print(f"\n问题: {r['question']}")
        for mode in MODES:
            print(f"  [{mode}] ({r[mode]['time']}s) {r[mode]['answer']}")

if __name__ == "__main__":
    asyncio.run(run_experiment())
