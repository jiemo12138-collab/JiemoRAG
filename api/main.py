"""
Day 3: FastAPI 接口 —— 上传文档 + 查询
启动: uvicorn api.main:app --reload
"""
import os
import sys
import json
import asyncio
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from lightrag import LightRAG, QueryParam
from config import deepseek_llm, EMBEDDING_FUNC

WORKING_DIR = "./data/lightrag_storage"
DOC_MAP_FILE = "./data/lightrag_storage/doc_filename_map.json"
os.makedirs(WORKING_DIR, exist_ok=True)

def load_doc_map() -> dict:
    if os.path.exists(DOC_MAP_FILE):
        with open(DOC_MAP_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_doc_map(doc_map: dict):
    with open(DOC_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, ensure_ascii=False, indent=2)

def compute_doc_id(content: str) -> str:
    return "doc-" + hashlib.md5(content.encode()).hexdigest()

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=deepseek_llm,
    embedding_func=EMBEDDING_FUNC,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag.initialize_storages()
    yield

app = FastAPI(title="MindBase API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="只支持 .txt 文件")
    content = (await file.read()).decode("utf-8")
    await rag.ainsert(content)
    doc_map = load_doc_map()
    doc_map[file.filename] = compute_doc_id(content)
    save_doc_map(doc_map)
    return {"status": "ok", "filename": file.filename}


class DeleteRequest(BaseModel):
    filename: str

@app.post("/delete-doc")
async def delete_doc(req: DeleteRequest):
    doc_map = load_doc_map()
    if req.filename not in doc_map:
        raise HTTPException(status_code=404, detail="未找到该文档记录")
    doc_id = doc_map[req.filename]
    await rag.adelete_by_doc_id(doc_id)
    del doc_map[req.filename]
    save_doc_map(doc_map)
    return {"status": "ok"}


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"


@app.post("/query")
async def query(req: QueryRequest):
    result = await rag.aquery(req.question, param=QueryParam(mode=req.mode))
    return {"answer": result}


@app.post("/query-stream")
async def query_stream(req: QueryRequest):
    result = await rag.aquery(req.question, param=QueryParam(mode=req.mode))
    text = result or ""
    delay = min(0.012, 5.0 / max(len(text), 1))

    async def event_stream():
        for char in text:
            yield f"data: {json.dumps({'t': char})}\n\n"
            await asyncio.sleep(delay)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"status": "running"}
