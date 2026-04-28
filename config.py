"""
统一配置：DeepSeek LLM + 本地 BGE-M3 Embedding
"""
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── DeepSeek 客户端 ───────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
    return _client

# LightRAG v1.4.x 会传 keyword_extraction / token_tracker 等内部参数
_OPENAI_PARAMS = {"temperature", "max_tokens", "top_p", "stream",
                  "frequency_penalty", "presence_penalty", "seed", "stop", "n"}

async def deepseek_llm(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,   # LightRAG 具名参数，接住但不传给 OpenAI
    token_tracker=None,         # 同上
    **kwargs,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 只保留 OpenAI 标准参数
    clean = {k: v for k, v in kwargs.items() if k in _OPENAI_PARAMS}

    # keyword_extraction 时要求 JSON 输出，用 DeepSeek 支持的格式
    if keyword_extraction:
        clean["response_format"] = {"type": "json_object"}

    resp = await _get_client().chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        **clean,
    )
    return resp.choices[0].message.content


# ── 本地 Embedding（BAAI/bge-m3）────────────────────────────
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("加载 BGE-M3 模型...")
        _embed_model = SentenceTransformer("BAAI/bge-m3")
    return _embed_model

async def local_embedding(texts: list[str]):
    # 返回 numpy array，LightRAG 内部需要 .size 等 numpy 属性
    return _get_embed_model().encode(texts, normalize_embeddings=True)

EMBEDDING_FUNC = EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=8192,
    func=local_embedding,
)
