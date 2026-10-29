# -*- coding: utf-8 -*-
import os
import chromadb
import dashscope
from http import HTTPStatus

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
COLLECTION_NAME = "aviation_term"
CHROMA_DB_PATH = "./chroma_db"

dashscope.api_key = DASHSCOPE_API_KEY


def get_embedding(text: str):
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v2,
        input=text
    )
    if resp.status_code == HTTPStatus.OK:
        return resp.output["embeddings"][0]["embedding"]
    raise Exception(f"向量生成失败:{resp.code} {resp.message}")


def search_knowledge(query: str, top_n=10):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll = client.get_collection(name=COLLECTION_NAME)
    q_emb = get_embedding(query)
    result = coll.query(query_embeddings=[q_emb], n_results=top_n)
    context = ""
    for doc, dist in zip(result["documents"][0], result["distances"][0]):
        doc_str = str(doc).strip()
        if (doc_str and "nan" not in doc_str and dist < 1.2 and "释义：" in doc_str):
            context += doc_str + "\n"
    return context


def llm_ask(user_query: str):
    knowledge_context = search_knowledge(user_query)
    prompt = (
        "你是航空术语翻译助手，严格依据下面知识库材料完成术语翻译。"
        "知识库材料：" + knowledge_context +
        " 用户问题：" + user_query +
        " 规则：1.如果用户输入英文航空术语，输出对应的标准中文译法；"
        " 2.如果用户输入中文航空术语，输出对应的标准英文术语；"
        " 3.只输出翻译结果，不要输出释义、不要输出多余解释文字；"
        " 4.如果知识库没有该术语，直接回复【未查询到该航空术语】，禁止编造。"
    )
    resp = dashscope.Generation.call(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        result_format="message"
    )
    if resp.status_code == HTTPStatus.OK:
        return resp.output.choices[0].message.content
    return f"调用大模型失败：{resp.message}"
