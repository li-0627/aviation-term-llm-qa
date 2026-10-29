# -*- coding: utf-8 -*-
"""从 MySQL 读取航空术语表，向量化写入 chroma_db（云端重建用）。
用法： python build_kb.py
"""
import os
import time
import chromadb
import pymysql
from http import HTTPStatus
import dashscope

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PWD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "")
MYSQL_TABLE = "term_main"

COLLECTION_NAME = "aviation_term"
CHROMA_DB_PATH = "./chroma_db"
BATCH_SIZE = 50

dashscope.api_key = DASHSCOPE_API_KEY


def get_embedding(text: str):
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v2,
        input=text
    )
    if resp.status_code == HTTPStatus.OK:
        return resp.output["embeddings"][0]["embedding"]
    raise Exception(f"向量接口调用失败:{resp.code} {resp.message}")


def build():
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB,
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT `english_name`, `term_name`, `explanation` FROM `{MYSQL_TABLE}`")
            rows = cur.fetchall()
    finally:
        conn.close()

    total = len(rows)
    print(f"从 MySQL 读到 {total} 条记录")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll = client.get_or_create_collection(name=COLLECTION_NAME)
    existing = coll.count()
    if existing > 0:
        print(f"清空已有 {existing} 条")
        coll.delete(ids=coll.get()["ids"])

    batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []
    written, failed, t0 = 0, 0, time.time()
    for idx, row in enumerate(rows):
        term_en = str(row.get("english_name", "") or "").strip()
        term_zh = str(row.get("term_name", "") or "").strip()
        explain = str(row.get("explanation", "") or "").strip()
        if not term_zh:
            continue
        doc_text = f"航空术语（英文：{term_en}）：{term_zh}\n释义：{explain}"
        try:
            emb = get_embedding(doc_text)
        except Exception as e:
            failed += 1
            if failed <= 5 or idx % 500 == 0:
                print(f"  跳过第{idx}条: {e}")
            continue
        batch_ids.append(f"term_{idx}")
        batch_docs.append(doc_text)
        batch_embs.append(emb)
        batch_metas.append({"term_name": term_zh})
        if len(batch_ids) >= BATCH_SIZE:
            coll.add(ids=batch_ids, documents=batch_docs, embeddings=batch_embs, metadatas=batch_metas)
            written += len(batch_ids)
            batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []
            if written % 500 == 0:
                print(f"  进度: 已写入 {written}/{total}，耗时 {time.time()-t0:.1f}s")
    if batch_ids:
        coll.add(ids=batch_ids, documents=batch_docs, embeddings=batch_embs, metadatas=batch_metas)
        written += len(batch_ids)
    print(f"✅ 向量库构建完成，写入 {written} 条，跳过 {failed} 条，耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    build()
