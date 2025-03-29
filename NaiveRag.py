#!/usr/bin/env python
# coding: utf-8

"""
# 简单RAG介绍
# Retrieval-Augmented Generation (RAG)是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，提高准确性和事实正确性。
# 
# 在一个简单的RAG设置中，我们遵循以下步骤：
# 1. 数据导入：加载并预处理文本数据
# 2. 分块：将数据分成更小的块以提高检索性能
# 3. 嵌入创建：使用嵌入模型将文本块转换为数字表示
# 4. 语义搜索：根据用户查询检索相关块
# 5. 响应生成：使用语言模型基于检索到的文本生成响应
# 
# 本脚本实现了一个简单的RAG方法，评估模型的响应，并探索各种改进。
"""

import fitz
import os
import numpy as np
import json
from openai import OpenAI

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并返回提取的文本内容。

    参数：
    pdf_path (str): PDF文件路径

    返回：
    str: 从PDF中提取的文本
    """
    mypdf = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text

def chunk_text(text, n, overlap):
    """
    将给定文本分割成指定长度的块，并带有重叠部分。

    参数：
    text (str): 待分割的文本
    n (int): 每个块的长度
    overlap (int): 块之间的重叠字符数

    返回：
    List[str]: 文本块列表
    """
    chunks = []
    
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])

    return chunks

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)

pdf_path = "data/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(extracted_text, 1000, 200)

print("Number of text chunks:", len(text_chunks))
print("\nFirst text chunk:")
print(text_chunks[0])

def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    使用指定的OpenAI模型为给定文本创建嵌入向量。

    参数：
    text (str): 输入文本
    model (str): 用于创建嵌入的模型，默认为"BAAI/bge-en-icl"

    返回：
    dict: 包含嵌入向量的OpenAI API响应
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response

response = create_embeddings(text_chunks)

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    参数：
    vec1 (np.ndarray): 第一个向量
    vec2 (np.ndarray): 第二个向量

    返回：
    float: 两个向量之间的余弦相似度
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    使用给定的查询和嵌入向量对文本块进行语义搜索。

    参数：
    query (str): 搜索查询
    text_chunks (List[str]): 文本块列表
    embeddings (List[dict]): 文本块的嵌入向量列表
    k (int): 返回的最相关文本块数量，默认为5

    返回：
    List[str]: 基于查询的最相关文本块列表
    """
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []

    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]

with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']
top_chunks = semantic_search(query, text_chunks, response.data, k=2)

print("Query:", query)
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")

system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于系统提示和用户消息生成AI模型的响应。

    参数：
    system_prompt (str): 指导AI行为的系统提示
    user_message (str): 用户消息或查询
    model (str): 用于生成响应的模型，默认为"meta-llama/Llama-2-7B-chat-hf"

    返回：
    dict: AI模型的响应
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"
ai_response = generate_response(system_prompt, user_prompt)

evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."

evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

print(evaluation_response.choices[0].message.content)

