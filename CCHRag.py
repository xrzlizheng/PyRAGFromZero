#!/usr/bin/env python
# coding: utf-8

"""
# 上下文分块标题（CCH）在简单RAG中的应用

检索增强生成（RAG）通过检索相关外部知识来提高语言模型的事实准确性。然而，标准分块经常会丢失重要上下文，导致检索效果不佳。

上下文分块标题（CCH）通过在嵌入之前为每个分块添加高级上下文（如文档标题或章节标题）来增强RAG。这提高了检索质量并防止了上下文外的响应。

## 步骤：

1. **数据摄取**：加载和预处理文本数据
2. **带上下文标题的分块**：提取章节标题并将其添加到分块前
3. **嵌入创建**：将上下文增强的分块转换为数值表示
4. **语义搜索**：根据用户查询检索相关分块
5. **响应生成**：使用语言模型从检索到的文本生成响应
6. **评估**：使用评分系统评估响应准确性
"""

import os
import numpy as np
import json
from openai import OpenAI
import fitz
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并打印前`num_chars`个字符

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

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)

def generate_chunk_header(chunk, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    使用LLM为给定文本块生成标题/头部

    参数：
    chunk (str): 要生成标题的文本块
    model (str): 用于生成标题的模型。默认为"meta-llama/Llama-3.2-3B-Instruct"

    返回：
    str: 生成的标题/头部
    """
    system_prompt = "Generate a concise and informative title for the given text."
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )

    return response.choices[0].message.content.strip()

def chunk_text_with_headers(text, n, overlap):
    """
    将文本分割成较小的片段并生成标题

    参数：
    text (str): 要分块的完整文本
    n (int): 以字符为单位的块大小
    overlap (int): 块之间的重叠字符数

    返回：
    List[dict]: 包含'header'和'text'键的字典列表
    """
    chunks = []

    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        header = generate_chunk_header(chunk)
        chunks.append({"header": header, "text": chunk})

    return chunks

pdf_path = "data/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text_with_headers(extracted_text, 1000, 200)

print("Sample Chunk:")
print("Header:", text_chunks[0]['header'])
print("Content:", text_chunks[0]['text'])

def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    为给定文本创建嵌入

    参数：
    text (str): 要嵌入的输入文本
    model (str): 要使用的嵌入模型。默认为"BAAI/bge-en-icl"

    返回：
    dict: 包含输入文本嵌入的响应
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

embeddings = []

for chunk in tqdm(text_chunks, desc="Generating embeddings"):
    text_embedding = create_embeddings(chunk["text"])
    header_embedding = create_embeddings(chunk["header"])
    embeddings.append({"header": chunk["header"], "text": chunk["text"], "embedding": text_embedding, "header_embedding": header_embedding})

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度

    参数：
    vec1 (np.ndarray): 第一个向量
    vec2 (np.ndarray): 第二个向量

    返回：
    float: 余弦相似度得分
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, chunks, k=5):
    """
    基于查询搜索最相关的分块

    参数：
    query (str): 用户查询
    chunks (List[dict]): 包含嵌入的文本块列表
    k (int): 返回的顶部结果数量

    返回：
    List[dict]: 最相关的k个分块
    """
    query_embedding = create_embeddings(query)
    similarities = []
    
    for chunk in chunks:
        sim_text = cosine_similarity(np.array(query_embedding), np.array(chunk["embedding"]))
        sim_header = cosine_similarity(np.array(query_embedding), np.array(chunk["header_embedding"]))
        avg_similarity = (sim_text + sim_header) / 2
        similarities.append((chunk, avg_similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in similarities[:k]]

with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']
top_chunks = semantic_search(query, embeddings, k=2)

print("Query:", query)
for i, chunk in enumerate(top_chunks):
    print(f"Header {i+1}: {chunk['header']}")
    print(f"Content:\n{chunk['text']}\n")

system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于系统提示和用户消息生成AI模型的响应

    参数：
    system_prompt (str): 指导AI行为的系统提示
    user_message (str): 用户的消息或查询
    model (str): 用于生成响应的模型。默认为"meta-llama/Llama-3.2-3B-Instruct"

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

user_prompt = "\n".join([f"Header: {chunk['header']}\nContent:\n{chunk['text']}" for chunk in top_chunks])
user_prompt = f"{user_prompt}\nQuestion: {query}"
ai_response = generate_response(system_prompt, user_prompt)

evaluate_system_prompt = """You are an intelligent evaluation system. 
Assess the AI assistant's response based on the provided context. 
- Assign a score of 1 if the response is very close to the true answer. 
- Assign a score of 0.5 if the response is partially correct. 
- Assign a score of 0 if the response is incorrect.
Return only the score (0, 0.5, or 1)."""

true_answer = data[0]['ideal_answer']
evaluation_prompt = f"""
User Query: {query}
AI Response: {ai_response}
True Answer: {true_answer}
{evaluate_system_prompt}
"""

evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print("Evaluation Score:", evaluation_response.choices[0].message.content)

