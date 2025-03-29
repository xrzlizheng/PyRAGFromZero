#!/usr/bin/env python
# coding: utf-8

"""
# Reranking for Enhanced RAG Systems

## 重排技术增强RAG系统

本脚本实现了重排技术，用于提升RAG系统中的检索质量。重排作为初始检索后的第二道过滤步骤，确保使用最相关的内容进行响应生成。

## 重排的关键概念

1. **初始检索**：第一轮使用基本相似度搜索（速度较快但精度较低）
2. **文档评分**：评估每个检索到的文档与查询的相关性
3. **重新排序**：根据相关性分数对文档进行排序
4. **选择**：仅使用最相关的文档进行响应生成
"""

import fitz
import os
import numpy as np
import json
from openai import OpenAI
import re


def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本
    
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
    将给定文本分块，每块n个字符，有重叠部分
    
    参数：
    text (str): 待分块的文本
    n (int): 每块的字符数
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


class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储
    """
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def similarity_search(self, query_embedding, k=5):
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        
        return results


def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    使用指定的OpenAI模型为给定文本创建嵌入
    
    参数：
    text (str): 需要创建嵌入的输入文本
    model (str): 用于创建嵌入的模型
    
    返回：
    List[float]: 嵌入向量
    """
    input_text = text if isinstance(text, list) else [text]
    
    response = client.embeddings.create(
        model=model,
        input=input_text
    )
    
    if isinstance(text, str):
        return response.data[0].embedding
    
    return [item.embedding for item in response.data]


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    处理RAG文档
    
    参数：
    pdf_path (str): PDF文件路径
    chunk_size (int): 每个块的字符数
    chunk_overlap (int): 块之间的重叠字符数
    
    返回：
    SimpleVectorStore: 包含文档块及其嵌入的向量存储
    """
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")
    
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)
    
    store = SimpleVectorStore()
    
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    
    print(f"Added {len(chunks)} chunks to the vector store")
    return store


def rerank_with_llm(query, results, top_n=3, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    使用LLM进行重排
    
    参数：
        query (str): 用户查询
        results (List[Dict]): 初始搜索结果
        top_n (int): 重排后返回的结果数量
        model (str): 用于评分的模型
        
    返回：
        List[Dict]: 重排后的结果
    """
    print(f"Reranking {len(results)} documents...")
    
    scored_results = []
    
    system_prompt = """You are an expert at evaluating document relevance for search queries.
Your task is to rate documents on a scale from 0 to 10 based on how well they answer the given query.

Guidelines:
- Score 0-2: Document is completely irrelevant
- Score 3-5: Document has some relevant information but doesn't directly answer the query
- Score 6-8: Document is relevant and partially answers the query
- Score 9-10: Document is highly relevant and directly answers the query

You MUST respond with ONLY a single integer score between 0 and 10. Do not include ANY other text."""
    
    for i, result in enumerate(results):
        if i % 5 == 0:
            print(f"Scoring document {i+1}/{len(results)}...")
        
        user_prompt = f"""Query: {query}

Document:
{result['text']}

Rate this document's relevance to the query on a scale from 0 to 10:"""
        
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        score_text = response.choices[0].message.content.strip()
        
        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            print(f"Warning: Could not extract score from response: '{score_text}', using similarity score instead")
            score = result["similarity"] * 10
        
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })
    
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
    
    return reranked_results[:top_n]


def rerank_with_keywords(query, results, top_n=3):
    """
    基于关键词匹配和位置的简单重排方法
    
    参数：
        query (str): 用户查询
        results (List[Dict]): 初始搜索结果
        top_n (int): 重排后返回的结果数量
        
    返回：
        List[Dict]: 重排后的结果
    """
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    
    scored_results = []
    
    for result in results:
        document_text = result["text"].lower()
        
        base_score = result["similarity"] * 0.5
        
        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                keyword_score += 0.1
                
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:
                    keyword_score += 0.1
                
                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)
        
        final_score = base_score + keyword_score
        
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })
    
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
    
    return reranked_results[:top_n]


def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于查询和上下文生成响应
    
    参数：
        query (str): 用户查询
        context (str): 检索到的上下文
        model (str): 用于响应生成的模型
        
    返回：
        str: 生成的响应
    """
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."
    
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
    """
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content


def rag_with_reranking(query, vector_store, reranking_method="llm", top_n=3, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    包含重排的完整RAG流程
    
    参数：
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        reranking_method (str): 重排方法（'llm' 或 'keywords'）
        top_n (int): 重排后返回的结果数量
        model (str): 响应生成模型
        
    返回：
        Dict: 结果，包括查询、上下文和响应
    """
    query_embedding = create_embeddings(query)
    
    initial_results = vector_store.similarity_search(query_embedding, k=10)
    
    if reranking_method == "llm":
        reranked_results = rerank_with_llm(query, initial_results, top_n=top_n)
    elif reranking_method == "keywords":
        reranked_results = rerank_with_keywords(query, initial_results, top_n=top_n)
    else:
        reranked_results = initial_results[:top_n]
    
    context = "\n\n===\n\n".join([result["text"] for result in reranked_results])
    
    response = generate_response(query, context, model)
    
    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }


with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']
reference_answer = data[0]['ideal_answer']
pdf_path = "data/AI_Information.pdf"


vector_store = process_document(pdf_path)

query = "Does AI have the potential to transform the way we live and work?"

print("Comparing retrieval methods...")

print("\n=== STANDARD RETRIEVAL ===")
standard_results = rag_with_reranking(query, vector_store, reranking_method="none")
print(f"\nQuery: {query}")
print(f"\nResponse:\n{standard_results['response']}")

print("\n=== LLM-BASED RERANKING ===")
llm_results = rag_with_reranking(query, vector_store, reranking_method="llm")
print(f"\nQuery: {query}")
print(f"\nResponse:\n{llm_results['response']}")

print("\n=== KEYWORD-BASED RERANKING ===")
keyword_results = rag_with_reranking(query, vector_store, reranking_method="keywords")
print(f"\nQuery: {query}")
print(f"\nResponse:\n{keyword_results['response']}")


def evaluate_reranking(query, standard_results, reranked_results, reference_answer=None):
    """
    评估重排结果的质量
    
    参数：
        query (str): 用户查询
        standard_results (Dict): 标准检索结果
        reranked_results (Dict): 重排后的检索结果
        reference_answer (str, optional): 用于比较的参考答案
        
    返回：
        str: 评估输出
    """
    system_prompt = """You are an expert evaluator of RAG systems.
    Compare the retrieved contexts and responses from two different retrieval methods.
    Assess which one provides better context and a more accurate, comprehensive answer."""
    
    comparison_text = f"""Query: {query}

Standard Retrieval Context:
{standard_results['context'][:1000]}... [truncated]

Standard Retrieval Answer:
{standard_results['response']}

Reranked Retrieval Context:
{reranked_results['context'][:1000]}... [truncated]

Reranked Retrieval Answer:
{reranked_results['response']}"""

    if reference_answer:
        comparison_text += f"""
        
Reference Answer:
{reference_answer}"""

    user_prompt = f"""
{comparison_text}

Please evaluate which retrieval method provided:
1. More relevant context
2. More accurate answer
3. More comprehensive answer
4. Better overall performance

Provide a detailed analysis with specific examples.
"""
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content


evaluation = evaluate_reranking(
    query=query,
    standard_results=standard_results,
    reranked_results=llm_results,
    reference_answer=reference_answer
)

print("\n=== EVALUATION RESULTS ===")
print(evaluation)

