#!/usr/bin/env python
# coding: utf-8

# # 融合检索：结合向量搜索和关键词搜索
# 
# 在这个脚本中，我实现了一个融合检索系统，它结合了语义向量搜索和基于关键词的BM25检索的优势。这种方法通过同时捕获概念相似性和精确关键词匹配来提高检索质量。
# 
# ## 为什么融合检索很重要
# 
# 传统的RAG系统通常仅依赖于向量搜索，但这有一些局限性：
# 
# - 向量搜索在语义相似性方面表现出色，但可能会错过精确的关键词匹配
# - 关键词搜索对特定术语很有效，但缺乏语义理解
# - 不同的查询在不同的检索方法下表现更好
# 
# 融合检索通过以下方式为我们提供了两全其美的解决方案：
# 
# - 同时执行基于向量和基于关键词的检索
# - 对每种方法的分数进行归一化处理
# - 使用加权公式将它们组合起来
# - 基于组合分数对文档进行排序

# ## 设置环境
# 首先导入必要的库。

import os
import numpy as np
from rank_bm25 import BM25Okapi
import fitz
from openai import OpenAI
import re
import json
import time
from sklearn.metrics.pairwise import cosine_similarity


# ## 设置OpenAI API客户端
# 我们初始化OpenAI客户端以生成嵌入和响应。

# 使用基础URL和API密钥初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)


# ## 文档处理函数

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容。
    
    参数:
        pdf_path (str): PDF文件的路径
        
    返回:
        str: 提取的文本内容
    """
    print(f"Extracting text from {pdf_path}...")  # 打印正在处理的PDF路径
    pdf_document = fitz.open(pdf_path)  # 使用PyMuPDF打开PDF文件
    text = ""  # 初始化一个空字符串来存储提取的文本
    
    # 遍历PDF中的每一页
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]  # 获取页面对象
        text += page.get_text()  # 从页面中提取文本并添加到文本字符串中
    
    return text  # 返回提取的文本内容


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    将文本分割成重叠的块。
    
    参数:
        text (str): 要分块的输入文本
        chunk_size (int): 每个块的字符大小
        chunk_overlap (int): 块之间的重叠字符数
        
    返回:
        List[Dict]: 包含文本和元数据的块列表
    """
    chunks = []  # 初始化一个空列表来存储块
    
    # 使用指定的块大小和重叠量遍历文本
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]  # 提取指定大小的块
        if chunk:  # 确保我们不添加空块
            chunk_data = {
                "text": chunk,  # 块文本
                "metadata": {
                    "start_char": i,  # 块的起始字符索引
                    "end_char": i + len(chunk)  # 块的结束字符索引
                }
            }
            chunks.append(chunk_data)  # 将块数据添加到列表中
    
    print(f"Created {len(chunks)} text chunks")  # 打印创建的块数量
    return chunks  # 返回块列表


def clean_text(text):
    """
    通过移除多余的空白和特殊字符来清理文本。
    
    参数:
        text (str): 输入文本
        
    返回:
        str: 清理后的文本
    """
    # 将多个空白字符（包括换行符和制表符）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 修复常见的OCR问题，将制表符和换行符替换为空格
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')
    
    # 移除任何前导或尾随空白，并确保单词之间只有单个空格
    text = ' '.join(text.split())
    
    return text


# ## 创建我们的向量存储

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为给定文本创建嵌入。
    
    参数:
        texts (str or List[str]): 输入文本
        model (str): 嵌入模型名称
        
    返回:
        List[List[float]]: 嵌入向量
    """
    # 同时处理字符串和列表输入
    input_texts = texts if isinstance(texts, list) else [texts]
    
    # 如果需要，分批处理（OpenAI API限制）
    batch_size = 100
    all_embeddings = []
    
    # 分批遍历输入文本
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]  # 获取当前批次的文本
        
        # 为当前批次创建嵌入
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        # 从响应中提取嵌入
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 将批次嵌入添加到列表中
    
    # 如果输入是字符串，则只返回第一个嵌入
    if isinstance(texts, str):
        return all_embeddings[0]
    
    # 否则返回所有嵌入
    return all_embeddings


class SimpleVectorStore:
    """
    使用NumPy的简单向量存储实现。
    """
    def __init__(self):
        self.vectors = []  # 用于存储嵌入向量的列表
        self.texts = []  # 用于存储文本内容的列表
        self.metadata = []  # 用于存储元数据的列表
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加项目。
        
        参数:
            text (str): 文本内容
            embedding (List[float]): 嵌入向量
            metadata (Dict, optional): 额外的元数据
        """
        self.vectors.append(np.array(embedding))  # 添加嵌入向量
        self.texts.append(text)  # 添加文本内容
        self.metadata.append(metadata or {})  # 添加元数据（如果为None则为空字典）
    
    def add_items(self, items, embeddings):
        """
        向向量存储中添加多个项目。
        
        参数:
            items (List[Dict]): 文本项目列表
            embeddings (List[List[float]]): 嵌入向量列表
        """
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],  # 从项目中提取文本
                embedding=embedding,  # 使用对应的嵌入
                metadata={**item.get("metadata", {}), "index": i}  # 合并项目元数据与索引
            )
    
    def similarity_search_with_scores(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目及其相似度分数。
        
        参数:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 要返回的结果数量
            
        返回:
            List[Tuple[Dict, float]]: 前k个最相似的项目及其分数
        """
        if not self.vectors:
            return []  # 如果没有存储向量，则返回空列表
        
        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)
        
        # 使用余 cosine_similarity 计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = cosine_similarity([query_vector], [vector])[0][0]  # 计算余弦相似度
            similarities.append((i, similarity))  # 添加索引和相似度分数
        
        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果及其分数
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 通过索引检索文本
                "metadata": self.metadata[idx],  # 通过索引检索元数据
                "similarity": float(score)  # 添加相似度分数
            })
        
        return results
    
    def get_all_documents(self):
        """
        获取存储中的所有文档。
        
        返回:
            List[Dict]: 所有文档
        """
        return [{"text": text, "metadata": meta} for text, meta in zip(self.texts, self.metadata)]  # 组合文本和元数据


# ## BM25实现

def create_bm25_index(chunks):
    """
    从给定的块创建BM25索引。
    
    参数:
        chunks (List[Dict]): 文本块列表
        
    返回:
        BM25Okapi: BM25索引
    """
    # 从每个块中提取文本
    texts = [chunk["text"] for chunk in chunks]
    
    # 通过按空格分割来标记每个文档
    tokenized_docs = [text.split() for text in texts]
    
    # 使用标记化的文档创建BM25索引
    bm25 = BM25Okapi(tokenized_docs)
    
    # 打印BM25索引中的文档数量
    print(f"Created BM25 index with {len(texts)} documents")
    
    return bm25


def bm25_search(bm25, chunks, query, k=5):
    """
    使用查询搜索BM25索引。
    
    参数:
        bm25 (BM25Okapi): BM25索引
        chunks (List[Dict]): 文本块列表
        query (str): 查询字符串
        k (int): 要返回的结果数量
        
    返回:
        List[Dict]: 带有分数的前k个结果
    """
    # 通过将查询分割成单个单词来标记查询
    query_tokens = query.split()
    
    # 获取查询标记对索引文档的BM25分数
    scores = bm25.get_scores(query_tokens)
    
    # 初始化一个空列表来存储带有分数的结果
    results = []
    
    # 遍历分数和对应的块
    for i, score in enumerate(scores):
        # 创建元数据的副本以避免修改原始数据
        metadata = chunks[i].get("metadata", {}).copy()
        # 向元数据添加索引
        metadata["index"] = i
        
        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,  # 添加带有索引的元数据
            "bm25_score": float(score)
        })
    
    # 按BM25分数降序排序结果
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    
    # 返回前k个结果
    return results[:k]


# ## 融合检索函数

def fusion_retrieval(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    执行结合向量和BM25搜索的融合检索。
    
    参数:
        query (str): 查询字符串
        chunks (List[Dict]): 原始文本块
        vector_store (SimpleVectorStore): 向量存储
        bm25_index (BM25Okapi): BM25索引
        k (int): 要返回的结果数量
        alpha (float): 向量分数的权重(0-1)，其中1-alpha是BM25权重
        
    返回:
        List[Dict]: 基于组合分数的前k个结果
    """
    print(f"Performing fusion retrieval for query: {query}")
    
    # 定义小的epsilon值以避免除以零
    epsilon = 1e-8
    
    # 获取向量搜索结果
    query_embedding = create_embeddings(query)  # 为查询创建嵌入
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))  # 执行向量搜索
    
    # 获取BM25搜索结果
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))  # 执行BM25搜索
    
    # 创建字典将文档索引映射到分数
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    
    # 确保所有文档都有两种方法的分数
    all_docs = vector_store.get_all_documents()
    combined_results = []
    
    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)  # 获取向量分数，如果未找到则为0
        bm25_score = bm25_scores_dict.get(i, 0.0)  # 获取BM25分数，如果未找到则为0
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    
    # 将分数提取为数组
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
    
    # 归一化分数
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    
    # 计算组合分数
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores
    
    # 将组合分数添加到结果中
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)
    
    # 按组合分数排序（降序）
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # 返回前k个结果
    top_results = combined_results[:k]
    
    print(f"Retrieved {len(top_results)} documents with fusion retrieval")
    return top_results


# ## 文档处理流程

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    处理用于融合检索的文档。
    
    参数:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的字符大小
        chunk_overlap (int): 块之间的重叠字符数
        
    返回:
        Tuple[List[Dict], SimpleVectorStore, BM25Okapi]: 块、向量存储和BM25索引
    """
    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)
    
    # 清理提取的文本，移除多余的空白和特殊字符
    cleaned_text = clean_text(text)
    
    # 将清理后的文本分割成重叠的块
    chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
    
    # 从每个块中提取文本内容以创建嵌入
    chunk_texts = [chunk["text"] for chunk in chunks]
    print("Creating embeddings for chunks...")
    
    # 为块文本创建嵌入
    embeddings = create_embeddings(chunk_texts)
    
    # 初始化向量存储
    vector_store = SimpleVectorStore()
    
    # 将块及其嵌入添加到向量存储中
    vector_store.add_items(chunks, embeddings)
    print(f"Added {len(chunks)} items to vector store")
    
    # 从块创建BM25索引
    bm25_index = create_bm25_index(chunks)
    
    # 返回块、向量存储和BM25索引
    return chunks, vector_store, bm25_index


# ## 响应生成

def generate_response(query, context):
    """
    基于查询和上下文生成响应。
    
    参数:
        query (str): 用户查询
        context (str): 从检索文档中获取的上下文
        
    返回:
        str: 生成的响应
    """
    # 定义系统提示以指导AI助手
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
    If the context doesn't contain relevant information to answer the question fully, acknowledge this limitation."""

    # 使用上下文和查询格式化用户提示
    user_prompt = f"""Context:
    {context}

    Question: {query}

    Please answer the question based on the provided context."""

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定要使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 指导助手的系统消息
            {"role": "user", "content": user_prompt}  # 带有上下文和查询的用户消息
        ],
        temperature=0.1  # 设置响应生成的温度
    )
    
    # 返回生成的响应
    return response.choices[0].message.content


# ## 主检索函数

def answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    使用融合RAG回答查询。
    
    参数:
        query (str): 用户查询
        chunks (List[Dict]): 文本块
        vector_store (SimpleVectorStore): 向量存储
        bm25_index (BM25Okapi): BM25索引
        k (int): 要检索的文档数量
        alpha (float): 向量分数的权重
        
    返回:
        Dict: 包括检索文档和响应的查询结果
    """
    # 使用融合检索方法检索文档
    retrieved_docs = fusion_retrieval(query, chunks, vector_store, bm25_index, k=k, alpha=alpha)
    
    # 通过用分隔符连接文本来格式化检索文档的上下文
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    
    # 基于查询和格式化的上下文生成响应
    response = generate_response(query, context)
    
    # 返回查询、检索文档和生成的响应
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


# ## 比较检索方法

def vector_only_rag(query, vector_store, k=5):
    """
    仅使用基于向量的RAG回答查询。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要检索的文档数量
        
    返回:
        Dict: 查询结果
    """
    # 创建查询嵌入
    query_embedding = create_embeddings(query)
    
    # 使用基于向量的相似度搜索检索文档
    retrieved_docs = vector_store.similarity_search_with_scores(query_embedding, k=k)
    
    # 通过用分隔符连接文本来格式化检索文档的上下文
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    
    # 基于查询和格式化的上下文生成响应
    response = generate_response(query, context)
    
    # 返回查询、检索文档和生成的响应
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


def bm25_only_rag(query, chunks, bm25_index, k=5):
    """
    仅使用基于BM25的RAG回答查询。
    
    参数:
        query (str): 用户查询
        chunks (List[Dict]): 文本块
        bm25_index (BM25Okapi): BM25索引
        k (int): 要检索的文档数量
        
    返回:
        Dict: 查询结果
    """
    # 使用BM25搜索检索文档
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)
    
    # 通过用分隔符连接文本来格式化检索文档的上下文
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    
    # 基于查询和格式化的上下文生成响应
    response = generate_response(query, context)
    
    # 返回查询、检索文档和生成的响应
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


# ## 评估函数

def compare_retrieval_methods(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, reference_answer=None):
    """
    比较查询的不同检索方法。
    
    参数:
        query (str): 用户查询
        chunks (List[Dict]): 文本块
        vector_store (SimpleVectorStore): 向量存储
        bm25_index (BM25Okapi): BM25索引
        k (int): 要检索的文档数量
        alpha (float): 融合检索中向量分数的权重
        reference_answer (str, optional): 用于比较的参考答案
        
    返回:
        Dict: 比较结果
    """
    print(f"\n=== Comparing retrieval methods for query: {query} ===\n")
    
    # 运行仅向量RAG
    print("\nRunning vector-only RAG...")
    vector_result = vector_only_rag(query, vector_store, k)
    
    # 运行仅BM25 RAG
    print("\nRunning BM25-only RAG...")
    bm25_result = bm25_only_rag(query, chunks, bm25_index, k)
    
    # 运行融合RAG
    print("\nRunning fusion RAG...")
    fusion_result = answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k, alpha)
    
    # 比较不同检索方法的响应
    print("\nComparing responses...")
    comparison = evaluate_responses(
        query, 
        vector_result["response"], 
        bm25_result["response"], 
        fusion_result["response"],
        reference_answer
    )
    
    # 返回比较结果
    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result,
        "comparison": comparison
    }


def evaluate_responses(query, vector_response, bm25_response, fusion_response, reference_answer=None):
    """
    评估不同检索方法的响应。
    
    参数:
        query (str): 用户查询
        vector_response (str): 仅向量RAG的响应
        bm25_response (str): 仅BM25 RAG的响应
        fusion_response (str): 融合RAG的响应
        reference_answer (str, optional): 参考答案
        
    返回:
        str: 响应评估
    """
    # 评估者的系统提示，指导评估过程
    system_prompt = """You are an expert evaluator of RAG systems. Compare responses from three different retrieval approaches:
    1. Vector-based retrieval: Uses semantic similarity for document retrieval
    2. BM25 keyword retrieval: Uses keyword matching for document retrieval
    3. Fusion retrieval: Combines both vector and keyword approaches

    Evaluate the responses based on:
    - Relevance to the query
    - Factual correctness
    - Comprehensiveness
    - Clarity and coherence"""

    # 包含查询和响应的用户提示
    user_prompt = f"""Query: {query}

    Vector-based response:
    {vector_response}

    BM25 keyword response:
    {bm25_response}

    Fusion response:
    {fusion_response}
    """

    # 如果提供了参考答案，则将其添加到提示中
    if reference_answer:
        user_prompt += f"""
            Reference answer:
            {reference_answer}
        """

    # 向用户提示添加详细比较的指示
    user_prompt += """
    Please provide a detailed comparison of these three responses. Which approach performed best for this query and why?
    Be specific about the strengths and weaknesses of each approach for this particular query.
    """

    # 使用meta-llama/Llama-3.2-3B-Instruct生成评估
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定要使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 指导评估者的系统消息
            {"role": "user", "content": user_prompt}  # 带有查询和响应的用户消息
        ],
        temperature=0  # 设置响应生成的温度
    )
    
    # 返回生成的评估内容
    return response.choices[0].message.content


# ## 完整评估流程

def evaluate_fusion_retrieval(pdf_path, test_queries, reference_answers=None, k=5, alpha=0.5):
    """
    评估与其他方法相比的融合检索。
    
    参数:
        pdf_path (str): PDF文件的路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 参考答案
        k (int): 要检索的文档数量
        alpha (float): 融合检索中向量分数的权重
        
    返回:
        Dict: 评估结果
    """
    print("=== EVALUATING FUSION RETRIEVAL ===\n")
    
    # 处理文档以提取文本、创建块并构建向量和BM25索引
    chunks, vector_store, bm25_index = process_document(pdf_path)
    
    # 初始化一个列表来存储每个查询的结果
    results = []
    
    # 遍历每个测试查询
    for i, query in enumerate(test_queries):
        print(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        print(f"Query: {query}")
        
        # 获取参考答案（如果有）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        
        # 比较当前查询的检索方法
        comparison = compare_retrieval_methods(
            query, 
            chunks, 
            vector_store, 
            bm25_index, 
            k=k, 
            alpha=alpha,
            reference_answer=reference
        )
        
        # 将比较结果添加到结果列表中
        results.append(comparison)
        
        # 打印不同检索方法的响应
        print("\n=== Vector-based Response ===")
        print(comparison["vector_result"]["response"])
        
        print("\n=== BM25 Response ===")
        print(comparison["bm25_result"]["response"])
        
        print("\n=== Fusion Response ===")
        print(comparison["fusion_result"]["response"])
        
        print("\n=== Comparison ===")
        print(comparison["comparison"])
    
    # 生成融合检索性能的整体分析
    overall_analysis = generate_overall_analysis(results)
    
    # 返回结果和整体分析
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }




def generate_overall_analysis(results):
    """
    生成融合检索的整体分析。
    
    参数:
        results (List[Dict]): 评估查询的结果
        
    返回:
        str: 整体分析
    """
    # 指导评估过程的系统提示
    system_prompt = """You are an expert at evaluating information retrieval systems. 
    Based on multiple test queries, provide an overall analysis comparing three retrieval approaches:
    1. Vector-based retrieval (semantic similarity)
    2. BM25 keyword retrieval (keyword matching)
    3. Fusion retrieval (combination of both)

    Focus on:
    1. Types of queries where each approach performs best
    2. Overall strengths and weaknesses of each approach
    3. How fusion retrieval balances the trade-offs
    4. Recommendations for when to use each approach"""

    # 为每个查询创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison Summary: {result['comparison'][:200]}...\n\n"

    # 包含评估摘要的用户提示
    user_prompt = f"""Based on the following evaluations of different retrieval methods across {len(results)} queries, 
    provide an overall analysis comparing these three approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis of vector-based, BM25, and fusion retrieval approaches,
    highlighting when and why fusion retrieval provides advantages over the individual methods."""

    # 使用meta-llama/Llama-3.2-3B-Instruct生成整体分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 返回生成的分析内容
    return response.choices[0].message.content


# ## 评估融合检索



# PDF文档路径
# 包含AI信息的PDF文档路径，用于知识检索测试
pdf_path = "data/AI_Information.pdf"

# 定义一个与AI相关的测试查询
test_queries = [
    "What are the main applications of transformer models in natural language processing?"  # AI特定查询
]

# 可选的参考答案
reference_answers = [
    "Transformer models have revolutionized natural language processing with applications including machine translation, text summarization, question answering, sentiment analysis, and text generation. They excel at capturing long-range dependencies in text and have become the foundation for models like BERT, GPT, and T5.",
]

# 设置参数
k = 5  # 要检索的文档数量
alpha = 0.5  # 向量分数的权重（0.5表示向量和BM25之间的权重相等）

# 运行评估
evaluation_results = evaluate_fusion_retrieval(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers,
    k=k,
    alpha=alpha
)

# 打印整体分析
print("\n\n=== 整体分析 ===\n")
print(evaluation_results["overall_analysis"])

