#!/usr/bin/env python
# coding: utf-8

"""
# 自适应检索增强型RAG系统

在本代码中，我实现了一个自适应检索系统，该系统能够根据查询类型动态选择最合适的检索策略。
这种方法显著提高了我们RAG系统在各种问题类型上提供准确和相关回答的能力。

不同的问题需要不同的检索策略。我们的系统：

1. 对查询类型进行分类（事实型、分析型、观点型或上下文型）
2. 选择合适的检索策略
3. 执行专门的检索技术
4. 生成定制化的回答
"""

"""
## 环境设置
首先导入必要的库。
"""

import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re

"""
## 从PDF文件中提取文本
为了实现RAG，我们首先需要一个文本数据源。在这种情况下，我们使用PyMuPDF库从PDF文件中提取文本。
"""

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并打印前`num_chars`个字符。

    参数:
    pdf_path (str): PDF文件的路径。

    返回:
    str: 从PDF中提取的文本。
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串来存储提取的文本

    # 遍历PDF中的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取页面
        text = page.get_text("text")  # 从页面中提取文本
        all_text += text  # 将提取的文本添加到all_text字符串中

    return all_text  # 返回提取的文本

"""
## 对提取的文本进行分块
一旦我们提取了文本，我们将其分成更小的、重叠的块，以提高检索准确性。
"""

def chunk_text(text, n, overlap):
    """
    将给定文本分成具有重叠部分的n个字符的段落。

    参数:
    text (str): 要分块的文本。
    n (int): 每个块中的字符数。
    overlap (int): 块之间重叠的字符数。

    返回:
    List[str]: 文本块列表。
    """
    chunks = []  # 初始化一个空列表来存储块
    
    # 以(n - overlap)的步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 将从索引i到i + n的文本块添加到chunks列表中
        chunks.append(text[i:i + n])

    return chunks  # 返回文本块列表

"""
## 设置OpenAI API客户端
我们初始化OpenAI客户端以生成嵌入和响应。
"""

# 使用基础URL和API密钥初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)

"""
## 简单向量存储实现
我们将创建一个基本的向量存储来管理文档块及其嵌入。
"""

class SimpleVectorStore:
    """
    使用NumPy的简单向量存储实现。
    """
    def __init__(self):
        """
        初始化向量存储。
        """
        self.vectors = []  # 用于存储嵌入向量的列表
        self.texts = []  # 用于存储原始文本的列表
        self.metadata = []  # 用于存储每个文本的元数据的列表
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加项目。

        参数:
        text (str): 原始文本。
        embedding (List[float]): 嵌入向量。
        metadata (dict, optional): 额外的元数据。
        """
        self.vectors.append(np.array(embedding))  # 将嵌入转换为numpy数组并添加到向量列表中
        self.texts.append(text)  # 将原始文本添加到文本列表中
        self.metadata.append(metadata or {})  # 将元数据添加到元数据列表中，如果为None则默认为空字典
    
    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        查找与查询嵌入最相似的项目。

        参数:
        query_embedding (List[float]): 查询嵌入向量。
        k (int): 返回结果的数量。
        filter_func (callable, optional): 用于过滤结果的函数。

        返回:
        List[Dict]: 前k个最相似项目及其文本和元数据。
        """
        if not self.vectors:
            return []  # 如果没有存储向量，则返回空列表
        
        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)
        
        # 使用余弦相似度计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 如果提供了过滤器，则应用过滤器
            if filter_func and not filter_func(self.metadata[i]):
                continue
                
            # 计算余弦相似度
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 添加索引和相似度分数
        
        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 添加文本
                "metadata": self.metadata[idx],  # 添加元数据
                "similarity": score  # 添加相似度分数
            })
        
        return results  # 返回前k个结果列表

"""
## 创建嵌入
"""

def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    为给定文本创建嵌入。

    参数:
    text (str or List[str]): 要为其创建嵌入的输入文本。
    model (str): 用于创建嵌入的模型。

    返回:
    List[float] or List[List[float]]: 嵌入向量。
    """
    # 通过将字符串输入转换为列表来处理字符串和列表输入
    input_text = text if isinstance(text, list) else [text]
    
    # 使用指定模型为输入文本创建嵌入
    response = client.embeddings.create(
        model=model,
        input=input_text
    )
    
    # 如果输入是单个字符串，则仅返回第一个嵌入
    if isinstance(text, str):
        return response.data[0].embedding
    
    # 否则，返回文本列表的所有嵌入
    return [item.embedding for item in response.data]

"""
## 文档处理流程
"""

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    处理文档以用于自适应检索。

    参数:
    pdf_path (str): PDF文件的路径。
    chunk_size (int): 每个块的字符大小。
    chunk_overlap (int): 块之间重叠的字符数。

    返回:
    Tuple[List[str], SimpleVectorStore]: 文档块和向量存储。
    """
    # 从PDF文件中提取文本
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # 对提取的文本进行分块
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")
    
    # 为文本块创建嵌入
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)
    
    # 初始化向量存储
    store = SimpleVectorStore()
    
    # 将每个块及其嵌入添加到向量存储中，并带有元数据
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    
    print(f"Added {len(chunks)} chunks to the vector store")
    
    # 返回块和向量存储
    return chunks, store

"""
## 查询分类
"""

def classify_query(query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    将查询分类为四类之一：事实型、分析型、观点型或上下文型。
    
    参数:
        query (str): 用户查询
        model (str): 要使用的LLM模型
        
    返回:
        str: 查询类别
    """
    # 定义系统提示以指导AI的分类
    system_prompt = """You are an expert at classifying questions. 
        Classify the given query into exactly one of these categories:
        - Factual: Queries seeking specific, verifiable information.
        - Analytical: Queries requiring comprehensive analysis or explanation.
        - Opinion: Queries about subjective matters or seeking diverse viewpoints.
        - Contextual: Queries that depend on user-specific context.

        Return ONLY the category name, without any explanation or additional text.
    """

    # 创建包含要分类的查询的用户提示
    user_prompt = f"Classify this query: {query}"
    
    # 从AI模型生成分类响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从响应中提取并去除类别
    category = response.choices[0].message.content.strip()
    
    # 定义有效类别列表
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
    
    # 确保返回的类别有效
    for valid in valid_categories:
        if valid in category:
            return valid
    
    # 如果分类失败，默认为"Factual"
    return "Factual"

"""
## 实现专门的检索策略
### 1. 事实型策略 - 注重精确性
"""

def factual_retrieval_strategy(query, vector_store, k=4):
    """
    事实型查询的检索策略，注重精确性。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 返回文档的数量
        
    返回:
        List[Dict]: 检索到的文档
    """
    print(f"Executing Factual retrieval strategy for: '{query}'")
    
    # 使用LLM增强查询以提高精确性
    system_prompt = """You are an expert at enhancing search queries.
        Your task is to reformulate the given factual query to make it more precise and 
        specific for information retrieval. Focus on key entities and their relationships.

        Provide ONLY the enhanced query without any explanation.
    """

    user_prompt = f"Enhance this factual query: {query}"
    
    # 使用LLM生成增强查询
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 提取并打印增强查询
    enhanced_query = response.choices[0].message.content.strip()
    print(f"Enhanced query: {enhanced_query}")
    
    # 为增强查询创建嵌入
    query_embedding = create_embeddings(enhanced_query)
    
    # 执行初始相似度搜索以检索文档
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)
    
    # 初始化列表以存储排名结果
    ranked_results = []
    
    # 使用LLM对文档进行相关性评分和排名
    for doc in initial_results:
        relevance_score = score_document_relevance(enhanced_query, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })
    
    # 按相关性分数降序排序结果
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # 返回前k个结果
    return ranked_results[:k]

"""
### 2. 分析型策略 - 全面覆盖
"""

def analytical_retrieval_strategy(query, vector_store, k=4):
    """
    分析型查询的检索策略，注重全面覆盖。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 返回文档的数量
        
    返回:
        List[Dict]: 检索到的文档
    """
    print(f"Executing Analytical retrieval strategy for: '{query}'")
    
    # 定义系统提示以指导AI生成子问题
    system_prompt = """You are an expert at breaking down complex questions.
    Generate sub-questions that explore different aspects of the main analytical query.
    These sub-questions should cover the breadth of the topic and help retrieve 
    comprehensive information.

    Return a list of exactly 3 sub-questions, one per line.
    """

    # 创建包含主查询的用户提示
    user_prompt = f"Generate sub-questions for this analytical query: {query}"
    
    # 使用LLM生成子问题
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    # 提取并清理子问题
    sub_queries = response.choices[0].message.content.strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    print(f"Generated sub-queries: {sub_queries}")
    
    # 为每个子查询检索文档
    all_results = []
    for sub_query in sub_queries:
        # 为子查询创建嵌入
        sub_query_embedding = create_embeddings(sub_query)
        # 为子查询执行相似度搜索
        results = vector_store.similarity_search(sub_query_embedding, k=2)
        all_results.extend(results)
    
    # 通过从不同子查询结果中选择来确保多样性
    # 删除重复项（相同的文本内容）
    unique_texts = set()
    diverse_results = []
    
    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)
    
    # 如果我们需要更多结果以达到k，从初始结果中添加更多
    if len(diverse_results) < k:
        # 对主查询进行直接检索
        main_query_embedding = create_embeddings(query)
        main_results = vector_store.similarity_search(main_query_embedding, k=k)
        
        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)
    
    # 返回前k个多样化结果
    return diverse_results[:k]

"""
### 3. 观点型策略 - 多样化视角
"""

def opinion_retrieval_strategy(query, vector_store, k=4):
    """
    观点型查询的检索策略，注重多样化视角。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 返回文档的数量
        
    返回:
        List[Dict]: 检索到的文档
    """
    print(f"Executing Opinion retrieval strategy for: '{query}'")
    
    # 定义系统提示以指导AI识别不同视角
    system_prompt = """You are an expert at identifying different perspectives on a topic.
        For the given query about opinions or viewpoints, identify different perspectives 
        that people might have on this topic.

        Return a list of exactly 3 different viewpoint angles, one per line.
    """

    # 创建包含主查询的用户提示
    user_prompt = f"Identify different perspectives on: {query}"
    
    # 使用LLM生成不同视角
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    # 提取并清理视角
    viewpoints = response.choices[0].message.content.strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    print(f"Identified viewpoints: {viewpoints}")
    
    # 检索代表每个视角的文档
    all_results = []
    for viewpoint in viewpoints:
        # 将主查询与视角结合
        combined_query = f"{query} {viewpoint}"
        # 为组合查询创建嵌入
        viewpoint_embedding = create_embeddings(combined_query)
        # 为组合查询执行相似度搜索
        results = vector_store.similarity_search(viewpoint_embedding, k=2)
        
        # 用它们代表的视角标记结果
        for result in results:
            result["viewpoint"] = viewpoint
        
        # 将结果添加到所有结果列表中
        all_results.extend(results)
    
    # 选择多样化的观点范围
    # 确保我们尽可能从每个视角获取至少一个文档
    selected_results = []
    for viewpoint in viewpoints:
        # 按视角过滤文档
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])
    
    # 用相似度最高的文档填充剩余槽位
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        # 按相似度排序剩余文档
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])
    
    # 返回前k个结果
    return selected_results[:k]

"""
### 4. 上下文型策略 - 用户上下文集成
"""

def contextual_retrieval_strategy(query, vector_store, k=4, user_context=None):
    """
    上下文型查询的检索策略，集成用户上下文。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 返回文档的数量
        user_context (str): 额外的用户上下文
        
    返回:
        List[Dict]: 检索到的文档
    """
    print(f"Executing Contextual retrieval strategy for: '{query}'")
    
    # 如果没有提供用户上下文，尝试从查询中推断
    if not user_context:
        system_prompt = """You are an expert at understanding implied context in questions.
For the given query, infer what contextual information might be relevant or implied 
but not explicitly stated. Focus on what background would help answering this query.

Return a brief description of the implied context."""

        user_prompt = f"Infer the implied context in this query: {query}"
        
        # 使用LLM生成推断的上下文
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        # 提取并打印推断的上下文
        user_context = response.choices[0].message.content.strip()
        print(f"Inferred context: {user_context}")
    
    # 重新表述查询以纳入上下文
    system_prompt = """You are an expert at reformulating questions with context.
    Given a query and some contextual information, create a more specific query that 
    incorporates the context to get more relevant information.

    Return ONLY the reformulated query without explanation."""

    user_prompt = f"""
    Query: {query}
    Context: {user_context}

    Reformulate the query to incorporate this context:"""
    
    # 使用LLM生成上下文化查询
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 提取并打印上下文化查询
    contextualized_query = response.choices[0].message.content.strip()
    print(f"Contextualized query: {contextualized_query}")
    
    # 基于上下文化查询检索文档
    query_embedding = create_embeddings(contextualized_query)
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)
    
    # 考虑相关性和用户上下文对文档进行排名
    ranked_results = []
    
    for doc in initial_results:
        # 考虑上下文对文档相关性进行评分
        context_relevance = score_document_context_relevance(query, user_context, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })
    
    # 按上下文相关性排序并返回前k个结果
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]

"""
## 文档评分的辅助函数
"""

def score_document_relevance(query, document, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    使用LLM对文档与查询的相关性进行评分。
    
    参数:
        query (str): 用户查询
        document (str): 文档文本
        model (str): LLM模型
        
    返回:
        float: 0-10的相关性分数
    """
    # 系统提示，指导模型如何评价相关性
    system_prompt = """You are an expert at evaluating document relevance.
        Rate the relevance of a document to a query on a scale from 0 to 10, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query

        Return ONLY a numerical score between 0 and 10, nothing else.
    """

    # 如果文档太长，则截断
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    # 包含查询和文档预览的用户提示
    user_prompt = f"""
        Query: {query}

        Document: {doc_preview}

        Relevance score (0-10):
    """
    
    # 从模型生成响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从模型的响应中提取分数
    score_text = response.choices[0].message.content.strip()
    
    # 使用正则表达式提取数字分数
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # 确保分数在0-10范围内
    else:
        # 如果提取失败，则默认分数
        return 5.0

def score_document_context_relevance(query, context, document, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    考虑查询和上下文对文档相关性进行评分。
    
    参数:
        query (str): 用户查询
        context (str): 用户上下文
        document (str): 文档文本
        model (str): LLM模型
        
    返回:
        float: 0-10的相关性分数
    """
    # 系统提示，指导模型如何考虑上下文评价相关性
    system_prompt = """You are an expert at evaluating document relevance considering context.
        Rate the document on a scale from 0 to 10 based on how well it addresses the query
        when considering the provided context, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query in the given context

        Return ONLY a numerical score between 0 and 10, nothing else.
    """

    # 如果文档太长，则截断
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    # 包含查询、上下文和文档预览的用户提示
    user_prompt = f"""
    Query: {query}
    Context: {context}

    Document: {doc_preview}

    Relevance score considering context (0-10):
    """
    
    # 从模型生成响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从模型的响应中提取分数
    score_text = response.choices[0].message.content.strip()
    
    # 使用正则表达式提取数字分数
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # 确保分数在0-10范围内
    else:
        # 如果提取失败，则默认分数
        return 5.0

"""
## 核心自适应检索器
"""

def adaptive_retrieval(query, vector_store, k=4, user_context=None):
    """
    通过选择并执行适当的策略进行自适应检索。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要检索的文档数量
        user_context (str): 上下文型查询的可选用户上下文
        
    返回:
        List[Dict]: 检索到的文档
    """
    # 对查询进行分类以确定其类型
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # 根据查询类型选择并执行适当的检索策略
    if query_type == "Factual":
        # 使用事实型检索策略获取精确信息
        results = factual_retrieval_strategy(query, vector_store, k)
    elif query_type == "Analytical":
        # 使用分析型检索策略获取全面覆盖
        results = analytical_retrieval_strategy(query, vector_store, k)
    elif query_type == "Opinion":
        # 使用观点型检索策略获取多样化视角
        results = opinion_retrieval_strategy(query, vector_store, k)
    elif query_type == "Contextual":
        # 使用上下文型检索策略，整合用户上下文
        results = contextual_retrieval_strategy(query, vector_store, k, user_context)
    else:
        # 如果分类失败，默认使用事实型检索策略
        results = factual_retrieval_strategy(query, vector_store, k)
    
    return results  # 返回检索到的文档


"""
## 响应生成
"""

def generate_response(query, results, query_type, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于查询、检索到的文档和查询类型生成响应。
    
    参数:
        query (str): 用户查询
        results (List[Dict]): 检索到的文档
        query_type (str): 查询类型
        model (str): LLM模型
        
    返回:
        str: 生成的响应
    """
    # 通过将文档文本与分隔符连接来准备上下文
    context = "\n\n---\n\n".join([r["text"] for r in results])
    
    # 根据查询类型创建自定义系统提示
    if query_type == "Factual":
        system_prompt = """You are a helpful assistant providing factual information.
    Answer the question based on the provided context. Focus on accuracy and precision.
    If the context doesn't contain the information needed, acknowledge the limitations."""
        
    elif query_type == "Analytical":
        system_prompt = """You are a helpful assistant providing analytical insights.
    Based on the provided context, offer a comprehensive analysis of the topic.
    Cover different aspects and perspectives in your explanation.
    If the context has gaps, acknowledge them while providing the best analysis possible."""
        
    elif query_type == "Opinion":
        system_prompt = """You are a helpful assistant discussing topics with multiple viewpoints.
    Based on the provided context, present different perspectives on the topic.
    Ensure fair representation of diverse opinions without showing bias.
    Acknowledge where the context presents limited viewpoints."""
        
    elif query_type == "Contextual":
        system_prompt = """You are a helpful assistant providing contextually relevant information.
    Answer the question considering both the query and its context.
    Make connections between the query context and the information in the provided documents.
    If the context doesn't fully address the specific situation, acknowledge the limitations."""
        
    else:
        system_prompt = """You are a helpful assistant. Answer the question based on the provided context. If you cannot answer from the context, acknowledge the limitations."""
    
    # 通过组合上下文和查询创建用户提示
    user_prompt = f"""
    Context:
    {context}

    Question: {query}

    Please provide a helpful response based on the context.
    """
    
    # 使用OpenAI客户端生成响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    # 返回生成的响应内容
    return response.choices[0].message.content


"""
## 具有自适应检索的完整RAG流程
"""

def rag_with_adaptive_retrieval(pdf_path, query, k=4, user_context=None):
    """
    具有自适应检索的完整RAG流程。
    
    参数:
        pdf_path (str): PDF文档的路径
        query (str): 用户查询
        k (int): 要检索的文档数量
        user_context (str): 可选的用户上下文
        
    返回:
        Dict: 包括查询、检索到的文档、查询类型和响应的结果
    """
    print("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")
    print(f"Query: {query}")
    
    # 处理文档以提取文本、分块并创建嵌入
    chunks, vector_store = process_document(pdf_path)
    
    # 对查询进行分类以确定其类型
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # 使用基于查询类型的自适应检索策略检索文档
    retrieved_docs = adaptive_retrieval(query, vector_store, k, user_context)
    
    # 基于查询、检索到的文档和查询类型生成响应
    response = generate_response(query, retrieved_docs, query_type)
    
    # 将结果编译成字典
    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_documents": retrieved_docs,
        "response": response
    }
    
    print("\n=== RESPONSE ===")
    print(response)
    
    return result


"""
## 评估框架
"""

def evaluate_adaptive_vs_standard(pdf_path, test_queries, reference_answers=None):
    """
    在一组测试查询上比较自适应检索与标准检索。
    
    此函数处理文档，对每个测试查询运行标准和自适应检索方法，
    并比较它们的性能。如果提供了参考答案，它还会根据这些参考
    评估响应质量。
    
    参数:
        pdf_path (str): 作为知识源处理的PDF文档路径
        test_queries (List[str]): 用于评估两种检索方法的测试查询列表
        reference_answers (List[str], optional): 用于评估指标的参考答案
        
    返回:
        Dict: 包含单个查询结果和整体比较的评估结果
    """
    print("=== EVALUATING ADAPTIVE VS. STANDARD RETRIEVAL ===")
    
    # 处理文档以提取文本、创建块并构建向量存储
    chunks, vector_store = process_document(pdf_path)
    
    # 初始化用于存储比较结果的集合
    results = []
    
    # 使用两种检索方法处理每个测试查询
    for i, query in enumerate(test_queries):
        print(f"\n\nQuery {i+1}: {query}")
        
        # --- 标准检索方法 ---
        print("\n--- Standard Retrieval ---")
        # 为查询创建嵌入
        query_embedding = create_embeddings(query)
        # 使用简单的向量相似度检索文档
        standard_docs = vector_store.similarity_search(query_embedding, k=4)
        # 使用通用方法生成响应
        standard_response = generate_response(query, standard_docs, "General")
        
        # --- 自适应检索方法 ---
        print("\n--- Adaptive Retrieval ---")
        # 对查询进行分类以确定其类型（事实型、分析型、观点型、上下文型）
        query_type = classify_query(query)
        # 使用适合此查询类型的策略检索文档
        adaptive_docs = adaptive_retrieval(query, vector_store, k=4)
        # 生成针对查询类型定制的响应
        adaptive_response = generate_response(query, adaptive_docs, query_type)
        
        # 存储此查询的完整结果
        result = {
            "query": query,
            "query_type": query_type,
            "standard_retrieval": {
                "documents": standard_docs,
                "response": standard_response
            },
            "adaptive_retrieval": {
                "documents": adaptive_docs,
                "response": adaptive_response
            }
        }
        
        # 如果有此查询的参考答案，则添加
        if reference_answers and i < len(reference_answers):
            result["reference_answer"] = reference_answers[i]
            
        results.append(result)
        
        # 显示两种响应的预览以进行快速比较
        print("\n--- Responses ---")
        print(f"Standard: {standard_response[:200]}...")
        print(f"Adaptive: {adaptive_response[:200]}...")
    
    # 如果有参考答案，则计算比较指标
    if reference_answers:
        comparison = compare_responses(results)
        print("\n=== EVALUATION RESULTS ===")
        print(comparison)
    
    # 返回完整的评估结果
    return {
        "results": results,
        "comparison": comparison if reference_answers else "No reference answers provided for evaluation"
    }


def compare_responses(results):
    """
    将标准和自适应响应与参考答案进行比较。
    
    参数:
        results (List[Dict]): 包含两种类型响应的结果
        
    返回:
        str: 比较分析
    """
    # 定义系统提示以指导AI比较响应
    comparison_prompt = """You are an expert evaluator of information retrieval systems.
    Compare the standard retrieval and adaptive retrieval responses for each query.
    Consider factors like accuracy, relevance, comprehensiveness, and alignment with the reference answer.
    Provide a detailed analysis of the strengths and weaknesses of each approach."""
    
    # 用标题初始化比较文本
    comparison_text = "# Evaluation of Standard vs. Adaptive Retrieval\n\n"
    
    # 遍历每个结果以比较响应
    for i, result in enumerate(results):
        # 如果查询没有参考答案，则跳过
        if "reference_answer" not in result:
            continue
            
        # 将查询详情添加到比较文本中
        comparison_text += f"## Query {i+1}: {result['query']}\n"
        comparison_text += f"*Query Type: {result['query_type']}*\n\n"
        comparison_text += f"**Reference Answer:**\n{result['reference_answer']}\n\n"
        
        # 将标准检索响应添加到比较文本中
        comparison_text += f"**Standard Retrieval Response:**\n{result['standard_retrieval']['response']}\n\n"
        
        # 将自适应检索响应添加到比较文本中
        comparison_text += f"**Adaptive Retrieval Response:**\n{result['adaptive_retrieval']['response']}\n\n"
        
        # 创建用户提示以让AI比较响应
        user_prompt = f"""
        Reference Answer: {result['reference_answer']}
        
        Standard Retrieval Response: {result['standard_retrieval']['response']}
        
        Adaptive Retrieval Response: {result['adaptive_retrieval']['response']}
        
        Provide a detailed comparison of the two responses.
        """
        
        # 使用OpenAI客户端生成比较分析
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": comparison_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        # 将AI的比较分析添加到比较文本中
        comparison_text += f"**Comparison Analysis:**\n{response.choices[0].message.content}\n\n"
    
    return comparison_text  # 返回完整的比较分析


"""
## 评估自适应检索系统（自定义查询）

使用自适应RAG评估系统的最后一步是使用您的PDF文档和测试查询调用evaluate_adaptive_vs_standard()函数：
"""

# PDF文档路径
# 此PDF文件包含RAG系统将使用的信息
pdf_path = "data/AI_Information.pdf"

# 定义涵盖不同查询类型的测试查询，以展示
# 自适应检索如何处理各种查询意图
test_queries = [
    "What is Explainable AI (XAI)?",                                              # 事实型查询 - 寻求定义/特定信息
    # "How do AI ethics and governance frameworks address potential societal impacts?",  # 分析型查询 - 需要全面分析
    # "Is AI development moving too fast for proper regulation?",                   # 观点型查询 - 寻求多样化视角
    # "How might explainable AI help in healthcare decisions?",                     # 上下文型查询 - 受益于上下文感知
]

# 用于更全面评估的参考答案
# 这些可用于根据已知标准客观评估响应质量
reference_answers = [
    "Explainable AI (XAI) aims to make AI systems transparent and understandable by providing clear explanations of how decisions are made. This helps users trust and effectively manage AI technologies.",
    # "AI ethics and governance frameworks address potential societal impacts by establishing guidelines and principles to ensure AI systems are developed and used responsibly. These frameworks focus on fairness, accountability, transparency, and the protection of human rights to mitigate risks and promote beneficial output.5.",
    # "Opinions on whether AI development is moving too fast for proper regulation vary. Some argue that rapid advancements outpace regulatory efforts, leading to potential risks and ethical concerns. Others believe that innovation should continue at its current pace, with regulations evolving alongside to address emerging challenges.",
    # "Explainable AI can significantly aid healthcare decisions by providing transparent and understandable insights into AI-driven recommendations. This transparency helps healthcare professionals trust AI systems, make informed decisions, and improve patient output by understanding the rationale behind AI suggestions."
]

# 运行评估比较自适应与标准检索
# 这将使用两种方法处理每个查询并比较结果
evaluation_results = evaluate_adaptive_vs_standard(
    pdf_path=pdf_path,                  # 知识提取的源文档
    test_queries=test_queries,          # 要评估的测试查询列表
    reference_answers=reference_answers  # 用于比较的可选基准事实
)

# 结果将显示标准检索和自适应检索性能在不同查询类型上的详细比较，
# 突出显示自适应策略在哪些方面提供了改进的结果
print(evaluation_results["comparison"])