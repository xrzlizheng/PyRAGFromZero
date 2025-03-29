#!/usr/bin/env python
# coding: utf-8


"""
# 纠正式RAG（CRAG）实现

在本代码中，我实现了纠正式RAG - 一种高级方法，它能动态评估检索到的信息，并在必要时纠正检索过程，使用网络搜索作为备选方案。

CRAG相比传统RAG的改进：

- 在使用检索内容前先进行评估
- 根据相关性动态切换知识源
- 当本地知识不足时，通过网络搜索进行纠正
- 在适当情况下结合多个来源的信息
"""

"""
## 环境设置
我们首先导入必要的库。
"""

import os
import numpy as np
import json
import fitz  # PyMuPDF
from openai import OpenAI
import requests
from typing import List, Dict, Tuple, Any
import re
from urllib.parse import quote_plus
import time

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
## 文档处理函数
"""

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容。
    
    参数:
        pdf_path (str): PDF文件的路径
        
    返回:
        str: 提取的文本内容
    """
    print(f"Extracting text from {pdf_path}...")
    
    # 打开PDF文件
    pdf = fitz.open(pdf_path)
    text = ""
    
    # 遍历PDF中的每一页
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        # 从当前页提取文本并将其添加到text变量中
        text += page.get_text()
    
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    """
    将文本分割成重叠的块，以便高效检索和处理。
    
    此函数将大文本分割成更小、更易管理的块，
    并在连续块之间指定重叠部分。分块对RAG系统至关重要，
    因为它允许更精确地检索相关信息。
    
    参数:
        text (str): 要分块的输入文本
        chunk_size (int): 每个块的最大字符数
        overlap (int): 连续块之间重叠的字符数，以保持块边界之间的上下文
        
    返回:
        List[Dict]: 文本块列表，每个包含:
                   - text: 块内容
                   - metadata: 带有位置信息和源类型的字典
    """
    chunks = []
    
    # 使用滑动窗口方法遍历文本
    # 每次移动(chunk_size - overlap)确保块之间适当重叠
    for i in range(0, len(text), chunk_size - overlap):
        # 提取当前块，受chunk_size限制
        chunk_text = text[i:i + chunk_size]
        
        # 只添加非空块
        if chunk_text:
            chunks.append({
                "text": chunk_text,  # 实际文本内容
                "metadata": {
                    "start_pos": i,  # 原始文本中的起始位置
                    "end_pos": i + len(chunk_text),  # 结束位置
                    "source_type": "document"  # 表示此文本的来源
                }
            })
    
    print(f"Created {len(chunks)} text chunks")
    return chunks

"""
## 简单向量存储实现
"""

class SimpleVectorStore:
    """
    使用NumPy的简单向量存储实现。
    """
    def __init__(self):
        # 初始化列表以存储向量、文本和元数据
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储添加项目。
        
        参数:
            text (str): 文本内容
            embedding (List[float]): 嵌入向量
            metadata (Dict, optional): 附加元数据
        """
        # 将嵌入、文本和元数据分别添加到它们的列表中
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def add_items(self, items, embeddings):
        """
        向向量存储添加多个项目。
        
        参数:
            items (List[Dict]): 带有文本和元数据的项目列表
            embeddings (List[List[float]]): 嵌入向量列表
        """
        # 遍历项目和嵌入，并将它们添加到存储中
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )
    
    def similarity_search(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目。
        
        参数:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 返回结果的数量
            
        返回:
            List[Dict]: 前k个最相似的项目
        """
        # 如果存储中没有向量，则返回空列表
        if not self.vectors:
            return []
        
        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)
        
        # 使用余弦相似度计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        
        return results

"""
## 创建嵌入
"""

def create_embeddings(texts, model="text-embedding-3-small"):
    """
    使用OpenAI的嵌入模型为文本输入创建向量嵌入。
    
    嵌入是文本的密集向量表示，它捕获语义含义，
    允许进行相似性比较。在RAG系统中，嵌入对于
    将查询与相关文档块匹配至关重要。
    
    参数:
        texts (str or List[str]): 要嵌入的输入文本。可以是单个字符串
                                  或字符串列表。
        model (str): 要使用的嵌入模型名称。默认为"text-embedding-3-small"。
        
    返回:
        List[List[float]]: 如果输入是列表，则返回嵌入向量列表。
                          如果输入是单个字符串，则返回单个嵌入向量。
    """
    # 通过将单个字符串转换为列表来处理单个字符串和列表输入
    input_texts = texts if isinstance(texts, list) else [texts]
    
    # 分批处理以避免API速率限制和有效负载大小限制
    # OpenAI API通常对请求大小和速率有限制
    batch_size = 100
    all_embeddings = []
    
    # 处理每批文本
    for i in range(0, len(input_texts), batch_size):
        # 提取当前批次的文本
        batch = input_texts[i:i + batch_size]
        
        # 调用API为当前批次生成嵌入
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        # 从响应中提取嵌入向量
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    # 如果原始输入是单个字符串，则只返回第一个嵌入
    if isinstance(texts, str):
        return all_embeddings[0]
    
    # 否则返回完整的嵌入列表
    return all_embeddings

"""
## 文档处理流程
"""

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    将文档处理到向量存储中。
    
    参数:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的字符大小
        chunk_overlap (int): 块之间重叠的字符数
        
    返回:
        SimpleVectorStore: 包含文档块的向量存储
    """
    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)
    
    # 将提取的文本分割成指定大小和重叠的块
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # 为每个文本块创建嵌入
    print("Creating embeddings for chunks...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = create_embeddings(chunk_texts)
    
    # 初始化新的向量存储
    vector_store = SimpleVectorStore()
    
    # 将块及其嵌入添加到向量存储中
    vector_store.add_items(chunks, chunk_embeddings)
    
    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store

"""
## 相关性评估函数
"""

def evaluate_document_relevance(query, document):
    """
    评估文档对查询的相关性。
    
    参数:
        query (str): 用户查询
        document (str): 文档文本
        
    返回:
        float: 相关性分数(0-1)
    """
    # 定义系统提示，指导模型如何评估相关性
    system_prompt = """
    You are an expert at evaluating document relevance. 
    Rate how relevant the given document is to the query on a scale from 0 to 1.
    0 means completely irrelevant, 1 means perfectly relevant.
    Provide ONLY the score as a float between 0 and 1.
    """
    
    # 定义包含查询和文档的用户提示
    user_prompt = f"Query: {query}\n\nDocument: {document}"
    
    try:
        # 向OpenAI API发出请求以评估相关性
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 指定要使用的模型
            messages=[
                {"role": "system", "content": system_prompt},  # 引导助手的系统消息
                {"role": "user", "content": user_prompt}  # 包含查询和文档的用户消息
            ],
            temperature=0,  # 设置响应生成的温度
            max_tokens=5  # 需要非常短的响应
        )
        
        # 从响应中提取分数
        score_text = response.choices[0].message.content.strip()
        # 使用正则表达式在响应中查找浮点值
        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
        if score_match:
            return float(score_match.group(1))  # 将提取的分数作为浮点数返回
        return 0.5  # 如果解析失败，默认为中间值
    
    except Exception as e:
        # 打印错误消息并在错误时返回默认值
        print(f"Error evaluating document relevance: {e}")
        return 0.5  # 错误时默认为中间值

"""
## 网络搜索函数
"""

def duck_duck_go_search(query, num_results=3):
    """
    使用DuckDuckGo执行网络搜索。
    
    参数:
        query (str): 搜索查询
        num_results (int): 返回结果的数量
        
    返回:
        Tuple[str, List[Dict]]: 组合的搜索结果文本和源元数据
    """
    # 为URL编码查询
    encoded_query = quote_plus(query)
    
    # DuckDuckGo搜索API端点（非官方）
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
    
    try:
        # 执行网络搜索请求
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        data = response.json()
        
        # 初始化变量以存储结果文本和来源
        results_text = ""
        sources = []
        
        # 如果有摘要，则添加摘要
        if data.get("AbstractText"):
            results_text += f"{data['AbstractText']}\n\n"
            sources.append({
                "title": data.get("AbstractSource", "Wikipedia"),
                "url": data.get("AbstractURL", "")
            })
        
        # 添加相关主题
        for topic in data.get("RelatedTopics", [])[:num_results]:
            if "Text" in topic and "FirstURL" in topic:
                results_text += f"{topic['Text']}\n\n"
                sources.append({
                    "title": topic.get("Text", "").split(" - ")[0],
                    "url": topic.get("FirstURL", "")
                })
        
        return results_text, sources
    
    except Exception as e:
        # 如果主搜索失败，则打印错误消息
        print(f"Error performing web search: {e}")
        
        # 回退到备用搜索API
        try:
            backup_url = f"https://serpapi.com/search.json?q={encoded_query}&engine=duckduckgo"
            response = requests.get(backup_url)
            data = response.json()
            
            # 初始化变量以存储结果文本和来源
            results_text = ""
            sources = []
            
            # 从备用API提取结果
            for result in data.get("organic_results", [])[:num_results]:
                results_text += f"{result.get('title', '')}: {result.get('snippet', '')}\n\n"
                sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", "")
                })
            
            return results_text, sources
        except Exception as backup_error:
            # 如果备用搜索也失败，则打印错误消息
            print(f"Backup search also failed: {backup_error}")
            return "Failed to retrieve search results.", []


def rewrite_search_query(query):
    """
    重写查询，使其更适合网络搜索。
    
    参数:
        query (str): 原始查询
        
    返回:
        str: 重写的查询
    """
    # 定义系统提示，指导模型如何重写查询
    system_prompt = """
    You are an expert at creating effective search queries.
    Rewrite the given query to make it more suitable for a web search engine.
    Focus on keywords and facts, remove unnecessary words, and make it concise.
    """
    
    try:
        # 向OpenAI API发出请求以重写查询
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 指定要使用的模型
            messages=[
                {"role": "system", "content": system_prompt},  # 引导助手的系统消息
                {"role": "user", "content": f"Original query: {query}\n\nRewritten query:"}  # 包含原始查询的用户消息
            ],
            temperature=0.3,  # 设置响应生成的温度
            max_tokens=50  # 限制响应长度
        )
        
        # 从响应中返回重写的查询
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 打印错误消息并在错误时返回原始查询
        print(f"Error rewriting search query: {e}")
        return query  # 错误时返回原始查询


def perform_web_search(query):
    """
    使用查询重写执行网络搜索。
    
    参数:
        query (str): 原始用户查询
        
    返回:
        Tuple[str, List[Dict]]: 搜索结果文本和源元数据
    """
    # 重写查询以改善搜索结果
    rewritten_query = rewrite_search_query(query)
    print(f"Rewritten search query: {rewritten_query}")
    
    # 使用重写的查询执行网络搜索
    results_text, sources = duck_duck_go_search(rewritten_query)
    
    # 返回搜索结果文本和源元数据
    return results_text, sources

"""
## 知识精炼函数
"""

def refine_knowledge(text):
    """
    从文本中提取和精炼关键信息。
    
    参数:
        text (str): 要精炼的输入文本
        
    返回:
        str: 从文本中精炼的关键点
    """
    # 定义系统提示，指导模型如何提取关键信息
    system_prompt = """
    Extract the key information from the following text as a set of clear, concise bullet points.
    Focus on the most relevant facts and important details.
    Format your response as a bulleted list with each point on a new line starting with "• ".
    """
    
    try:
        # 向OpenAI API发出请求以精炼文本
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 指定要使用的模型
            messages=[
                {"role": "system", "content": system_prompt},  # 引导助手的系统消息
                {"role": "user", "content": f"Text to refine:\n\n{text}"}  # 包含要精炼的文本的用户消息
            ],
            temperature=0.3  # 设置响应生成的温度
        )
        
        # 从响应中返回精炼的关键点
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 打印错误消息并在错误时返回原始文本
        print(f"Error refining knowledge: {e}")
        return text  # 错误时返回原始文本

"""
## 核心CRAG流程
"""

def crag_process(query, vector_store, k=3):
    """
    运行ncorrectiveRAG流程。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        k (int): 要检索的初始文档数量
        
    返回:
        Dict: 包括响应和调试信息的流程结果
    """
    print(f"\n=== Processing query with CRAG: {query} ===\n")
    
    # 步骤1: 创建查询嵌入并检索文档
    print("Retrieving initial documents...")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=k)
    
    # 步骤2: 评估文档相关性
    print("Evaluating document relevance...")
    relevance_scores = []
    for doc in retrieved_docs:
        score = evaluate_document_relevance(query, doc["text"])
        relevance_scores.append(score)
        doc["relevance"] = score
        print(f"Document scored {score:.2f} relevance")
    
    # 步骤3: 根据最佳相关性分数确定行动
    max_score = max(relevance_scores) if relevance_scores else 0
    best_doc_idx = relevance_scores.index(max_score) if relevance_scores else -1
    
    # 跟踪来源以供归因
    sources = []
    final_knowledge = ""
    
    # 步骤4: 执行适当的知识获取策略
    if max_score > 0.7:
        # 情况1: 高相关性 - 直接使用文档
        print(f"High relevance ({max_score:.2f}) - Using document directly")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        final_knowledge = best_doc
        sources.append({
            "title": "Document",
            "url": ""
        })
        
    elif max_score < 0.3:
        # 情况2: 低相关性 - 使用网络搜索
        print(f"Low relevance ({max_score:.2f}) - Performing web search")
        web_results, web_sources = perform_web_search(query)
        final_knowledge = refine_knowledge(web_results)
        sources.extend(web_sources)
        
    else:
        # 情况3: 中等相关性 - 结合文档和网络搜索
        print(f"Medium relevance ({max_score:.2f}) - Combining document with web search")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        refined_doc = refine_knowledge(best_doc)
        
        # 获取网络结果
        web_results, web_sources = perform_web_search(query)
        refined_web = refine_knowledge(web_results)
        
        # 组合知识
        final_knowledge = f"From document:\n{refined_doc}\n\nFrom web search:\n{refined_web}"
        
        # 添加来源
        sources.append({
            "title": "Document",
            "url": ""
        })
        sources.extend(web_sources)
    
    # 步骤5: 生成最终响应
    print("Generating final response...")
    response = generate_response(query, final_knowledge, sources)
    
    # 返回综合结果
    return {
        "query": query,
        "response": response,
        "retrieved_docs": retrieved_docs,
        "relevance_scores": relevance_scores,
        "max_relevance": max_score,
        "final_knowledge": final_knowledge,
        "sources": sources
    }

"""
## 响应生成
"""

def generate_response(query, knowledge, sources):
    """
    基于查询和知识生成响应。
    
    参数:
        query (str): 用户查询
        knowledge (str): 作为响应基础的知识
        sources (List[Dict]): 带有标题和URL的来源列表
        
    返回:
        str: 生成的响应
    """
    # 格式化来源以包含在提示中
    sources_text = ""
    for source in sources:
        title = source.get("title", "Unknown Source")
        url = source.get("url", "")
        if url:
            sources_text += f"- {title}: {url}\n"
        else:
            sources_text += f"- {title}\n"
    
    # 定义系统提示，指导模型如何生成响应
    system_prompt = """
    You are a helpful AI assistant. Generate a comprehensive, informative response to the query based on the provided knowledge.
    Include all relevant information while keeping your answer clear and concise.
    If the knowledge doesn't fully answer the query, acknowledge this limitation.
    Include source attribution at the end of your response.
    """
    
    # 定义包含查询、信息和来源的用户提示
    user_prompt = f"""
    Query: {query}
    
    Knowledge:
    {knowledge}
    
    Sources:
    {sources_text}
    
    Please provide an informative response to the query based on this information.
    Include the sources at the end of your response.
    """
    
    try:
        # 向OpenAI API发出请求以生成响应
        response = client.chat.completions.create(
            model="gpt-4",  # 使用GPT-4获得高质量响应
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        # 返回生成的响应
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 打印错误消息并返回错误响应
        print(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error while generating a response to your query: '{query}'. The error was: {str(e)}"

"""
## 评估函数
"""

def evaluate_crag_response(query, response, reference_answer=None):
    """
    评估CRAG响应的质量。
    
    参数:
        query (str): 用户查询
        response (str): 生成的响应
        reference_answer (str, optional): 用于比较的参考答案
        
    返回:
        Dict: 评估指标
    """
    # 评估标准的系统提示
    system_prompt = """
    You are an expert at evaluating the quality of responses to questions.
    Please evaluate the provided response based on the following criteria:
    
    1. Relevance (0-10): How directly does the response address the query?
    2. Accuracy (0-10): How factually correct is the information?
    3. Completeness (0-10): How thoroughly does the response answer all aspects of the query?
    4. Clarity (0-10): How clear and easy to understand is the response?
    5. Source Quality (0-10): How well does the response cite relevant sources?
    
    Return your evaluation as a JSON object with scores for each criterion and a brief explanation for each score.
    Also include an "overall_score" (0-10) and a brief "summary" of your evaluation.
    """
    
    # 包含要评估的查询和响应的用户提示
    user_prompt = f"""
    Query: {query}
    
    Response to evaluate:
    {response}
    """
    
    # 如果提供了参考答案，则将其包含在提示中
    if reference_answer:
        user_prompt += f"""
    Reference answer (for comparison):
    {reference_answer}
    """
    
    try:
        # 从GPT-4模型请求评估
        evaluation_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        # 解析评估响应
        evaluation = json.loads(evaluation_response.choices[0].message.content)
        return evaluation
    except Exception as e:
        # 处理评估过程中的任何错误
        print(f"Error evaluating response: {e}")
        return {
            "error": str(e),
            "overall_score": 0,
            "summary": "Evaluation failed due to an error."
        }


def compare_crag_vs_standard_rag(query, vector_store, reference_answer=None):
    """
    比较CRAG和标准RAG对查询的处理。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        reference_answer (str, optional): 用于比较的参考答案
        
    返回:
        Dict: 比较结果
    """
    # 运行CRAG流程
    print("\n=== Running CRAG ===")
    crag_result = crag_process(query, vector_store)
    crag_response = crag_result["response"]
    
    # 运行标准RAG（直接检索和响应）
    print("\n=== Running standard RAG ===")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=3)
    combined_text = "\n\n".join([doc["text"] for doc in retrieved_docs])
    standard_sources = [{"title": "Document", "url": ""}]
    standard_response = generate_response(query, combined_text, standard_sources)
    
    # 评估两种方法
    print("\n=== Evaluating CRAG response ===")
    crag_eval = evaluate_crag_response(query, crag_response, reference_answer)
    
    print("\n=== Evaluating standard RAG response ===")
    standard_eval = evaluate_crag_response(query, standard_response, reference_answer)
    
    # 比较方法
    print("\n=== Comparing approaches ===")
    comparison = compare_responses(query, crag_response, standard_response, reference_answer)
    
    return {
        "query": query,
        "crag_response": crag_response,
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "crag_evaluation": crag_eval,
        "standard_evaluation": standard_eval,
        "comparison": comparison
    }




def compare_responses(query, crag_response, standard_response, reference_answer=None):
    """
    比较CRAG和标准RAG响应。
    
    参数:
        query (str): 用户查询
        crag_response (str): CRAG响应
        standard_response (str): 标准RAG响应
        reference_answer (str, optional): 参考答案
        
    返回:
        str: 比较分析结果
    """
    # 用于比较两种方法的系统提示
    system_prompt = """
    You are an expert evaluator comparing two response generation approaches:
    
    1. CRAG (Corrective RAG): A system that evaluates document relevance and dynamically switches to web search when needed.
    2. Standard RAG: A system that directly retrieves documents based on embedding similarity and uses them for response generation.
    
    Compare the responses from these two systems based on:
    - Accuracy and factual correctness
    - Relevance to the query
    - Completeness of the answer
    - Clarity and organization
    - Source attribution quality
    
    Explain which approach performed better for this specific query and why.
    """
    
    # 包含查询和待比较响应的用户提示
    user_prompt = f"""
    Query: {query}
    
    CRAG Response:
    {crag_response}
    
    Standard RAG Response:
    {standard_response}
    """
    
    # 如果提供了参考答案，则将其包含在提示中
    if reference_answer:
        user_prompt += f"""
    Reference Answer:
    {reference_answer}
    """
    
    try:
        # 从GPT-4模型请求比较分析
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        # 返回比较分析结果
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 处理比较过程中的任何错误
        print(f"Error comparing responses: {e}")
        return f"Error comparing responses: {str(e)}"


"""
## 完整评估流程
"""

def run_crag_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    使用多个测试查询运行CRAG的完整评估。
    
    参数:
        pdf_path (str): PDF文档的路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 查询的参考答案
        
    返回:
        Dict: 完整的评估结果
    """
    # 处理文档并创建向量存储
    vector_store = process_document(pdf_path)
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n\n===== Evaluating Query {i+1}/{len(test_queries)} =====")
        print(f"Query: {query}")
        
        # 如果可用，获取参考答案
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        
        # 运行CRAG和标准RAG之间的比较
        result = compare_crag_vs_standard_rag(query, vector_store, reference)
        results.append(result)
        
        # 显示比较结果
        print("\n=== Comparison ===")
        print(result["comparison"])
    
    # 从个别结果生成总体分析
    overall_analysis = generate_overall_analysis(results)
    
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results):
    """
    生成评估结果的总体分析。
    
    参数:
        results (List[Dict]): 来自个别查询评估的结果
        
    返回:
        str: 总体分析
    """
    # 分析的系统提示
    system_prompt = """
    You are an expert at evaluating information retrieval and response generation systems.
    Based on multiple test queries, provide an overall analysis comparing CRAG (Corrective RAG) 
    with standard RAG.
    
    Focus on:
    1. When CRAG performs better and why
    2. When standard RAG performs better and why
    3. The overall strengths and weaknesses of each approach
    4. Recommendations for when to use each approach
    """
    
    # 创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        if 'crag_evaluation' in result and 'overall_score' in result['crag_evaluation']:
            crag_score = result['crag_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"CRAG score: {crag_score}\n"
        if 'standard_evaluation' in result and 'overall_score' in result['standard_evaluation']:
            std_score = result['standard_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"Standard RAG score: {std_score}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"
    
    # 分析的用户提示
    user_prompt = f"""
    Based on the following evaluations comparing CRAG vs standard RAG across {len(results)} queries, 
    provide an overall analysis of these two approaches:
    
    {evaluations_summary}
    
    Please provide a comprehensive analysis of the relative strengths and weaknesses of CRAG 
    compared to standard RAG, focusing on when and why one approach outperforms the other.
    """
    
    try:
        # 使用GPT-4生成总体分析
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating overall analysis: {e}")
        return f"Error generating overall analysis: {str(e)}"


"""
## 使用测试查询评估CRAG
"""

# AI信息PDF文档的路径
pdf_path = "data/AI_Information.pdf"

# 使用多个AI相关查询运行综合评估
test_queries = [
    "How does machine learning differ from traditional programming?",
]

# 可选的参考答案，用于更好的质量评估
reference_answers = [
    "Machine learning differs from traditional programming by having computers learn patterns from data rather than following explicit instructions. In traditional programming, developers write specific rules for the computer to follow, while in machine learning",
]

# 运行完整评估，比较CRAG与标准RAG
evaluation_results = run_crag_evaluation(pdf_path, test_queries, reference_answers)
print("\n=== Overall Analysis of CRAG vs Standard RAG ===")
print(evaluation_results["overall_analysis"])

