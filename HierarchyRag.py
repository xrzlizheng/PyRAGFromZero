#!/usr/bin/env python
# coding: utf-8

"""
# 层次化索引的RAG系统

在本笔记本中，我实现了一种用于RAG系统的层次化索引方法。该技术通过使用两层搜索方法来提高检索效率：首先通过摘要识别相关文档部分，然后从这些部分中检索具体细节。

传统的RAG方法将所有文本块视为同等重要，这可能导致：

- 当文本块太小时，上下文丢失
- 当文档集合很大时，检索结果不相关
- 在整个语料库中进行低效搜索

层次化检索通过以下方式解决这些问题：

- 为较大的文档部分创建简洁的摘要
- 首先搜索这些摘要以识别相关部分
- 然后仅从这些部分中检索详细信息
- 在保留具体细节的同时保持上下文
"""

# 设置环境
# 首先导入必要的库
import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re
import pickle

# 设置OpenAI API客户端
# 初始化OpenAI客户端以生成嵌入和响应
# 使用基础URL和API密钥初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)

# 文档处理函数
def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容，并按页面分隔。
    
    参数:
        pdf_path (str): PDF文件路径
        
    返回:
        List[Dict]: 包含文本内容和元数据的页面列表
    """
    print(f"Extracting text from {pdf_path}...")  # 打印正在处理的PDF路径
    pdf = fitz.open(pdf_path)  # 使用PyMuPDF打开PDF文件
    pages = []  # 初始化空列表以存储带有文本内容的页面
    
    # 遍历PDF中的每一页
    for page_num in range(len(pdf)):
        page = pdf[page_num]  # 获取当前页面
        text = page.get_text()  # 从当前页面提取文本
        
        # 跳过文本很少的页面（少于50个字符）
        if len(text.strip()) > 50:
            # 将页面文本和元数据添加到列表中
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,  # 源文件路径
                    "page": page_num + 1  # 页码（从1开始索引）
                }
            })
    
    print(f"Extracted {len(pages)} pages with content")  # 打印提取的页面数量
    return pages  # 返回包含文本内容和元数据的页面列表

def chunk_text(text, metadata, chunk_size=1000, overlap=200):
    """
    将文本分割成重叠的块，同时保留元数据。
    
    参数:
        text (str): 要分块的输入文本
        metadata (Dict): 要保留的元数据
        chunk_size (int): 每个块的字符大小
        overlap (int): 块之间的重叠字符数
        
    返回:
        List[Dict]: 带有元数据的文本块列表
    """
    chunks = []  # 初始化空列表以存储块
    
    # 使用指定的块大小和重叠遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # 提取文本块
        
        # 跳过非常小的块（少于50个字符）
        if chunk_text and len(chunk_text.strip()) > 50:
            # 创建元数据的副本并添加块特定信息
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),  # 块的索引
                "start_char": i,  # 块的起始字符索引
                "end_char": i + len(chunk_text),  # 块的结束字符索引
                "is_summary": False  # 表示这不是摘要的标志
            })
            
            # 将块及其元数据添加到列表中
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
    
    return chunks  # 返回带有元数据的块列表

# 简单向量存储实现
class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储。
    """
    def __init__(self):
        self.vectors = []  # 用于存储向量嵌入的列表
        self.texts = []  # 用于存储文本内容的列表
        self.metadata = []  # 用于存储元数据的列表
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加项目。
        
        参数:
            text (str): 文本内容
            embedding (List[float]): 向量嵌入
            metadata (Dict, optional): 附加元数据
        """
        self.vectors.append(np.array(embedding))  # 将嵌入作为numpy数组添加
        self.texts.append(text)  # 添加文本内容
        self.metadata.append(metadata or {})  # 添加元数据或空字典（如果为None）
    
    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        查找与查询嵌入最相似的项目。
        
        参数:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 返回结果的数量
            filter_func (callable, optional): 过滤结果的函数
            
        返回:
            List[Dict]: 前k个最相似的项目
        """
        if not self.vectors:
            return []  # 如果没有向量，则返回空列表
        
        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)
        
        # 使用余弦相似度计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 如果不通过过滤器则跳过
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
                "text": self.texts[idx],  # 添加文本内容
                "metadata": self.metadata[idx],  # 添加元数据
                "similarity": float(score)  # 添加相似度分数
            })
        
        return results  # 返回前k个结果列表

# 创建嵌入
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为给定文本创建嵌入。
    
    参数:
        texts (List[str]): 输入文本
        model (str): 嵌入模型名称
        
    返回:
        List[List[float]]: 嵌入向量
    """
    # 处理空输入
    if not texts:
        return []
        
    # 如果需要，分批处理（OpenAI API限制）
    batch_size = 100
    all_embeddings = []
    
    # 分批遍历输入文本
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # 获取当前批次的文本
        
        # 为当前批次创建嵌入
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        # 从响应中提取嵌入
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 将批次嵌入添加到列表中
    
    return all_embeddings  # 返回所有嵌入

# 摘要生成函数
def generate_page_summary(page_text):
    """
    生成页面的简洁摘要。
    
    参数:
        page_text (str): 页面的文本内容
        
    返回:
        str: 生成的摘要
    """
    # 定义系统提示以指导摘要模型
    system_prompt = """You are an expert summarization system.
    Create a detailed summary of the provided text. 
    Focus on capturing the main topics, key information, and important facts.
    Your summary should be comprehensive enough to understand what the page contains
    but more concise than the original."""

    # 如果输入文本超过最大令牌限制，则截断
    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(page_text) > max_tokens else page_text

    # 向OpenAI API发出请求以生成摘要
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定要使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息，指导助手
            {"role": "user", "content": f"Please summarize this text:\n\n{truncated_text}"}  # 用户消息，包含要摘要的文本
        ],
        temperature=0.3  # 设置响应生成的温度
    )
    
    # 返回生成的摘要内容
    return response.choices[0].message.content

# 层次化文档处理
def process_document_hierarchically(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    将文档处理成层次化索引。
    
    参数:
        pdf_path (str): PDF文件路径
        chunk_size (int): 每个详细块的大小
        chunk_overlap (int): 块之间的重叠
        
    返回:
        Tuple[SimpleVectorStore, SimpleVectorStore]: 摘要和详细向量存储
    """
    # 从PDF中提取页面
    pages = extract_text_from_pdf(pdf_path)
    
    # 为每个页面创建摘要
    print("Generating page summaries...")
    summaries = []
    for i, page in enumerate(pages):
        print(f"Summarizing page {i+1}/{len(pages)}...")
        summary_text = generate_page_summary(page["text"])
        
        # 创建摘要元数据
        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})
        
        # 将摘要文本和元数据添加到摘要列表中
        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })
    
    # 为每个页面创建详细块
    detailed_chunks = []
    for page in pages:
        # 对页面的文本进行分块
        page_chunks = chunk_text(
            page["text"], 
            page["metadata"], 
            chunk_size, 
            chunk_overlap
        )
        # 用当前页面的块扩展detailed_chunks列表
        detailed_chunks.extend(page_chunks)
    
    print(f"Created {len(detailed_chunks)} detailed chunks")
    
    # 为摘要创建嵌入
    print("Creating embeddings for summaries...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = create_embeddings(summary_texts)
    
    # 为详细块创建嵌入
    print("Creating embeddings for detailed chunks...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)
    
    # 创建向量存储
    summary_store = SimpleVectorStore()
    detailed_store = SimpleVectorStore()
    
    # 将摘要添加到摘要存储中
    for i, summary in enumerate(summaries):
        summary_store.add_item(
            text=summary["text"],
            embedding=summary_embeddings[i],
            metadata=summary["metadata"]
        )
    
    # 将块添加到详细存储中
    for i, chunk in enumerate(detailed_chunks):
        detailed_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )
    
    print(f"Created vector stores with {len(summaries)} summaries and {len(detailed_chunks)} chunks")
    return summary_store, detailed_store

# 层次化检索
def retrieve_hierarchically(query, summary_store, detailed_store, k_summaries=3, k_chunks=5):
    """
    使用层次化索引检索信息。
    
    参数:
        query (str): 用户查询
        summary_store (SimpleVectorStore): 文档摘要存储
        detailed_store (SimpleVectorStore): 详细块存储
        k_summaries (int): 要检索的摘要数量
        k_chunks (int): 每个摘要要检索的块数量
        
    返回:
        List[Dict]: 带有相关性分数的检索块
    """
    print(f"Performing hierarchical retrieval for query: {query}")
    
    # 创建查询嵌入
    query_embedding = create_embeddings(query)
    
    # 首先，检索相关摘要
    summary_results = summary_store.similarity_search(
        query_embedding, 
        k=k_summaries
    )
    
    print(f"Retrieved {len(summary_results)} relevant summaries")
    
    # 从相关摘要中收集页面
    relevant_pages = [result["metadata"]["page"] for result in summary_results]
    
    # 创建一个过滤函数，只保留来自相关页面的块
    def page_filter(metadata):
        return metadata["page"] in relevant_pages
    
    # 然后，仅从这些相关页面中检索详细块
    detailed_results = detailed_store.similarity_search(
        query_embedding, 
        k=k_chunks * len(relevant_pages),
        filter_func=page_filter
    )
    
    print(f"Retrieved {len(detailed_results)} detailed chunks from relevant pages")
    
    # 对于每个结果，添加它来自哪个摘要/页面
    for result in detailed_results:
        page = result["metadata"]["page"]
        matching_summaries = [s for s in summary_results if s["metadata"]["page"] == page]
        if matching_summaries:
            result["summary"] = matching_summaries[0]["text"]
    
    return detailed_results

# 基于上下文的响应生成
def generate_response(query, retrieved_chunks):
    """
    基于查询和检索的块生成响应。
    
    参数:
        query (str): 用户查询
        retrieved_chunks (List[Dict]): 从层次化搜索中检索的块
        
    返回:
        str: 生成的响应
    """
    # 从块中提取文本并准备上下文部分
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks):
        page_num = chunk["metadata"]["page"]  # 从元数据中获取页码
        context_parts.append(f"[Page {page_num}]: {chunk['text']}")  # 用页码格式化块文本
    
    # 将所有上下文部分组合成单个上下文字符串
    context = "\n\n".join(context_parts)
    
    # 定义系统消息以指导AI助手
    system_message = """You are a helpful AI assistant answering questions based on the provided context.
Use the information from the context to answer the user's question accurately.
If the context doesn't contain relevant information, acknowledge that.
Include page numbers when referencing specific information."""

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定要使用的模型
        messages=[
            {"role": "system", "content": system_message},  # 系统消息，指导助手
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"}  # 用户消息，包含上下文和查询
        ],
        temperature=0.2  # 设置响应生成的温度
    )
    
    # 返回生成的响应内容
    return response.choices[0].message.content

# 带有层次化检索的完整RAG管道
def hierarchical_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200, 
                    k_summaries=3, k_chunks=5, regenerate=False):
    """
    完整的层次化RAG管道。
    
    参数:
        query (str): 用户查询
        pdf_path (str): PDF文档路径
        chunk_size (int): 每个详细块的大小
        chunk_overlap (int): 块之间的重叠
        k_summaries (int): 要检索的摘要数量
        k_chunks (int): 每个摘要要检索的块数量
        regenerate (bool): 是否重新生成向量存储
        
    返回:
        Dict: 包括响应和检索块的结果
    """
    # 创建存储文件名用于缓存
    summary_store_file = f"{os.path.basename(pdf_path)}_summary_store.pkl"
    detailed_store_file = f"{os.path.basename(pdf_path)}_detailed_store.pkl"
    
    # 如果需要，处理文档并创建存储
    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        print("Processing document and creating vector stores...")
        # 处理文档以创建层次化索引和向量存储
        summary_store, detailed_store = process_document_hierarchically(
            pdf_path, chunk_size, chunk_overlap
        )
        
        # 将摘要存储保存到文件中以供将来使用
        with open(summary_store_file, 'wb') as f:
            pickle.dump(summary_store, f)
        
        # 将详细存储保存到文件中以供将来使用
        with open(detailed_store_file, 'wb') as f:
            pickle.dump(detailed_store, f)
    else:
        # 从文件加载现有摘要存储
        print("Loading existing vector stores...")
        with open(summary_store_file, 'rb') as f:
            summary_store = pickle.load(f)
        
        # 从文件加载现有详细存储
        with open(detailed_store_file, 'rb') as f:
            detailed_store = pickle.load(f)
    
    # 使用查询层次化检索相关块
    retrieved_chunks = retrieve_hierarchically(
        query, summary_store, detailed_store, k_summaries, k_chunks
    )
    
    # 基于检索的块生成响应
    response = generate_response(query, retrieved_chunks)
    
    # 返回结果，包括查询、响应、检索块以及摘要和详细块的计数
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "summary_count": len(summary_store.texts),
        "detailed_count": len(detailed_store.texts)
    }

# 标准（非层次化）RAG用于比较
def standard_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200, k=15):
    """
    没有层次化检索的标准RAG管道。
    
    参数:
        query (str): 用户查询
        pdf_path (str): PDF文档路径
        chunk_size (int): 每个块的大小
        chunk_overlap (int): 块之间的重叠
        k (int): 要检索的块数量
        
    返回:
        Dict: 包括响应和检索块的结果
    """
    # 从PDF文档中提取页面
    pages = extract_text_from_pdf(pdf_path)
    
    # 直接从所有页面创建块
    chunks = []
    for page in pages:
        # 对页面的文本进行分块
        page_chunks = chunk_text(
            page["text"], 
            page["metadata"], 
            chunk_size, 
            chunk_overlap
        )
        # 用当前页面的块扩展chunks列表
        chunks.extend(page_chunks)
    
    print(f"Created {len(chunks)} chunks for standard RAG")
    
    # 创建一个向量存储来保存块
    store = SimpleVectorStore()
    
    # 为块创建嵌入
    print("Creating embeddings for chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)
    
    # 将块添加到向量存储中
    for i, chunk in enumerate(chunks):
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata=chunk["metadata"]
        )
    
    # 为查询创建嵌入
    query_embedding = create_embeddings(query)
    
    # 基于查询嵌入检索最相关的块
    retrieved_chunks = store.similarity_search(query_embedding, k=k)
    print(f"Retrieved {len(retrieved_chunks)} chunks with standard RAG")
    
    # 基于检索的块生成响应
    response = generate_response(query, retrieved_chunks)
    
    # 返回结果，包括查询、响应和检索块
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }

# 评估函数
def compare_approaches(query, pdf_path, reference_answer=None):
    """
    比较层次化和标准RAG方法。
    
    参数:
        query (str): 用户查询
        pdf_path (str): PDF文档路径
        reference_answer (str, optional): 用于评估的参考答案
        
    返回:
        Dict: 比较结果
    """
    print(f"\n=== Comparing RAG approaches for query: {query} ===")
    
    # 运行层次化RAG
    print("\nRunning hierarchical RAG...")
    hierarchical_result = hierarchical_rag(query, pdf_path)
    hier_response = hierarchical_result["response"]
    
    # 运行标准RAG
    print("\nRunning standard RAG...")
    standard_result = standard_rag(query, pdf_path)
    std_response = standard_result["response"]
    
    # 比较层次化和标准RAG的结果
    comparison = compare_responses(query, hier_response, std_response, reference_answer)
    
    # 返回包含比较结果的字典
    return {
        "query": query,  # 原始查询
        "hierarchical_response": hier_response,  # 层次化RAG的响应
        "standard_response": std_response,  # 标准RAG的响应
        "reference_answer": reference_answer,  # 用于评估的参考答案
        "comparison": comparison,  # 比较分析
        "hierarchical_chunks_count": len(hierarchical_result["retrieved_chunks"]),  # 层次化RAG检索的块数量
        "standard_chunks_count": len(standard_result["retrieved_chunks"])  # 标准RAG检索的块数量
    }

def compare_responses(query, hierarchical_response, standard_response, reference=None):
    """
    比较层次化和标准RAG的响应。
    
    参数:
        query (str): 用户查询
        hierarchical_response (str): 层次化RAG的响应
        standard_response (str): 标准RAG的响应
        reference (str, optional): 参考答案
        
    返回:
        str: 比较分析
    """
    # 定义系统提示以指导模型如何评估响应
    system_prompt = """You are an expert evaluator of information retrieval systems. 
Compare the two responses to the same query, one generated using hierarchical retrieval
and the other using standard retrieval.

Evaluate them based on:
1. Accuracy: Which response provides more factually correct information?
2. Comprehensiveness: Which response better covers all aspects of the query?
3. Coherence: Which response has better logical flow and organization?
4. Page References: Does either response make better use of page references?

Be specific in your analysis of the strengths and weaknesses of each approach."""

    # 创建包含查询和两个响应的用户提示
    user_prompt = f"""Query: {query}

Response from Hierarchical RAG:
{hierarchical_response}

Response from Standard RAG:
{standard_response}"""

    # 如果提供了参考答案，将其包含在用户提示中
    if reference:
        user_prompt += f"""

Reference Answer:
{reference}"""

    # 向用户提示添加最终指令
    user_prompt += """

Please provide a detailed comparison of these two responses, highlighting which approach performed better and why."""

    # 向OpenAI API发出请求以生成比较分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息，指导助手
            {"role": "user", "content": user_prompt}  # 用户消息，包含查询和响应
        ],
        temperature=0  # 设置响应生成的温度
    )
    
    # 返回生成的比较分析
    return response.choices[0].message.content

def run_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    使用多个测试查询运行完整评估。
    
    参数:
        pdf_path (str): PDF文档路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 查询的参考答案
        
    返回:
        Dict: 评估结果
    """
    results = []  # 初始化空列表以存储结果
    
    # 遍历测试查询中的每个查询
    for i, query in enumerate(test_queries):
        print(f"Query: {query}")  # 打印当前查询
        
        # 如果可用，获取参考答案
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]  # 检索当前查询的参考答案
        
        # 比较层次化和标准RAG方法
        result = compare_approaches(query, pdf_path, reference)
        results.append(result)  # 将结果添加到结果列表中
    
    # 生成评估结果的总体分析
    overall_analysis = generate_overall_analysis(results)
    
    return {
        "results": results,  # 返回单个结果
        "overall_analysis": overall_analysis  # 返回总体分析
    }

def generate_overall_analysis(results):
    """
    生成评估结果的总体分析。
    
    参数:
        results (List[Dict]): 来自单个查询评估的结果
        
    返回:
        str: 总体分析
    """
    # 定义系统提示以指导模型如何评估结果
    system_prompt = """You are an expert at evaluating information retrieval systems.
Based on multiple test queries, provide an overall analysis comparing hierarchical RAG 
with standard RAG.

Focus on:
1. When hierarchical retrieval performs better and why
2. When standard retrieval performs better and why
3. The overall strengths and weaknesses of each approach
4. Recommendations for when to use each approach"""

    # 创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Hierarchical chunks: {result['hierarchical_chunks_count']}, Standard chunks: {result['standard_chunks_count']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    # 定义包含评估摘要的用户提示
    user_prompt = f"""Based on the following evaluations comparing hierarchical vs standard RAG across {len(results)} queries, 
provide an overall analysis of these two approaches:

{evaluations_summary}

Please provide a comprehensive analysis of the relative strengths and weaknesses of hierarchical RAG 
compared to standard RAG, with specific focus on retrieval quality and response generation."""

    # 向OpenAI API发出请求以生成总体分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息，指导助手
            {"role": "user", "content": user_prompt}  # 用户消息，包含评估摘要
        ],
        temperature=0  # 设置响应生成的温度
    )
    
    # 返回生成的总体分析
    return response.choices[0].message.content


# 层次化和标准RAG方法的评估

# 包含AI信息的PDF文档路径
pdf_path = "data/AI_Information.pdf"

# 用于测试层次化RAG方法的关于AI的示例查询
query = "What are the key applications of transformer models in natural language processing?"
result = hierarchical_rag(query, pdf_path)

print("\n=== Response ===")
print(result["response"])

# 用于正式评估的测试查询（按要求仅使用一个查询）
test_queries = [
    "How do transformers handle sequential data compared to RNNs?"
]

# 测试查询的参考答案，用于启用比较
reference_answers = [
    "Transformers handle sequential data differently from RNNs by using self-attention mechanisms instead of recurrent connections. This allows transformers to process all tokens in parallel rather than sequentially, capturing long-range dependencies more efficiently and enabling better parallelization during training. Unlike RNNs, transformers don't suffer from vanishing gradient problems with long sequences."
]

# 运行评估，比较层次化和标准RAG方法
evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# 打印比较的总体分析
print("\n=== OVERALL ANALYSIS ===")
print(evaluation_results["overall_analysis"])

