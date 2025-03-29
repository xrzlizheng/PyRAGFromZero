#!/usr/bin/env python
# coding: utf-8

"""
# 相关段落提取（RSE）增强版RAG
#
# 在这个脚本中，我们实现了一个相关段落提取（RSE）技术，用来提升RAG系统的上下文质量。
# 与其简单地检索一堆孤立的文本块，我们识别并重建连续的文本段落，为语言模型提供更好的上下文。
#
# ## 核心概念
#
# 相关文本块往往会在文档中聚集在一起。通过识别这些集群并保持它们的连续性，
# 我们为LLM提供了更连贯的上下文。
"""

# 首先，让我们导入必要的库，就像厨师准备食材一样
import fitz
import os
import numpy as np
import json
from openai import OpenAI
import re


def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本，就像从西瓜里挖出果肉一样简单
    
    参数：
    pdf_path (str): PDF文件的路径

    返回：
    str: 从PDF中提取的文本
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串来存储提取的文本

    # 遍历PDF的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取当前页
        text = page.get_text("text")  # 从页面提取文本
        all_text += text  # 将提取的文本追加到all_text字符串中

    return all_text  # 返回提取的文本


def chunk_text(text, chunk_size=800, overlap=0):
    """
    将文本分割成不重叠的块，就像把披萨切成均匀的切片
    
    参数：
        text (str): 要分割的输入文本
        chunk_size (int): 每个块的大小（字符数）
        overlap (int): 块之间的重叠字符数
        
    返回：
        List[str]: 文本块列表
    """
    chunks = []
    
    # 简单的基于字符的分块
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:  # 确保不添加空块
            chunks.append(chunk)
    
    return chunks


# 初始化OpenAI客户端，就像启动一台超级计算机
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)


# 构建一个简单的向量存储
# 让我们实现一个简单的向量存储。
# 我们将使用NumPy作为向量存储引擎，就像一个图书馆的数据库   

class SimpleVectorStore:
    """
    一个轻量级的向量存储实现，使用NumPy作为引擎
    """
    def __init__(self, dimension=1536):
        """
        初始化向量存储，就像建造一个数据仓库
        
        参数：
            dimension (int): 嵌入的维度
        """
        self.dimension = dimension
        self.vectors = []
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents, vectors=None, metadata=None):
        """
        向向量存储添加文档，就像往书架上摆放书籍
        
        参数：
            documents (List[str]): 文档块列表
            vectors (List[List[float]], 可选): 嵌入向量列表
            metadata (List[Dict], 可选): 元数据字典列表
        """
        if vectors is None:
            vectors = [None] * len(documents)
        
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        
        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(vec)
            self.metadata.append(meta)
    
    def search(self, query_vector, top_k=5):
        """
        搜索最相似的文档，就像在图书馆里找书
        
        参数：
            query_vector (List[float]): 查询嵌入向量
            top_k (int): 返回的结果数量
            
        返回：
            List[Dict]: 包含文档、分数和元数据的结果列表
        """
        if not self.vectors or not self.documents:
            return []
        
        # 将查询向量转换为numpy数组
        query_array = np.array(query_vector)
        
        # 计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            if vector is not None:
                # 计算余弦相似度
                similarity = np.dot(query_array, vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(vector)
                )
                similarities.append((i, similarity))
        
        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 获取top-k结果
        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                "score": float(score),
                "metadata": self.metadata[i]
            })
        
        return results


def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为文本生成嵌入，就像把文字变成数字密码
    
    参数：
        texts (List[str]): 要嵌入的文本列表
        model (str): 使用的嵌入模型
        
    返回：
        List[List[float]]: 嵌入向量列表
    """
    if not texts:
        return []  # 如果没有提供文本，返回空列表
        
    # 如果列表很长，分批处理
    batch_size = 100  # 根据API限制调整
    all_embeddings = []  # 初始化一个列表来存储所有嵌入
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # 获取当前批次的文本
        
        # 使用指定模型为当前批次创建嵌入
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        
        # 从响应中提取嵌入
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 将批次嵌入添加到列表中
        
    return all_embeddings  # 返回所有嵌入的列表


def process_document(pdf_path, chunk_size=800):
    """
    处理文档以供RSE使用，就像准备一顿丰盛的大餐
    
    参数：
        pdf_path (str): PDF文档的路径
        chunk_size (int): 每个块的大小（字符数）
        
    返回：
        Tuple[List[str], SimpleVectorStore, Dict]: 文本块、向量存储和文档信息
    """
    print("Extracting text from document...")
    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)
    
    print("Chunking text into non-overlapping segments...")
    # 将提取的文本分割成不重叠的段落
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
    print(f"Created {len(chunks)} chunks")
    
    print("Generating embeddings for chunks...")
    # 为文本块生成嵌入
    chunk_embeddings = create_embeddings(chunks)
    
    # 创建SimpleVectorStore实例
    vector_store = SimpleVectorStore()
    
    # 添加带有元数据的文档（包括块索引以便后续重建）
    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]
    vector_store.add_documents(chunks, chunk_embeddings, metadata)
    
    # 跟踪原始文档结构以便段落重建
    doc_info = {
        "chunks": chunks,
        "source": pdf_path,
    }
    
    return chunks, vector_store, doc_info


def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    """
    通过结合相关性和位置来计算块值，就像给每个块打分一样
    
    参数：
        query (str): 查询文本
        chunks (List[str]): 文档块列表
        vector_store (SimpleVectorStore): 包含块的向量存储
        irrelevant_chunk_penalty (float): 不相关块的惩罚值
        
    返回：
        List[float]: 块值列表
    """
    # 创建查询嵌入
    query_embedding = create_embeddings([query])[0]
    
    # 获取所有带有相似度得分的块
    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)
    
    # 创建块索引到相关性得分的映射
    relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}
    
    # 计算块值（相关性得分减去惩罚值）
    chunk_values = []
    for i in range(num_chunks):
        # 获取相关性得分，如果不在结果中则默认为0
        score = relevance_scores.get(i, 0.0)
        # 应用惩罚值，将不相关块的值变为负值
        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)
    
    return chunk_values


def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    使用最大子数组算法的变体找到最佳段落，就像在迷宫中寻找最佳路径
    
    参数：
        chunk_values (List[float]): 每个块的值
        max_segment_length (int): 单个段落的最大长度
        total_max_length (int): 所有段落的总最大长度
        min_segment_value (float): 段落被考虑的最小值
        
    返回：
        List[Tuple[int, int]]: 最佳段落的（开始，结束）索引列表
    """
    print("Finding optimal continuous text segments...")
    
    best_segments = []
    segment_scores = []
    total_included_chunks = 0
    
    # 不断寻找段落，直到达到我们的限制
    while total_included_chunks < total_max_length:
        best_score = min_segment_value  # 段落的最小阈值
        best_segment = None
        
        # 尝试每个可能的起始位置
        for start in range(len(chunk_values)):
            # 如果这个起始位置已经在选定的段落中，跳过
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue
                
            # 尝试每个可能的段落长度
            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length
                
                # 如果结束位置已经在选定的段落中，跳过
                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue
                
                # 计算段落值作为块值的总和
                segment_value = sum(chunk_values[start:end])
                
                # 如果这个段落更好，更新最佳段落
                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)
        
        # 如果找到了一个好的段落，添加它
        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            print(f"Found segment {best_segment} with score {best_score:.4f}")
        else:
            # 没有更多好的段落可找了
            break
    
    # 按起始位置排序以便阅读
    best_segments = sorted(best_segments, key=lambda x: x[0])
    
    return best_segments, segment_scores


def reconstruct_segments(chunks, best_segments):
    """
    根据块索引重建文本段落，就像拼图一样
    
    参数：
        chunks (List[str]): 所有文档块的列表
        best_segments (List[Tuple[int, int]]): 段落的（开始，结束）索引列表
        
    返回：
        List[str]: 重建的文本段落列表
    """
    reconstructed_segments = []  # 初始化一个空列表来存储重建的段落
    
    for start, end in best_segments:
        # 将这个段落中的块连接起来形成完整的段落文本
        segment_text = " ".join(chunks[start:end])
        # 将段落文本及其范围添加到reconstructed_segments列表中
        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })
    
    return reconstructed_segments  # 返回重建的文本段落列表


def format_segments_for_context(segments):
    """
    将段落格式化为LLM的上下文字符串，就像准备一份精美的菜单
    
    参数：
        segments (List[Dict]): 段落字典列表
        
    返回：
        str: 格式化后的上下文文本
    """
    context = []  # 初始化一个空列表来存储格式化的上下文
    
    for i, segment in enumerate(segments):
        # 为每个段落创建一个标题，包含索引和块范围
        segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
        context.append(segment_header)  # 将段落标题添加到上下文列表
        context.append(segment['text'])  # 将段落文本添加到上下文列表
        context.append("-" * 80)  # 添加分隔线以提高可读性
    
    # 用双换行符连接上下文列表中的所有元素并返回结果
    return "\n\n".join(context)


def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    根据查询和上下文生成响应，就像一位聪明的助手
    
    参数：
        query (str): 用户查询
        context (str): 来自相关段落的上下文文本
        model (str): 使用的LLM模型
        
    返回：
        str: 生成的响应
    """
    print("Generating response using relevant segments as context...")
    
    # 定义系统提示以指导AI的行为
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
    The context consists of document segments that have been retrieved as relevant to the user's query.
    Use the information from these segments to provide a comprehensive and accurate answer.
    If the context doesn't contain relevant information to answer the question, say so clearly."""
    
    # 通过结合上下文和查询创建用户提示
    user_prompt = f"""
Context:
{context}

Question: {query}

Please provide a helpful answer based on the context provided.
"""
    
    # 使用指定模型生成响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 返回生成的响应内容
    return response.choices[0].message.content


def rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    """
    完整的RAG管道，包含相关段落提取，就像一条高效的生产线
    
    参数：
        pdf_path (str): 文档路径
        query (str): 用户查询
        chunk_size (int): 块大小
        irrelevant_chunk_penalty (float): 不相关块的惩罚值
        
    返回：
        Dict: 包含查询、段落和响应的结果
    """
    print("\n=== STARTING RAG WITH RELEVANT SEGMENT EXTRACTION ===")
    print(f"Query: {query}")
    
    # 处理文档以提取文本、分块并创建嵌入
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)
    
    # 根据查询计算相关性得分和块值
    print("\nCalculating relevance scores and chunk values...")
    chunk_values = calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty)
    
    # 根据块值找到最佳文本段落
    best_segments, scores = find_best_segments(
        chunk_values, 
        max_segment_length=20, 
        total_max_length=30, 
        min_segment_value=0.2
    )
    
    # 从最佳块重建文本段落
    print("\nReconstructing text segments from chunks...")
    segments = reconstruct_segments(chunks, best_segments)
    
    # 将段落格式化为语言模型的上下文字符串
    context = format_segments_for_context(segments)
    
    # 使用上下文从语言模型生成响应
    response = generate_response(query, context)
    
    # 将结果编译成字典
    result = {
        "query": query,
        "segments": segments,
        "response": response
    }
    
    print("\n=== FINAL RESPONSE ===")
    print(response)
    
    return result


def standard_top_k_retrieval(pdf_path, query, k=10, chunk_size=800):
    """
    标准RAG与top-k检索，就像传统的图书馆检索系统
    
    参数：
        pdf_path (str): 文档路径
        query (str): 用户查询
        k (int): 要检索的块数量
        chunk_size (int): 块大小
        
    返回：
        Dict: 包含查询、块和响应的结果
    """
    print("\n=== STARTING STANDARD TOP-K RETRIEVAL ===")
    print(f"Query: {query}")
    
    # 处理文档以提取文本、分块并创建嵌入
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)
    
    # 创建查询的嵌入
    print("Creating query embedding and retrieving chunks...")
    query_embedding = create_embeddings([query])[0]
    
    # 根据查询嵌入检索top-k最相关的块
    results = vector_store.search(query_embedding, top_k=k)
    retrieved_chunks = [result["document"] for result in results]
    
    # 将检索到的块格式化为上下文字符串
    context = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk}" 
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    # 使用上下文从语言模型生成响应
    response = generate_response(query, context)
    
    # 将结果编译成字典
    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }
    
    print("\n=== FINAL RESPONSE ===")
    print(response)
    
    return result


# ## RSE评估
# 让我们来比较一下RSE和标准top-k检索的效果

def evaluate_methods(pdf_path, query, reference_answer=None):
    """
    比较RSE与标准top-k检索，就像比较两种不同的烹饪方法
    
    参数：
        pdf_path (str): 文档路径
        query (str): 用户查询
        reference_answer (str, 可选): 用于评估的参考答案
    """
    print("\n========= 评估开始 =========\n")
    
    # 运行带有相关段落提取（RSE）的RAG方法
    rse_result = rag_with_rse(pdf_path, query)
    
    # 运行标准top-k检索方法
    standard_result = standard_top_k_retrieval(pdf_path, query)
    
    # 如果提供了参考答案，则评估响应
    if reference_answer:
        print("\n=== 结果比较 ===")
        
        # 创建一个评估提示，将响应与参考答案进行比较
        evaluation_prompt = f"""
            Query: {query}

            Reference Answer:
            {reference_answer}

            Response from Standard Retrieval:
            {standard_result["response"]}

            Response from Relevant Segment Extraction:
            {rse_result["response"]}

            Compare these two responses against the reference answer. Which one is:
            1. More accurate and comprehensive
            2. Better at addressing the user's query
            3. Less likely to include irrelevant information

            Explain your reasoning for each point.
        """
        
        print("根据参考答案评估响应...")
        
        # 使用指定的模型生成评估
        evaluation = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": "You are an objective evaluator of RAG system responses."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        
        # 打印评估结果
        print("\n=== 评估结果 ===")
        print(evaluation.choices[0].message.content)
    
    # 返回两种方法的结果
    return {
        "rse_result": rse_result,
        "standard_result": standard_result
    }


# FromJSON文件加载验证数据
with open('data/val.json') as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[0]['question']

# 从验证数据中提取参考答案
reference_answer = data[0]['ideal_answer']

# pdf路径
pdf_path = "data/AI_Information.pdf"

# 运行评估
results = evaluate_methods(pdf_path, query, reference_answer)

