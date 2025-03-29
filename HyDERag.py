#!/usr/bin/env python
# coding: utf-8

"""
假设文档嵌入技术
一种创新的检索技术，该技术在执行检索之前将用户查询转换为假设的答案文档。此方法弥合了短查询和长文档之间的语义差距。
传统的嵌入系统直接嵌入用户的短查询，但这通常无法捕捉到优化检索所需的语义丰富性。HyDE通过以下方式解决了这个问题：
- 生成一个回答查询的假设文档
- 嵌入这个扩展的文档而不是原始查询
- 检索与这个假设文档相似的文档
- 创建更具上下文相关性的答案
"""

"""
环境设置
我们首先导入必要的库。
"""

import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re
import matplotlib.pyplot as plt

"""
设置 OpenAI API 客户端
我们初始化 OpenAI 客户端以生成嵌入和响应。
"""

# 初始化 OpenAI 客户端，使用基本 URL 和 API 密钥
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中检索 API 密钥
)

"""
文档处理函数
"""

def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取文本内容并进行页面分隔。
    
    参数:
        pdf_path (str): PDF 文件路径
        
    返回:
        List[Dict]: 包含文本内容和元数据的页面列表
    """
    print(f"正在从 {pdf_path} 提取文本...")  # 打印正在处理的 PDF 的路径
    pdf = fitz.open(pdf_path)  # 使用 PyMuPDF 打开 PDF 文件
    pages = []  # 初始化一个空列表来存储包含文本内容的页面
    
    # 遍历 PDF 中的每一页
    for page_num in range(len(pdf)):
        page = pdf[page_num]  # 获取当前页
        text = page.get_text()  # 从当前页提取文本
        
        # 跳过文本很少的页面（少于 50 个字符）
        if len(text.strip()) > 50:
            # 将页面文本和元数据添加到列表中
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,  # 源文件路径
                    "page": page_num + 1  # 页码（从 1 开始的索引）
                }
            })
    
    print(f"提取了 {len(pages)} 页内容")  # 打印提取的页面数量
    return pages  # 返回包含文本内容和元数据的页面列表

"""
超厉害的文本分块函数
这个函数可以把文本切成小块，就像切蛋糕一样，不过每块之间还会有点重叠哦！
"""
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    把文本切成小块，每块之间还有点重叠，就像拼图一样
    
    参数:
        text (str): 要切的文本
        chunk_size (int): 每块的大小（字符数）
        overlap (int): 块之间的重叠字符数
        
    返回:
        List[Dict]: 带着元数据的小块列表
    """
    chunks = []  # 准备一个空盒子来装切好的小块
    
    # 以 (chunk_size - overlap) 为步长切蛋糕
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # 切下一块
        if chunk_text:  # 确保不切到空气
            chunks.append({
                "text": chunk_text,  # 把切好的块放进盒子
                "metadata": {
                    "start_pos": i,  # 这块在原文中的起始位置
                    "end_pos": i + len(chunk_text)  # 这块在原文中的结束位置
                }
            })
    
    print(f"Created {len(chunks)} text chunks")  # 数数切了多少块
    return chunks  # 把装着小块的盒子交出去


"""
简单向量存储类
这个类就像一个智能书架，可以存储文本和它们的向量表示
"""
class SimpleVectorStore:
    """
    一个用NumPy实现的简单向量存储
    """
    def __init__(self):
        self.vectors = []  # 存放向量的小抽屉
        self.texts = []  # 存放文本的小抽屉
        self.metadata = []  # 存放元数据的小抽屉
    
    def add_item(self, text, embedding, metadata=None):
        """
        往书架上放新书
        
        参数:
            text (str): 文本内容
            embedding (List[float]): 向量表示
            metadata (Dict, optional): 附加的元数据
        """
        self.vectors.append(np.array(embedding))  # 把向量放进抽屉
        self.texts.append(text)  # 把文本放进抽屉
        self.metadata.append(metadata or {})  # 把元数据放进抽屉，如果没有就放个空字典
    
    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        在书架上找最相似的书
        
        参数:
            query_embedding (List[float]): 查询向量
            k (int): 要返回的结果数量
            filter_func (callable, optional): 过滤结果的函数
            
        返回:
            List[Dict]: 最相似的k个结果
        """
        if not self.vectors:
            return []  # 如果书架上没书，就空手而归
        
        # 把查询向量变成NumPy数组
        query_vector = np.array(query_embedding)
        
        # 计算相似度，就像在找失散多年的兄弟
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 如果不符合过滤条件就跳过
            if filter_func and not filter_func(self.metadata[i]):
                continue
                
            # 计算余弦相似度
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 记录相似度和索引
            
        # 按相似度从高到低排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回最相似的k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 文本内容
                "metadata": self.metadata[idx],  # 元数据
                "similarity": float(score)  # 相似度分数
            })
        
        return results  # 把找到的结果交出去


"""
创建嵌入向量
这个函数就像是一个魔法师，可以把文本变成神奇的向量
"""
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为给定的文本创建嵌入向量
    
    参数:
        texts (List[str]): 输入文本
        model (str): 嵌入模型名称
        
    返回:
        List[List[float]]: 嵌入向量
    """
    # 如果输入为空，就返回空列表
    if not texts:
        return []
        
    # 如果需要，分批处理（OpenAI API有限制）
    batch_size = 100
    all_embeddings = []
    
    # 分批处理文本
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # 获取当前批次
        
        # 为当前批次创建嵌入
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        # 从响应中提取嵌入
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 把当前批次的嵌入加入列表
    
    return all_embeddings  # 返回所有嵌入


"""
文档处理流水线
这个函数就像一个工厂流水线，把PDF文档变成向量存储
"""
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为RAG处理文档
    
    参数:
        pdf_path (str): PDF文件路径
        chunk_size (int): 每块的大小（字符数）
        chunk_overlap (int): 块之间的重叠字符数
        
    返回:
        SimpleVectorStore: 包含文档块的向量存储
    """
    # 从PDF文件中提取文本
    pages = extract_text_from_pdf(pdf_path)
    
    # 处理每一页并创建块
    all_chunks = []
    for page in pages:
        # 把文本内容（字符串）传给chunk_text，而不是字典
        page_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)
        
        # 为每个块更新元数据
        for chunk in page_chunks:
            chunk["metadata"].update(page["metadata"])
        
        all_chunks.extend(page_chunks)
    
    # 为文本块创建嵌入
    print("Creating embeddings for chunks...")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)
    
    # 创建向量存储来存放块和它们的嵌入
    vector_store = SimpleVectorStore()
    for i, chunk in enumerate(all_chunks):
        vector_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )
    
    print(f"Vector store created with {len(all_chunks)} chunks")
    return vector_store


"""
假设文档生成器
这个函数就像一个预言家，可以根据问题生成一个假设的答案文档
"""
def generate_hypothetical_document(query, desired_length=1000):
    """
    生成一个回答问题的假设文档
    
    参数:
        query (str): 用户问题
        desired_length (int): 目标文档长度
        
    返回:
        str: 生成的假设文档
    """
    # 定义系统提示，指导模型如何生成文档
    system_prompt = f"""You are an expert document creator. 
    Given a question, generate a detailed document that would directly answer this question.
    The document should be approximately {desired_length} characters long and provide an in-depth, 
    informative answer to the question. Write as if this document is from an authoritative source
    on the subject. Include specific details, facts, and explanations.
    Do not mention that this is a hypothetical document - just write the content directly."""

    # 定义用户提示，包含问题
    user_prompt = f"Question: {query}\n\nGenerate a document that fully answers this question:"
    
    # 向OpenAI API发送请求生成假设文档
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息指导助手
            {"role": "user", "content": user_prompt}  # 用户消息包含问题
        ],
        temperature=0.1  # 设置生成响应的温度
    )
    
    # 返回生成的文档内容
    return response.choices[0].message.content


"""
完整的HyDE RAG实现
这个函数就像一个智能助手，可以用假设文档嵌入来回答问题
"""
def hyde_rag(query, vector_store, k=5, should_generate_response=True):
    """
    使用假设文档嵌入执行RAG
    
    参数:
        query (str): 用户问题
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        k (int): 要检索的块数量
        generate_response (bool): 是否生成最终响应
        
    返回:
        Dict: 包含假设文档和检索到的块的结果
    """
    print(f"\n=== Processing query with HyDE: {query} ===\n")
    
    # 生成一个回答问题的假设文档
    print("Generating hypothetical document...")
    hypothetical_doc = generate_hypothetical_document(query)
    print(f"Generated hypothetical document of {len(hypothetical_doc)} characters")
    
    # 为假设文档创建嵌入
    print("Creating embedding for hypothetical document...")
    hypothetical_embedding = create_embeddings([hypothetical_doc])[0]
    
    # 检索相似块基于假设文档
    print(f"Retrieving {k} most similar chunks...")
    retrieved_chunks = vector_store.similarity_search(hypothetical_embedding, k=k)
    
    # 准备结果字典
    results = {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "retrieved_chunks": retrieved_chunks
    }
    
    # 生成最终响应
    if should_generate_response:
        print("Generating final response...")
        response = generate_response(query, retrieved_chunks)
        results["response"] = response
    
    return results


"""
标准RAG实现
这个函数就像一个直来直去的助手，直接用查询嵌入来回答问题
"""
def standard_rag(query, vector_store, k=5, should_generate_response=True):
    """
    使用直接查询嵌入执行RAG
    
    参数:
        query (str): 用户问题
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        k (int): 要检索的块数量
        generate_response (bool): 是否生成最终响应
        
    返回:
        Dict: 包含检索到的块的结果
    """
    print(f"\n=== Processing query with Standard RAG: {query} ===\n")
    
    # 第一步：为查询创建嵌入
    print("Creating embedding for query...")
    query_embedding = create_embeddings([query])[0]
    
    # 第二步：基于查询嵌入检索相似块
    print(f"Retrieving {k} most similar chunks...")
    retrieved_chunks = vector_store.similarity_search(query_embedding, k=k)
    
    # 准备结果字典
    results = {
        "query": query,
        "retrieved_chunks": retrieved_chunks
    }
    
    # 第三步：如果需要，生成最终响应
    if should_generate_response:
        print("Generating final response...")
        response = generate_response(query, retrieved_chunks)
        results["response"] = response
        
    return results


"""
响应生成器
这个函数就像一个翻译官，把检索到的块变成流畅的回答
"""
def generate_response(query, relevant_chunks):
    """
    基于问题和相关块生成最终响应
    
    参数:
        query (str): 用户问题
        relevant_chunks (List[Dict]): 检索到的相关块
        
    返回:
        str: 生成的响应
    """
    # 把块的文本拼接起来创建上下文
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    
    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.5,
        max_tokens=500
    )
    
    return response.choices[0].message.content


"""
评估函数
这些函数就像裁判，可以比较HyDE和标准RAG的表现
"""

def compare_approaches(query, vector_store, reference_answer=None):
    """
    比较HyDE和标准RAG方法
    
    参数:
        query (str): 用户问题
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        reference_answer (str, optional): 用于评估的参考答案
        
    返回:
        Dict: 比较结果
    """
    # 运行HyDE RAG
    hyde_result = hyde_rag(query, vector_store)
    hyde_response = hyde_result["response"]
    
    # 运行标准RAG
    standard_result = standard_rag(query, vector_store)
    standard_response = standard_result["response"]
    
    # 比较结果
    comparison = compare_responses(query, hyde_response, standard_response, reference_answer)
    
    return {
        "query": query,
        "hyde_response": hyde_response,
        "hyde_hypothetical_doc": hyde_result["hypothetical_document"],
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "comparison": comparison
    }


def compare_responses(query, hyde_response, standard_response, reference=None):
    """
    比较HyDE和标准RAG的响应
    
    参数:
        query (str): 用户问题
        hyde_response (str): HyDE RAG的响应
        standard_response (str): 标准RAG的响应
        reference (str, optional): 参考答案
        
    返回:
        str: 比较分析
    """
    system_prompt = """You are an expert evaluator of information retrieval systems.
Compare the two responses to the same query, one generated using HyDE (Hypothetical Document Embedding) 
and the other using standard RAG with direct query embedding.

Evaluate them based on:
1. Accuracy: Which response provides more factually correct information?
2. Relevance: Which response better addresses the query?
3. Completeness: Which response provides more thorough coverage of the topic?
4. Clarity: Which response is better organized and easier to understand?

Be specific about the strengths and weaknesses of each approach."""

    user_prompt = f"""Query: {query}

Response from HyDE RAG:
{hyde_response}

Response from Standard RAG:
{standard_response}"""

    if reference:
        user_prompt += f"""

Reference Answer:
{reference}"""

    user_prompt += """

Please provide a detailed comparison of these two responses, highlighting which approach performed better and why."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content


def run_evaluation(pdf_path, test_queries, reference_answers=None, chunk_size=1000, chunk_overlap=200):
    """
    使用多个测试问题运行完整评估
    
    参数:
        pdf_path (str): PDF文档路径
        test_queries (List[str]): 测试问题列表
        reference_answers (List[str], optional): 问题的参考答案
        chunk_size (int): 每块的大小（字符数）
        chunk_overlap (int): 块之间的重叠字符数
        
    返回:
        Dict: 评估结果
    """
    # 处理文档并创建向量存储
    vector_store = process_document(pdf_path, chunk_size, chunk_overlap)
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n\n===== Evaluating Query {i+1}/{len(test_queries)} =====")
        print(f"Query: {query}")
        
        # 如果有参考答案，获取它
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        
        # 比较方法
        result = compare_approaches(query, vector_store, reference)
        results.append(result)
    
    # 生成整体分析
    overall_analysis = generate_overall_analysis(results)
    
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results):
    """
    生成评估结果的整体分析
    
    参数:
        results (List[Dict]): 单个查询评估的结果
        
    返回:
        str: 整体分析
    """
    system_prompt = """You are an expert at evaluating information retrieval systems.
Based on multiple test queries, provide an overall analysis comparing HyDE RAG (using hypothetical document embedding)
with standard RAG (using direct query embedding).

Focus on:
1. When HyDE performs better and why
2. When standard RAG performs better and why
3. The types of queries that benefit most from HyDE
4. The overall strengths and weaknesses of each approach
5. Recommendations for when to use each approach"""

    # 创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    user_prompt = f"""Based on the following evaluations comparing HyDE vs standard RAG across {len(results)} queries, 
provide an overall analysis of these two approaches:

{evaluations_summary}

Please provide a comprehensive analysis of the relative strengths and weaknesses of HyDE compared to standard RAG,
focusing on when and why one approach outperforms the other."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content


"""
可视化函数
这些函数就像画家，可以把结果变成直观的图表
"""

def visualize_results(query, hyde_result, standard_result):
    """
    可视化HyDE和标准RAG方法的结果
    
    参数:
        query (str): 用户问题
        hyde_result (Dict): HyDE RAG的结果
        standard_result (Dict): 标准RAG的结果
    """
    # 创建一个包含3个子图的图形
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # 在第一个子图中绘制查询
    axs[0].text(0.5, 0.5, f"Query:\n\n{query}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, wrap=True)
    axs[0].axis('off')  # 隐藏查询图的坐标轴
    
    # 在第二个子图中绘制假设文档
    hypothetical_doc = hyde_result["hypothetical_document"]
    # 如果假设文档太长，就缩短它
    shortened_doc = hypothetical_doc[:500] + "..." if len(hypothetical_doc) > 500 else hypothetical_doc
    axs[1].text(0.5, 0.5, f"Hypothetical Document:\n\n{shortened_doc}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, wrap=True)
    axs[1].axis('off')  # 隐藏假设文档图的坐标轴
    
    # 在第三个子图中绘制检索到的块的比较
    # 缩短每个块的文本以便更好地可视化
    hyde_chunks = [chunk["text"][:100] + "..." for chunk in hyde_result["retrieved_chunks"]]
    std_chunks = [chunk["text"][:100] + "..." for chunk in standard_result["retrieved_chunks"]]
    
    # 准备比较文本
    comparison_text = "Retrieved by HyDE:\n\n"
    for i, chunk in enumerate(hyde_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"
    
    comparison_text += "\nRetrieved by Standard RAG:\n\n"
    for i, chunk in enumerate(std_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"
    
    # 在第三个子图中绘制比较文本
    axs[2].text(0.5, 0.5, comparison_text, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, wrap=True)
    axs[2].axis('off')  # 隐藏比较图的坐标轴
    
    # 调整布局以防止重叠
    plt.tight_layout()
    # 显示图形
    plt.show()


"""
假设文档嵌入（HyDE）与标准RAG的评估
"""

# AI信息文档的路径
pdf_path = "data/AI_Information.pdf"

# 处理文档并创建向量存储
# 这会加载文档，提取文本，分块，并创建嵌入
vector_store = process_document(pdf_path)

# 示例1：直接比较一个与AI相关的问题
query = "What are the main ethical considerations in artificial intelligence development?"

# 运行HyDE RAG方法
# 这会生成一个回答问题的假设文档，嵌入它，
# 并使用该嵌入来检索相关块
hyde_result = hyde_rag(query, vector_store)
print("\n=== HyDE Response ===")
print(hyde_result["response"])

# 运行标准RAG方法进行比较
# 这会直接嵌入查询并使用它来检索相关块
standard_result = standard_rag(query, vector_store)
print("\n=== Standard RAG Response ===")
print(standard_result["response"])

# 可视化HyDE和标准RAG方法之间的差异
# 并排显示查询、假设文档和检索到的块
visualize_results(query, hyde_result, standard_result)

# 示例2：使用多个AI相关问题运行完整评估
test_queries = [
    "How does neural network architecture impact AI performance?"
]

# 用于更好评估的可选参考答案
reference_answers = [
    "Neural network architecture significantly impacts AI performance through factors like depth (number of layers), width (neurons per layer), connectivity patterns, and activation functions. Different architectures like CNNs, RNNs, and Transformers are optimized for specific tasks such as image recognition, sequence processing, and natural language understanding respectively.",
]

# 运行全面评估，比较HyDE和标准RAG方法
evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# 打印整体分析，显示哪种方法在多个查询中表现更好
print("\n=== OVERALL ANALYSIS ===")
print(evaluation_results["overall_analysis"])

