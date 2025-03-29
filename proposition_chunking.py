#!/usr/bin/env python
# coding: utf-8

"""
# 命题分块技术增强RAG系统

在本脚本中，我实现了命题分块技术 - 一种先进的办法，将文档分解为原子化的事实陈述，以实现更精确的检索。与传统的按字符数量简单分割文本的分块方法不同，命题分块保留了单个事实的语义完整性。

命题分块通过以下方式提供更精确的检索：

1. 将内容分解为原子化、自包含的事实
2. 创建更小、更精细的检索单元
3. 实现查询与相关内容之间更精确的匹配
4. 过滤掉低质量或不完整的命题

让我们构建一个完整的实现，不依赖LangChain或FAISS。
"""

# ## 环境设置
# 首先导入必要的库

import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re


"""
## 从PDF文件中提取文本
为了实现RAG，我们首先需要文本数据源。在这个例子中，我们使用PyMuPDF库从PDF文件中提取文本。
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
一旦我们提取了文本，我们将其分成更小的、重叠的块以提高检索准确性。
"""

def chunk_text(text, chunk_size=800, overlap=100):
    """
    将文本分割成重叠的块。
    
    参数:
        text (str): 要分块的输入文本
        chunk_size (int): 每个块的字符大小
        overlap (int): 块之间重叠的字符数
        
    返回:
        List[Dict]: 包含文本和元数据的块字典列表
    """
    chunks = []  # 初始化一个空列表来存储块
    
    # 使用指定的块大小和重叠遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]  # 提取指定大小的块
        if chunk:  # 确保我们不添加空块
            chunks.append({
                "text": chunk,  # 块文本
                "chunk_id": len(chunks) + 1,  # 块的唯一ID
                "start_char": i,  # 块的起始字符索引
                "end_char": i + len(chunk)  # 块的结束字符索引
            })
    
    print(f"Created {len(chunks)} text chunks")  # 打印创建的块数量
    return chunks  # 返回块列表


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
        # 初始化列表以存储向量、文本和元数据
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加项目。
        
        参数:
            text (str): 文本内容
            embedding (List[float]): 嵌入向量
            metadata (Dict, optional): 附加元数据
        """
        # 将嵌入、文本和元数据附加到各自的列表中
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def add_items(self, texts, embeddings, metadata_list=None):
        """
        向向量存储中添加多个项目。
        
        参数:
            texts (List[str]): 文本内容列表
            embeddings (List[List[float]]): 嵌入向量列表
            metadata_list (List[Dict], optional): 元数据字典列表
        """
        # 如果没有提供元数据列表，为每个文本创建一个空字典
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(texts))]
        
        # 将每个文本、嵌入和元数据添加到存储中
        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            self.add_item(text, embedding, metadata)
    
    def similarity_search(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目。
        
        参数:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 要返回的结果数量
            
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
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 收集前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)  # 转换为float以便JSON序列化
            })
        
        return results


"""
## 创建嵌入
"""

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为给定文本创建嵌入。
    
    参数:
        texts (str or List[str]): 输入文本
        model (str): 嵌入模型名称
        
    返回:
        List[List[float]]: 嵌入向量
    """
    # 处理字符串和列表输入
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
    
    # 如果输入是单个字符串，则仅返回第一个嵌入
    if isinstance(texts, str):
        return all_embeddings[0]
    
    # 否则，返回所有嵌入
    return all_embeddings


"""
## 命题生成
"""

def generate_propositions(chunk):
    """
    从文本块生成原子化、自包含的命题。
    
    参数:
        chunk (Dict): 包含内容和元数据的文本块
        
    返回:
        List[str]: 生成的命题列表
    """
    # 系统提示，指导AI如何生成命题
    system_prompt = """Please break down the following text into simple, self-contained propositions. 
    Ensure that each proposition meets the following criteria:

    1. Express a Single Fact: Each proposition should state one specific fact or claim.
    2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
    3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
    4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
    5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses.

    Output ONLY the list of propositions without any additional text or explanations."""

    # 用户提示，包含要转换为命题的文本块
    user_prompt = f"Text to convert into propositions:\n\n{chunk['text']}"
    
    # 从模型生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 使用更强大的模型进行准确的命题生成
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从响应中提取命题
    raw_propositions = response.choices[0].message.content.strip().split('\n')
    
    # 清理命题（删除编号、项目符号等）
    clean_propositions = []
    for prop in raw_propositions:
        # 删除编号（1., 2.等）和项目符号
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:  # 对空或非常短的命题进行简单过滤
            clean_propositions.append(cleaned)
    
    return clean_propositions


"""
## 命题质量检查
"""

def evaluate_proposition(proposition, original_text):
    """
    基于准确性、清晰度、完整性和简洁性评估命题的质量。
    
    参数:
        proposition (str): 要评估的命题
        original_text (str): 用于比较的原始文本
        
    返回:
        Dict: 每个评估维度的分数
    """
    # 系统提示，指导AI如何评估命题
    system_prompt = """You are an expert at evaluating the quality of propositions extracted from text.
    Rate the given proposition on the following criteria (scale 1-10):

    - Accuracy: How well the proposition reflects information in the original text
    - Clarity: How easy it is to understand the proposition without additional context
    - Completeness: Whether the proposition includes necessary details (dates, qualifiers, etc.)
    - Conciseness: Whether the proposition is concise without losing important information

    The response must be in valid JSON format with numerical scores for each criterion:
    {"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}
    """

    # 用户提示，包含命题和原始文本
    user_prompt = f"""Proposition: {proposition}

    Original Text: {original_text}

    Please provide your evaluation scores in JSON format."""

    # 从模型生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    # 解析JSON响应
    try:
        scores = json.loads(response.choices[0].message.content.strip())
        return scores
    except json.JSONDecodeError:
        # 如果JSON解析失败，则使用回退值
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }


"""
## 完整命题处理流程
"""

def process_document_into_propositions(pdf_path, chunk_size=800, chunk_overlap=100, 
                                      quality_thresholds=None):
    """
    将文档处理成经过质量检查的命题。
    
    参数:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的字符大小
        chunk_overlap (int): 块之间重叠的字符数
        quality_thresholds (Dict): 命题质量的阈值分数
        
    返回:
        Tuple[List[Dict], List[Dict]]: 原始块和命题块
    """
    # 如果没有提供质量阈值，则设置默认值
    if quality_thresholds is None:
        quality_thresholds = {
            "accuracy": 7,
            "clarity": 7,
            "completeness": 7,
            "conciseness": 7
        }
    
    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)
    
    # 从提取的文本创建块
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # 初始化一个列表来存储所有命题
    all_propositions = []
    
    print("Generating propositions from chunks...")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        # 为当前块生成命题
        chunk_propositions = generate_propositions(chunk)
        print(f"Generated {len(chunk_propositions)} propositions")
        
        # 处理每个生成的命题
        for prop in chunk_propositions:
            proposition_data = {
                "text": prop,
                "source_chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"]
            }
            all_propositions.append(proposition_data)
    
    # 评估生成的命题的质量
    print("\nEvaluating proposition quality...")
    quality_propositions = []
    
    for i, prop in enumerate(all_propositions):
        if i % 10 == 0:  # 每10个命题更新一次状态
            print(f"Evaluating proposition {i+1}/{len(all_propositions)}...")
            
        # 评估当前命题的质量
        scores = evaluate_proposition(prop["text"], prop["source_text"])
        prop["quality_scores"] = scores
        
        # 检查命题是否通过质量阈值
        passes_quality = True
        for metric, threshold in quality_thresholds.items():
            if scores.get(metric, 0) < threshold:
                passes_quality = False
                break
        
        if passes_quality:
            quality_propositions.append(prop)
        else:
            print(f"Proposition failed quality check: {prop['text'][:50]}...")
    
    print(f"\nRetained {len(quality_propositions)}/{len(all_propositions)} propositions after quality filtering")
    
    return chunks, quality_propositions


"""
## 为两种方法构建向量存储
"""

def build_vector_stores(chunks, propositions):
    """
    为基于块和基于命题的方法构建向量存储。
    
    参数:
        chunks (List[Dict]): 原始文档块
        propositions (List[Dict]): 经过质量过滤的命题
        
    返回:
        Tuple[SimpleVectorStore, SimpleVectorStore]: 块和命题向量存储
    """
    # 为块创建向量存储
    chunk_store = SimpleVectorStore()
    
    # 提取块文本并创建嵌入
    chunk_texts = [chunk["text"] for chunk in chunks]
    print(f"Creating embeddings for {len(chunk_texts)} chunks...")
    chunk_embeddings = create_embeddings(chunk_texts)
    
    # 将块添加到向量存储中，带有元数据
    chunk_metadata = [{"chunk_id": chunk["chunk_id"], "type": "chunk"} for chunk in chunks]
    chunk_store.add_items(chunk_texts, chunk_embeddings, chunk_metadata)
    
    # 为命题创建向量存储
    prop_store = SimpleVectorStore()
    
    # 提取命题文本并创建嵌入
    prop_texts = [prop["text"] for prop in propositions]
    print(f"Creating embeddings for {len(prop_texts)} propositions...")
    prop_embeddings = create_embeddings(prop_texts)
    
    # 将命题添加到向量存储中，带有元数据
    prop_metadata = [
        {
            "type": "proposition", 
            "source_chunk_id": prop["source_chunk_id"],
            "quality_scores": prop["quality_scores"]
        } 
        for prop in propositions
    ]
    prop_store.add_items(prop_texts, prop_embeddings, prop_metadata)
    
    return chunk_store, prop_store


"""
## 查询和检索函数
"""

def retrieve_from_store(query, vector_store, k=5):
    """
    基于查询从向量存储中检索相关项目。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 要搜索的向量存储
        k (int): 要检索的结果数量
        
    返回:
        List[Dict]: 带有分数和元数据的检索项目
    """
    # 创建查询嵌入
    query_embedding = create_embeddings(query)
    
    # 在向量存储中搜索前k个最相似的项目
    results = vector_store.similarity_search(query_embedding, k=k)
    
    return results


def compare_retrieval_approaches(query, chunk_store, prop_store, k=5):
    """
    比较基于块和基于命题的检索方法。
    
    参数:
        query (str): 用户查询
        chunk_store (SimpleVectorStore): 基于块的向量存储
        prop_store (SimpleVectorStore): 基于命题的向量存储
        k (int): 从每个存储中检索的结果数量
        
    返回:
        Dict: 比较结果
    """
    print(f"\n=== Query: {query} ===")
    
    # 从基于命题的向量存储中检索结果
    print("\nRetrieving with proposition-based approach...")
    prop_results = retrieve_from_store(query, prop_store, k)
    
    # 从基于块的向量存储中检索结果
    print("Retrieving with chunk-based approach...")
    chunk_results = retrieve_from_store(query, chunk_store, k)
    
    # 显示基于命题的结果
    print("\n=== Proposition-Based Results ===")
    for i, result in enumerate(prop_results):
        print(f"{i+1}) {result['text']} (Score: {result['similarity']:.4f})")
    
    # 显示基于块的结果
    print("\n=== Chunk-Based Results ===")
    for i, result in enumerate(chunk_results):
        # 截断文本以保持输出可管理
        truncated_text = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
        print(f"{i+1}) {truncated_text} (Score: {result['similarity']:.4f})")
    
    # 返回比较结果
    return {
        "query": query,
        "proposition_results": prop_results,
        "chunk_results": chunk_results
    }


"""
## 响应生成和评估
"""

def generate_response(query, results, result_type="proposition"):
    """
    基于检索结果生成响应。
    
    参数:
        query (str): 用户查询
        results (List[Dict]): 检索项目
        result_type (str): 结果类型（'proposition'或'chunk'）
        
    返回:
        str: 生成的响应
    """
    # 将检索的文本组合成单个上下文字符串
    context = "\n\n".join([result["text"] for result in results])
    
    # 系统提示，指导AI如何生成响应
    system_prompt = f"""You are an AI assistant answering questions based on retrieved information.
Your answer should be based on the following {result_type}s that were retrieved from a knowledge base.
If the retrieved information doesn't answer the question, acknowledge this limitation."""

    # 用户提示，包含查询和检索的上下文
    user_prompt = f"""Query: {query}

Retrieved {result_type}s:
{context}

Please answer the query based on the retrieved information."""

    # 使用OpenAI客户端生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    # 返回生成的响应文本
    return response.choices[0].message.content


def evaluate_responses(query, prop_response, chunk_response, reference_answer=None):
    """
    评估和比较两种方法的响应。
    
    参数:
        query (str): 用户查询
        prop_response (str): 基于命题的方法的响应
        chunk_response (str): 基于块的方法的响应
        reference_answer (str, optional): 用于比较的参考答案
        
    返回:
        str: 评估分析
    """
    # 系统提示，指导AI如何评估响应
    system_prompt = """You are an expert evaluator of information retrieval systems. 
    Compare the two responses to the same query, one generated from proposition-based retrieval 
    and the other from chunk-based retrieval.

    Evaluate them based on:
    1. Accuracy: Which response provides more factually correct information?
    2. Relevance: Which response better addresses the specific query?
    3. Conciseness: Which response is more concise while maintaining completeness?
    4. Clarity: Which response is easier to understand?

    Be specific about the strengths and weaknesses of each approach."""

    # 用户提示，包含查询和要比较的响应
    user_prompt = f"""Query: {query}

    Response from Proposition-Based Retrieval:
    {prop_response}

    Response from Chunk-Based Retrieval:
    {chunk_response}"""

    # 如果提供了参考答案，则将其包含在用户提示中进行事实检查
    if reference_answer:
        user_prompt += f"""

    Reference Answer (for factual checking):
    {reference_answer}"""

    # 向用户提示添加最终指令
    user_prompt += """
    Please provide a detailed comparison of these two responses, highlighting which approach performed better and why."""

    # 使用OpenAI客户端生成评估分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 返回生成的评估分析
    return response.choices[0].message.content


"""
## 完整端到端评估流程
"""

def run_proposition_chunking_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    运行命题分块与标准分块的完整评估。
    
    参数:
        pdf_path (str): PDF文件的路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 查询的参考答案
        
    返回:
        Dict: 评估结果
    """
    print("=== 开始命题分块评估 ===\n")
    
    # 将文档处理成命题和块
    chunks, propositions = process_document_into_propositions(pdf_path)
    
    # 为块和命题构建向量存储
    chunk_store, prop_store = build_vector_stores(chunks, propositions)
    
    # 初始化一个列表来存储每个查询的结果
    results = []
    
    # 为每个查询运行测试
    for i, query in enumerate(test_queries):
        print(f"\n\n=== 测试查询 {i+1}/{len(test_queries)} ===")
        print(f"查询: {query}")
        
        # 从基于块和基于命题的方法获取检索结果
        retrieval_results = compare_retrieval_approaches(query, chunk_store, prop_store)
        
        # 基于检索的基于命题的结果生成响应
        print("\n生成基于命题结果的响应...")
        prop_response = generate_response(
            query, 
            retrieval_results["proposition_results"], 
            "proposition"
        )
        
        # 基于检索的基于块的结果生成响应
        print("生成基于块结果的响应...")
        chunk_response = generate_response(
            query, 
            retrieval_results["chunk_results"], 
            "chunk"
        )
        
        # 获取参考答案（如果有）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        
        # 评估生成的响应
        print("\n评估响应...")
        evaluation = evaluate_responses(query, prop_response, chunk_response, reference)
        
        # 编译当前查询的结果
        query_result = {
            "query": query,
            "proposition_results": retrieval_results["proposition_results"],
            "chunk_results": retrieval_results["chunk_results"],
            "proposition_response": prop_response,
            "chunk_response": chunk_response,
            "reference_answer": reference,
            "evaluation": evaluation
        }
        
        # 将结果附加到总体结果列表
        results.append(query_result)
        
        # 打印当前查询的响应和评估
        print("\n=== 基于命题的响应 ===")
        print(prop_response)
        
        print("\n=== 基于块的响应 ===")
        print(chunk_response)
        
        print("\n=== 评估 ===")
        print(evaluation)
    
    # 生成评估的总体分析
    print("\n\n=== 生成总体分析 ===")
    overall_analysis = generate_overall_analysis(results)
    print("\n" + overall_analysis)
    
    # 返回评估结果、总体分析以及命题和块的数量
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "proposition_count": len(propositions),
        "chunk_count": len(chunks)
    }


def generate_overall_analysis(results):
    """
    生成命题与块方法的总体分析。
    
    参数:
        results (List[Dict]): 每个测试查询的结果
        
    返回:
        str: 总体分析
    """
    # 系统提示，指导AI如何生成总体分析
    system_prompt = """You are an expert at evaluating information retrieval systems.
    Based on multiple test queries, provide an overall analysis comparing proposition-based retrieval 
    to chunk-based retrieval for RAG (Retrieval-Augmented Generation) systems.

    Focus on:
    1. When proposition-based retrieval performs better
    2. When chunk-based retrieval performs better
    3. The overall strengths and weaknesses of each approach
    4. Recommendations for when to use each approach"""

    # 创建每个查询评估的摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Evaluation Summary: {result['evaluation'][:200]}...\n\n"

    # 用户提示，包含评估摘要
    user_prompt = f"""Based on the following evaluations of proposition-based vs chunk-based retrieval across {len(results)} queries, 
    provide an overall analysis comparing these two approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis on the relative strengths and weaknesses of proposition-based 
    and chunk-based retrieval for RAG systems."""

    # 使用OpenAI客户端生成总体分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 返回生成的分析文本
    return response.choices[0].message.content


"""
## 命题分块评估
"""

# 将要处理的AI信息文档的路径
pdf_path = "data/AI_Information.pdf"

# 定义测试查询，涵盖AI的不同方面，以评估命题分块
test_queries = [
    "What are the main ethical concerns in AI development?",
    # "How does explainable AI improve trust in AI systems?",
    # "What are the key challenges in developing fair AI systems?",
    # "What role does human oversight play in AI safety?"
]

# 参考答案，用于更全面地评估和比较结果
# 这些提供了一个基准真相，用于衡量生成的响应的质量
reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
    # "Explainable AI improves trust by making AI decision-making processes transparent and understandable to users, helping them verify fairness, identify potential biases, and better understand AI limitations.",
    # "Key challenges in developing fair AI systems include addressing data bias, ensuring diverse representation in training data, creating transparent algorithms, defining fairness across different contexts, and balancing competing fairness criteria.",
    # "Human oversight plays a critical role in AI safety by monitoring system behavior, verifying outputs, intervening when necessary, setting ethical boundaries, and ensuring AI systems remain aligned with human values and intentions throughout their operation."
]

# 运行评估
evaluation_results = run_proposition_chunking_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# 打印总体分析
print("\n\n=== Overall Analysis ===")
print(evaluation_results["overall_analysis"])

