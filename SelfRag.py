#!/usr/bin/env python
# coding: utf-8

"""
# Self-RAG: 一种动态RAG方法

在本代码中，我实现了Self-RAG，这是一种高级RAG系统，能够动态决定何时以及如何使用检索到的信息。与传统的RAG方法不同，Self-RAG在检索和生成过程中引入了反思点，从而产生更高质量和更可靠的响应。

## Self-RAG的关键组件

1. **检索决策**：确定是否需要为给定查询进行检索
2. **文档检索**：在需要时获取潜在相关文档
3. **相关性评估**：评估每个检索文档的相关性
4. **响应生成**：基于相关上下文创建响应
5. **支持评估**：评估响应是否适当地基于上下文
6. **实用性评估**：评价生成响应的整体有用性
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
要实现RAG，我们首先需要文本数据源。在这种情况下，我们使用PyMuPDF库从PDF文件中提取文本。
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
        text = page.get_text("text")  # 从页面提取文本
        all_text += text  # 将提取的文本附加到all_text字符串中

    return all_text  # 返回提取的文本

"""
## 对提取的文本进行分块
一旦我们提取了文本，我们将其分成更小的、重叠的块，以提高检索准确性。
"""

def chunk_text(text, n, overlap):
    """
    将给定文本分成具有重叠的n个字符的段落。

    参数:
    text (str): 要分块的文本。
    n (int): 每个块中的字符数。
    overlap (int): 块之间的重叠字符数。

    返回:
    List[str]: 文本块列表。
    """
    chunks = []  # 初始化一个空列表来存储块
    
    # 以步长(n - overlap)遍历文本
    for i in range(0, len(text), n - overlap):
        # 将从索引i到i + n的文本块附加到chunks列表中
        chunks.append(text[i:i + n])

    return chunks  # 返回文本块列表

"""
## 设置OpenAI API客户端
我们初始化OpenAI客户端以生成嵌入和响应。
"""

# 使用基础URL和API密钥初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中检索API密钥
)

"""
## 简单向量存储实现
我们将创建一个基本的向量存储来管理文档块及其嵌入。
"""

class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储。
    """
    def __init__(self):
        """
        初始化向量存储。
        """
        self.vectors = []  # 存储嵌入向量的列表
        self.texts = []  # 存储原始文本的列表
        self.metadata = []  # 存储每个文本的元数据的列表
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储添加项目。

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
        filter_func (callable, optional): 过滤结果的函数。

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
            # 如果提供了过滤器，则应用它
            if filter_func and not filter_func(self.metadata[i]):
                continue
                
            # 计算余弦相似度
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 附加索引和相似度分数
        
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
    text (str or List[str]): 要创建嵌入的输入文本。
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
    处理Self-RAG的文档。

    参数:
        pdf_path (str): PDF文件的路径。
        chunk_size (int): 每个块的字符大小。
        chunk_overlap (int): 块之间的字符重叠。

    返回:
        SimpleVectorStore: 包含文档块及其嵌入的向量存储。
    """
    # 从PDF文件中提取文本
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # 对提取的文本进行分块
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")
    
    # 为每个块创建嵌入
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)
    
    # 初始化向量存储
    store = SimpleVectorStore()
    
    # 将每个块及其嵌入添加到向量存储中
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    
    print(f"Added {len(chunks)} chunks to the vector store")
    return store

"""
## Self-RAG组件
### 1. 检索决策
"""

def determine_if_retrieval_needed(query):
    """
    确定是否需要为给定查询进行检索。
    
    参数:
        query (str): 用户查询
        
    返回:
        bool: 如果需要检索则为True，否则为False
    """
    # 系统提示，指导AI如何确定是否需要检索
    system_prompt = """You are an AI assistant that determines if retrieval is necessary to answer a query.
    For factual questions, specific information requests, or questions about events, people, or concepts, answer "Yes".
    For opinions, hypothetical scenarios, or simple queries with common knowledge, answer "No".
    Answer with ONLY "Yes" or "No"."""

    # 包含查询的用户提示
    user_prompt = f"Query: {query}\n\nIs retrieval necessary to answer this query accurately?"
    
    # 从模型生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从模型的响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()
    
    # 如果答案包含"yes"则返回True，否则返回False
    return "yes" in answer

"""
### 2. 相关性评估
"""

def evaluate_relevance(query, context):
    """
    评估上下文与查询的相关性。
    
    参数:
        query (str): 用户查询
        context (str): 上下文文本
        
    返回:
        str: 'relevant'或'irrelevant'
    """
    # 系统提示，指导AI如何确定文档相关性
    system_prompt = """You are an AI assistant that determines if a document is relevant to a query.
    Consider whether the document contains information that would be helpful in answering the query.
    Answer with ONLY "Relevant" or "Irrelevant"."""

    # 如果上下文太长，则截断以避免超出令牌限制
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # 包含查询和文档内容的用户提示
    user_prompt = f"""Query: {query}
    Document content:
    {context}

    Is this document relevant to the query? Answer with ONLY "Relevant" or "Irrelevant".
    """
    
    # 从模型生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从模型的响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()
    
    return answer  # 返回相关性评估

"""
### 3. 支持评估
"""

def assess_support(response, context):
    """
    评估响应被上下文支持的程度。
    
    参数:
        response (str): 生成的响应
        context (str): 上下文文本
        
    返回:
        str: 'fully supported', 'partially supported', 或 'no support'
    """
    # 系统提示，指导AI如何评估支持
    system_prompt = """You are an AI assistant that determines if a response is supported by the given context.
    Evaluate if the facts, claims, and information in the response are backed by the context.
    Answer with ONLY one of these three options:
    - "Fully supported": All information in the response is directly supported by the context.
    - "Partially supported": Some information in the response is supported by the context, but some is not.
    - "No support": The response contains significant information not found in or contradicting the context.
    """

    # 如果上下文太长，则截断以避免超出令牌限制
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # 包含上下文和要评估的响应的用户提示
    user_prompt = f"""Context:
    {context}

    Response:
    {response}

    How well is this response supported by the context? Answer with ONLY "Fully supported", "Partially supported", or "No support".
    """
    
    # 从模型生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从模型的响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()
    
    return answer  # 返回支持评估

"""
### 4. 实用性评估
"""

def rate_utility(query, response):
    """
    评价响应对查询的实用性。
    
    参数:
        query (str): 用户查询
        response (str): 生成的响应
        
    返回:
        int: 1到5的实用性评分
    """
    # 系统提示，指导AI如何评价响应的实用性
    system_prompt = """You are an AI assistant that rates the utility of a response to a query.
    Consider how well the response answers the query, its completeness, correctness, and helpfulness.
    Rate the utility on a scale from 1 to 5, where:
    - 1: Not useful at all
    - 2: Slightly useful
    - 3: Moderately useful
    - 4: Very useful
    - 5: Exceptionally useful
    Answer with ONLY a single number from 1 to 5."""

    # 包含要评分的查询和响应的用户提示
    user_prompt = f"""Query: {query}
    Response:
    {response}

    Rate the utility of this response on a scale from 1 to 5:"""
    
    # 使用OpenAI客户端生成实用性评分
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从模型的响应中提取评分
    rating = response.choices[0].message.content.strip()
    
    # 从评分中仅提取数字
    rating_match = re.search(r'[1-5]', rating)
    if rating_match:
        return int(rating_match.group())  # 将提取的评分作为整数返回
    
    return 3  # 如果解析失败，默认为中等评分

"""
## 响应生成
"""

def generate_response(query, context=None):
    """
    基于查询和可选上下文生成响应。
    
    参数:
        query (str): 用户查询
        context (str, optional): 上下文文本
        
    返回:
        str: 生成的响应
    """
    # 系统提示，指导AI如何生成有用的响应
    system_prompt = """You are a helpful AI assistant. Provide a clear, accurate, and informative response to the query."""
    
    # 根据是否提供上下文创建用户提示
    if context:
        user_prompt = f"""Context:
        {context}

        Query: {query}

        Please answer the query based on the provided context.
        """
    else:
        user_prompt = f"""Query: {query}
        
        Please answer the query to the best of your ability."""
    
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
    return response.choices[0].message.content.strip()

"""
## 完整的Self-RAG实现
"""

def self_rag(query, vector_store, top_k=3):
    """
    实现完整的Self-RAG流程。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        top_k (int): 初始检索的文档数量
        
    返回:
        dict: 包括查询、响应和Self-RAG过程指标的结果
    """
    print(f"\n=== Starting Self-RAG for query: {query} ===\n")
    
    # 步骤1：确定是否需要检索
    print("Step 1: Determining if retrieval is necessary...")
    retrieval_needed = determine_if_retrieval_needed(query)
    print(f"Retrieval needed: {retrieval_needed}")
    
    # 初始化指标以跟踪Self-RAG过程
    metrics = {
        "retrieval_needed": retrieval_needed,
        "documents_retrieved": 0,
        "relevant_documents": 0,
        "response_support_ratings": [],
        "utility_ratings": []
    }
    
    best_response = None
    best_score = -1
    
    if retrieval_needed:
        # 步骤2：检索文档
        print("\nStep 2: Retrieving relevant documents...")
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
        metrics["documents_retrieved"] = len(results)
        print(f"Retrieved {len(results)} documents")
        
        # 步骤3：评估每个文档的相关性
        print("\nStep 3: Evaluating document relevance...")
        relevant_contexts = []
        
        for i, result in enumerate(results):
            context = result["text"]
            relevance = evaluate_relevance(query, context)
            print(f"Document {i+1} relevance: {relevance}")
            
            if relevance == "relevant":
                relevant_contexts.append(context)
        
        metrics["relevant_documents"] = len(relevant_contexts)
        print(f"Found {len(relevant_contexts)} relevant documents")
        
        if relevant_contexts:
            # 步骤4：处理每个相关上下文
            print("\nStep 4: Processing relevant contexts...")
            for i, context in enumerate(relevant_contexts):
                print(f"\nProcessing context {i+1}/{len(relevant_contexts)}...")
                
                # 基于上下文生成响应
                print("Generating response...")
                response = generate_response(query, context)
                
                # 评估响应被上下文支持的程度
                print("Assessing support...")
                support_rating = assess_support(response, context)
                print(f"Support rating: {support_rating}")
                metrics["response_support_ratings"].append(support_rating)
                
                # 评价响应的实用性
                print("Rating utility...")
                utility_rating = rate_utility(query, response)
                print(f"Utility rating: {utility_rating}/5")
                metrics["utility_ratings"].append(utility_rating)
                
                # 计算总体分数（更好的支持和实用性得分更高）
                support_score = {
                    "fully supported": 3, 
                    "partially supported": 1, 
                    "no support": 0
                }.get(support_rating, 0)
                
                overall_score = support_score * 5 + utility_rating
                print(f"Overall score: {overall_score}")
                
                # 跟踪最佳响应
                if overall_score > best_score:
                    best_response = response
                    best_score = overall_score
                    print("New best response found!")
        
        # 如果没有找到相关上下文或所有响应得分较低
        if not relevant_contexts or best_score <= 0:
            print("\nNo suitable context found or poor responses, generating without retrieval...")
            best_response = generate_response(query)
    else:
        # 不需要检索，直接生成
        print("\nNo retrieval needed, generating response directly...")
        best_response = generate_response(query)
    
    # 最终指标
    metrics["best_score"] = best_score
    metrics["used_retrieval"] = retrieval_needed and best_score > 0
    
    print("\n=== Self-RAG Completed ===")
    
    return {
        "query": query,
        "response": best_response,
        "metrics": metrics
    }

"""
## 运行完整的Self-RAG系统
"""

def run_self_rag_example():
    """
    用示例演示完整的Self-RAG系统。
    """
    # 处理文档
    pdf_path = "data/AI_Information.pdf"  # PDF文档的路径
    print(f"Processing document: {pdf_path}")
    vector_store = process_document(pdf_path)  # 处理文档并创建向量存储
    
    # 示例1：可能需要检索的查询
    query1 = "What are the main ethical concerns in AI development?"
    print("\n" + "="*80)
    print(f"EXAMPLE 1: {query1}")
    result1 = self_rag(query1, vector_store)  # 为第一个查询运行Self-RAG
    print("\nFinal response:")
    print(result1["response"])  # 打印第一个查询的最终响应
    print("\nMetrics:")
    print(json.dumps(result1["metrics"], indent=2))  # 打印第一个查询的指标
    
    # 示例2：可能不需要检索的查询
    query2 = "Can you write a short poem about artificial intelligence?"
    print("\n" + "="*80)
    print(f"EXAMPLE 2: {query2}")
    result2 = self_rag(query2, vector_store)  # 为第二个查询运行Self-RAG
    print("\nFinal response:")
    print(result2["response"])  # 打印第二个查询的最终响应
    print("\nMetrics:")
    print(json.dumps(result2["metrics"], indent=2))  # 打印第二个查询的指标
    
    # 示例3：与文档有一定相关性但需要额外知识的查询
    query3 = "How might AI impact economic growth in developing countries?"
    print("\n" + "="*80)
    print(f"EXAMPLE 3: {query3}")
    result3 = self_rag(query3, vector_store)  # 为第三个查询运行Self-RAG
    print("\nFinal response:")
    print(result3["response"])  # 打印第三个查询的最终响应
    print("\nMetrics:")
    print(json.dumps(result3["metrics"], indent=2))  # 打印第三个查询的指标
    
    return {
        "example1": result1,
        "example2": result2,
        "example3": result3
    }

"""
## 评估Self-RAG与传统RAG的对比
"""

def traditional_rag(query, vector_store, top_k=3):
    """
    实现传统RAG方法进行比较。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        top_k (int): 要检索的文档数量
        
    返回:
        str: 生成的响应
    """
    print(f"\n=== Running traditional RAG for query: {query} ===\n")
    
    # 检索文档
    print("Retrieving documents...")
    query_embedding = create_embeddings(query)  # 为查询创建嵌入
    results = vector_store.similarity_search(query_embedding, k=top_k)  # 搜索相似文档
    print(f"Retrieved {len(results)} documents")
    
    # 合并检索文档的上下文
    contexts = [result["text"] for result in results]  # 从结果中提取文本
    combined_context = "\n\n".join(contexts)  # 将文本合并为单个上下文
    
    # 使用合并的上下文生成响应
    print("Generating response...")
    response = generate_response(query, combined_context)  # 基于合并的上下文生成响应
    
    return response




def evaluate_rag_approaches(pdf_path, test_queries, reference_answers=None):
    """
    比较Self-RAG与传统RAG。
    
    参数:
        pdf_path (str): 文档路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 用于评估的参考答案
        
    返回:
        dict: 评估结果
    """
    print("=== Evaluating RAG Approaches ===")
    
    # 处理文档以创建向量存储
    vector_store = process_document(pdf_path)
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\nProcessing query {i+1}: {query}")
        
        # 运行Self-RAG
        self_rag_result = self_rag(query, vector_store)  # 从Self-RAG获取响应
        self_rag_response = self_rag_result["response"]
        
        # 运行传统RAG
        trad_rag_response = traditional_rag(query, vector_store)  # 从传统RAG获取响应
        
        # 如果有参考答案，则比较结果
        reference = reference_answers[i] if reference_answers and i < len(reference_answers) else None
        comparison = compare_responses(query, self_rag_response, trad_rag_response, reference)  # 比较响应
        
        results.append({
            "query": query,
            "self_rag_response": self_rag_response,
            "traditional_rag_response": trad_rag_response,
            "reference_answer": reference,
            "comparison": comparison,
            "self_rag_metrics": self_rag_result["metrics"]
        })
    
    # 生成整体分析
    overall_analysis = generate_overall_analysis(results)
    
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }




def compare_responses(query, self_rag_response, trad_rag_response, reference=None):
    """
    比较Self-RAG和传统RAG的响应。
    
    参数:
        query (str): 用户查询
        self_rag_response (str): Self-RAG的响应
        trad_rag_response (str): 传统RAG的响应
        reference (str, optional): 参考答案
        
    返回:
        str: 比较分析
    """
    system_prompt = """You are an expert evaluator of RAG systems. Your task is to compare responses from two different RAG approaches:
1. Self-RAG: A dynamic approach that decides if retrieval is needed and evaluates information relevance and response quality
2. Traditional RAG: Always retrieves documents and uses them to generate a response

Compare the responses based on:
- Relevance to the query
- Factual correctness
- Completeness and informativeness
- Conciseness and focus"""

    user_prompt = f"""Query: {query}

Response from Self-RAG:
{self_rag_response}

Response from Traditional RAG:
{trad_rag_response}
"""

    if reference:
        user_prompt += f"""
Reference Answer (for factual checking):
{reference}
"""

    user_prompt += """
Compare these responses and explain which one is better and why.
Focus on accuracy, relevance, completeness, and quality.
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 使用更强大的模型进行评估
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content


def generate_overall_analysis(results):
    """
    生成Self-RAG与传统RAG的整体分析。
    
    参数:
        results (List[Dict]): evaluate_rag_approaches的结果
        
    返回:
        str: 整体分析
    """
    system_prompt = """You are an expert evaluator of RAG systems. Your task is to provide an overall analysis comparing
    Self-RAG and Traditional RAG based on multiple test queries.

    Focus your analysis on:
    1. When Self-RAG performs better and why
    2. When Traditional RAG performs better and why
    3. The impact of dynamic retrieval decisions in Self-RAG
    4. The value of relevance and support evaluation in Self-RAG
    5. Overall recommendations on which approach to use for different types of queries"""

    # 准备个别比较的摘要
    comparisons_summary = ""
    for i, result in enumerate(results):
        comparisons_summary += f"Query {i+1}: {result['query']}\n"
        comparisons_summary += f"Self-RAG metrics: Retrieval needed: {result['self_rag_metrics']['retrieval_needed']}, "
        comparisons_summary += f"Relevant docs: {result['self_rag_metrics']['relevant_documents']}/{result['self_rag_metrics']['documents_retrieved']}\n"
        comparisons_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    user_prompt = f"""Based on the following comparison results from {len(results)} test queries, please provide an overall analysis of
Self-RAG versus Traditional RAG:

{comparisons_summary}

Please provide your comprehensive analysis.
"""

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
## 评估Self-RAG系统

最后一步是评估Self-RAG系统与传统RAG方法的对比。我们将比较两个系统生成的响应质量，并分析Self-RAG在不同场景下的表现。
"""

# AI信息文档的路径
pdf_path = "data/AI_Information.pdf"

# 定义测试查询，涵盖不同查询类型以测试Self-RAG的自适应检索
test_queries = [
    "What are the main ethical concerns in AI development?",        # 文档聚焦查询
    # "How does explainable AI improve trust in AI systems?",         # 文档聚焦查询
    # "Write a poem about artificial intelligence",                   # 创意查询，不需要检索
    # "Will superintelligent AI lead to human obsolescence?"          # 推测性查询，需要部分检索
]

# 更客观评估的参考答案
reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
    # "Explainable AI improves trust by making AI decision-making processes transparent and understandable to users, helping them verify fairness, identify potential biases, and better understand AI limitations.",
    # "A quality poem about artificial intelligence should creatively explore themes of AI's capabilities, limitations, relationship with humanity, potential futures, or philosophical questions about consciousness and intelligence.",
    # "Views on superintelligent AI's impact on human relevance vary widely. Some experts warn of potential risks if AI surpasses human capabilities across domains, possibly leading to economic displacement or loss of human agency. Others argue humans will remain relevant through complementary skills, emotional intelligence, and by defining AI's purpose. Most experts agree that thoughtful governance and human-centered design are essential regardless of the outcome."
]

# 运行评估，比较Self-RAG与传统RAG方法
evaluation_results = evaluate_rag_approaches(
    pdf_path=pdf_path,                  # 包含AI信息的源文档
    test_queries=test_queries,          # AI相关测试查询列表
    reference_answers=reference_answers  # 用于评估的标准答案
)

# 打印整体比较分析
print("\n=== OVERALL ANALYSIS ===\n")
print(evaluation_results["overall_analysis"])