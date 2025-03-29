#!/usr/bin/env python
# coding: utf-8

"""
# 上下文压缩：让RAG系统更上一层楼！
# 在这个脚本中，我们将实现一种上下文压缩技术，让我们的RAG系统更高效。
# 通过过滤和压缩检索到的文本块，只保留最相关的部分，减少噪音，提高响应质量。
#
# 在RAG中检索文档时，我们经常会得到包含相关和不相关信息的文本块。
# 上下文压缩技术可以帮助我们：
#
# - 删除不相关的句子和段落
# - 只关注与查询相关的信息
# - 最大化上下文窗口中的有用信息
#
# 让我们从零开始实现这个神奇的功能吧！
"""

"""
## 环境搭建
# 首先，让我们导入必要的库。
"""

import fitz
import os
import numpy as np
import json
from openai import OpenAI

"""
## 从PDF文件中提取文本
# 要实现RAG，我们首先需要文本数据源。这里我们使用PyMuPDF库从PDF文件中提取文本。
"""

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并打印前`num_chars`个字符。

    参数：
    pdf_path (str): PDF文件路径。

    返回：
    str: 从PDF中提取的文本。
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化空字符串来存储提取的文本

    # 遍历PDF的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取页面
        text = page.get_text("text")  # 从页面提取文本
        all_text += text  # 将提取的文本追加到all_text字符串中

    return all_text  # 返回提取的文本

"""
## 对提取的文本进行分块
# 提取文本后，我们将其分成较小的、有重叠的块，以提高检索准确性。
"""

def chunk_text(text, n=1000, overlap=200):
    """
    将给定文本分成n个字符的段，并带有重叠。

    参数：
    text (str): 要分块的文本。
    n (int): 每个块中的字符数。
    overlap (int): 块之间的重叠字符数。

    返回：
    List[str]: 文本块列表。
    """
    chunks = []  # 初始化空列表来存储块
    
    # 以(n - overlap)为步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 将索引i到i + n的文本块追加到chunks列表中
        chunks.append(text[i:i + n])

    return chunks  # 返回文本块列表

"""
## 设置OpenAI API客户端
# 我们初始化OpenAI客户端来生成嵌入和响应。
"""

# 使用基础URL和API密钥初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key= os.environ.get("OPENAI_API_KEY") # 使用你的OpenAI API密钥
)

"""
## 构建简单的向量存储
# 由于无法使用FAISS，我们实现一个简单的向量存储。
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
        self.metadata = []  # 存储每个文本元数据的列表
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加项目。

        参数：
        text (str): 原始文本。
        embedding (List[float]): 嵌入向量。
        metadata (dict, 可选): 附加元数据。
        """
        self.vectors.append(np.array(embedding))  # 将嵌入转换为numpy数组并添加到vectors列表
        self.texts.append(text)  # 将原始文本添加到texts列表
        self.metadata.append(metadata or {})  # 将元数据添加到metadata列表，如果为None则使用空字典
    
    def similarity_search(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目。

        参数：
        query_embedding (List[float]): 查询嵌入向量。
        k (int): 返回的结果数量。

        返回：
        List[Dict]: 包含文本和元数据的top k最相似项目。
        """
        if not self.vectors:
            return []  # 如果没有存储向量，返回空列表
        
        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)
        
        # 使用余弦相似度计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 追加索引和相似度分数
        
        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top k结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 添加索引对应的文本
                "metadata": self.metadata[idx],  # 添加索引对应的元数据
                "similarity": score  # 添加相似度分数
            })
        
        return results  # 返回top k结果列表

"""
## 嵌入生成
"""

def create_embeddings(text,  model="BAAI/bge-en-icl"):
    """
    为给定文本创建嵌入。

    参数：
    text (str or List[str]): 要创建嵌入的输入文本。
    model (str): 用于创建嵌入的模型。

    返回：
    List[float] or List[List[float]]: 嵌入向量。
    """
    # 通过确保input_text始终是列表来处理字符串和列表输入
    input_text = text if isinstance(text, list) else [text]
    
    # 使用指定模型为输入文本创建嵌入
    response = client.embeddings.create(
        model=model,
        input=input_text
    )
    
    # 如果输入是单个字符串，只返回第一个嵌入
    if isinstance(text, str):
        return response.data[0].embedding
    
    # 否则，返回所有输入文本的嵌入
    return [item.embedding for item in response.data]

"""
## 构建文档处理管道
"""

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    处理RAG的文档。

    参数：
    pdf_path (str): PDF文件路径。
    chunk_size (int): 每个块的字符数。
    chunk_overlap (int): 块之间的重叠字符数。

    返回：
    SimpleVectorStore: 包含文档块及其嵌入的向量存储。
    """
    # 从PDF文件中提取文本
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # 将提取的文本分成较小的段
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")
    
    # 为每个文本块创建嵌入
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)
    
    # 初始化简单向量存储来存储块及其嵌入
    store = SimpleVectorStore()
    
    # 将每个块及其对应的嵌入添加到向量存储中
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    
    print(f"Added {len(chunks)} chunks to the vector store")
    return store

"""
## 实现上下文压缩
# 这是我们方法的核心 - 我们将使用LLM来过滤和压缩检索到的内容。
"""

def compress_chunk(chunk, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    通过保留与查询相关的部分来压缩检索到的块。
    
    参数：
        chunk (str): 要压缩的文本块
        query (str): 用户查询
        compression_type (str): 压缩类型（"selective", "summary", 或 "extraction"）
        model (str): 使用的LLM模型
        
    返回：
        str: 压缩后的块
    """
    # 为不同的压缩方法定义系统提示
    if compression_type == "selective":
        system_prompt = """You are an expert at information filtering. 
        Your task is to analyze a document chunk and extract ONLY the sentences or paragraphs that are directly 
        relevant to the user's query. Remove all irrelevant content.

        Your output should:
        1. ONLY include text that helps answer the query
        2. Preserve the exact wording of relevant sentences (do not paraphrase)
        3. Maintain the original order of the text
        4. Include ALL relevant content, even if it seems redundant
        5. EXCLUDE any text that isn't relevant to the query

        Format your response as plain text with no additional comments."""
    elif compression_type == "summary":
        system_prompt = """You are an expert at summarization. 
        Your task is to create a concise summary of the provided chunk that focuses ONLY on 
        information relevant to the user's query.

        Your output should:
        1. Be brief but comprehensive regarding query-relevant information
        2. Focus exclusively on information related to the query
        3. Omit irrelevant details
        4. Be written in a neutral, factual tone

        Format your response as plain text with no additional comments."""
    else:  # extraction
        system_prompt = """You are an expert at information extraction.
        Your task is to extract ONLY the exact sentences from the document chunk that contain information relevant 
        to answering the user's query.

        Your output should:
        1. Include ONLY direct quotes of relevant sentences from the original text
        2. Preserve the original wording (do not modify the text)
        3. Include ONLY sentences that directly relate to the query
        4. Separate extracted sentences with newlines
        5. Do not add any commentary or additional text

        Format your response as plain text with no additional comments."""

    # 使用查询和文档块定义用户提示
    user_prompt = f"""
        Query: {query}

        Document Chunk:
        {chunk}

        Extract only the content relevant to answering this query.
    """
    
    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 从响应中提取压缩后的块
    compressed_chunk = response.choices[0].message.content.strip()
    
    # 计算压缩率
    original_length = len(chunk)
    compressed_length = len(compressed_chunk)
    compression_ratio = (original_length - compressed_length) / original_length * 100
    
    return compressed_chunk, compression_ratio

"""
## 实现批量压缩
# 为了提高效率，我们将尽可能一次性压缩多个块。
"""

def batch_compress_chunks(chunks, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    单独压缩多个块。
    
    参数：
        chunks (List[str]): 要压缩的文本块列表
        query (str): 用户查询
        compression_type (str): 压缩类型（"selective", "summary", 或 "extraction"）
        model (str): 使用的LLM模型
        
    返回：
        List[Tuple[str, float]]: 压缩后的块及其压缩率的列表
    """
    print(f"Compressing {len(chunks)} chunks...")  # 打印要压缩的块数
    results = []  # 初始化空列表来存储结果
    total_original_length = 0  # 初始化变量来存储块的总原始长度
    total_compressed_length = 0  # 初始化变量来存储块的总压缩长度
    
    # 遍历每个块
    for i, chunk in enumerate(chunks):
        print(f"Compressing chunk {i+1}/{len(chunks)}...")  # 打印压缩进度
        # 压缩块并获取压缩后的块和压缩率
        compressed_chunk, compression_ratio = compress_chunk(chunk, query, compression_type, model)
        results.append((compressed_chunk, compression_ratio))  # 将结果追加到results列表中
        
        total_original_length += len(chunk)  # 将原始块的长度添加到总原始长度中
        total_compressed_length += len(compressed_chunk)  # 将压缩块的长度添加到总压缩长度中
    
    # 计算总体压缩率
    overall_ratio = (total_original_length - total_compressed_length) / total_original_length * 100
    print(f"Overall compression ratio: {overall_ratio:.2f}%")  # 打印总体压缩率
    
    return results  # 返回压缩后的块及其压缩率的列表

"""
## 响应生成函数
"""

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    根据查询和上下文生成响应。
    
    参数：
        query (str): 用户查询
        context (str): 来自压缩块的上下文文本
        model (str): 使用的LLM模型
        
    返回：
        str: 生成的响应
    """
    # 定义系统提示以指导AI的行为
    system_prompt = """You are a helpful AI assistant. Answer the user's question based only on the provided context.
    If you cannot find the answer in the context, state that you don't have enough information."""
            
    # 通过组合上下文和查询创建用户提示
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
    """
    
    # 使用OpenAI API生成响应
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

"""
## 带有上下文压缩的完整RAG管道
"""

def rag_with_compression(pdf_path, query, k=10, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    带有上下文压缩的完整RAG管道。
    
    参数：
        pdf_path (str): PDF文档路径
        query (str): 用户查询
        k (int): 初始检索的块数
        compression_type (str): 压缩类型
        model (str): 使用的LLM模型
        
    返回：
        dict: 包含查询、压缩块和响应的结果
    """
    print("\n=== RAG WITH CONTEXTUAL COMPRESSION ===")
    print(f"Query: {query}")
    print(f"Compression type: {compression_type}")
    
    # 处理文档以提取文本、分块并创建嵌入
    vector_store = process_document(pdf_path)
    
    # 为查询创建嵌入
    query_embedding = create_embeddings(query)
    
    # 根据查询嵌入检索top k最相似的块
    print(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]
    
    # 对检索到的块应用压缩
    compressed_results = batch_compress_chunks(retrieved_chunks, query, compression_type, model)
    compressed_chunks = [result[0] for result in compressed_results]
    compression_ratios = [result[1] for result in compressed_results]
    
    # 过滤掉任何空的压缩块
    filtered_chunks = [(chunk, ratio) for chunk, ratio in zip(compressed_chunks, compression_ratios) if chunk.strip()]
    
    if not filtered_chunks:
        # 如果所有块都被压缩为空字符串，使用原始块
        print("Warning: All chunks were compressed to empty strings. Using original chunks.")
        filtered_chunks = [(chunk, 0.0) for chunk in retrieved_chunks]
    else:
        compressed_chunks, compression_ratios = zip(*filtered_chunks)
    
    # 从压缩块生成上下文
    context = "\n\n---\n\n".join(compressed_chunks)
    
    # 基于压缩块生成响应
    print("Generating response based on compressed chunks...")
    response = generate_response(query, context, model)
    
    # 准备结果字典
    result = {
        "query": query,
        "original_chunks": retrieved_chunks,
        "compressed_chunks": compressed_chunks,
        "compression_ratios": compression_ratios,
        "context_length_reduction": f"{sum(compression_ratios)/len(compression_ratios):.2f}%",
        "response": response
    }
    
    print("\n=== RESPONSE ===")
    print(response)
    
    return result

"""
## 比较带压缩和不带压缩的RAG
# 让我们创建一个函数来比较标准RAG和我们的压缩增强版本：
"""

def standard_rag(pdf_path, query, k=10, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    不带压缩的标准RAG。
    
    参数：
        pdf_path (str): PDF文档路径
        query (str): 用户查询
        k (int): 检索的块数
        model (str): 使用的LLM模型
        
    返回：
        dict: 包含查询、块和响应的结果
    """
    print("\n=== STANDARD RAG ===")
    print(f"Query: {query}")
    
    # 处理文档以提取文本、分块并创建嵌入
    vector_store = process_document(pdf_path)
    
    # 创建一个嵌入
    query_embedding = create_embeddings(query)
    
    # 检索top k最相似的块
    print(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]
    
    # 生成上下文
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    # 生成响应
    print("Generating response...")
    response = generate_response(query, context, model)
    
    # 准备结果字典
    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }
    
    print("\n=== RESPONSE ===")
    print(response)
    
    return result


"""
## 评估我们的方法
# 现在，让我们实现一个函数来评估和比较不同的响应：
"""

def evaluate_responses(query, responses, reference_answer):
    """
    根据参考答案评估多个响应。

    参数：
        query (str): 用户查询
        responses (Dict[str, str]): 按方法分类的响应字典
        reference_answer (str): 参考答案

    返回：
        str: 评估文本
    """
    # Define the system prompt to guide the AI's behavior for evaluation
    system_prompt = """You are an objective evaluator of RAG responses. Compare different responses to the same query
    and determine which is most accurate, comprehensive, and relevant to the query."""
    
    # Create the user prompt by combining the query and reference answer
    user_prompt = f"""
    Query: {query}

    Reference Answer: {reference_answer}

    """

    # 将每个响应添加到提示中
    for method, response in responses.items():
        user_prompt += f"\n{method.capitalize()} Response:\n{response}\n"
    
    # Add the evaluation criteria to the user prompt
    user_prompt += """
    Please evaluate these responses based on:
    1. Factual accuracy compared to the reference
    2. Comprehensiveness - how completely they answer the query
    3. Conciseness - whether they avoid irrelevant information
    4. Overall quality

    Rank the responses from best to worst with detailed explanations.
    """

    # 使用OpenAI API生成评估响应
    evaluation_response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中返回评估文本
    return evaluation_response.choices[0].message.content


def evaluate_compression(pdf_path, query, reference_answer=None, compression_types=["selective", "summary", "extraction"]):
    """
    Compare different compression techniques with standard RAG.
    
    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        reference_answer (str): Optional reference answer
        compression_types (List[str]): Compression types to evaluate
        
    Returns:
        dict: Evaluation results
    """
    print("\n=== EVALUATING CONTEXTUAL COMPRESSION ===")
    print(f"Query: {query}")
    
    # 比较不同的压缩技术与标准RAG。
    standard_result = standard_rag(pdf_path, query)
    
    # 存储不同压缩技术的结果
    compression_results = {}
    
    # 使用每种压缩技术运行RAG
    for comp_type in compression_types:
        print(f"\nTesting {comp_type} compression...")
        compression_results[comp_type] = rag_with_compression(pdf_path, query, compression_type=comp_type)
    
    # 准备所有响应
    responses = {
        "standard": standard_result["response"]
    }
    for comp_type in compression_types:
        responses[comp_type] = compression_results[comp_type]["response"]
    
    # 评估所有响应
    if reference_answer:
        evaluation = evaluate_responses(query, responses, reference_answer)
        print("\n=== EVALUATION RESULTS ===")
        print(evaluation)
    else:
        evaluation = "No reference answer provided for evaluation."
    
    # 计算每个压缩技术的平均压缩率和总上下文长度
    metrics = {}
    for comp_type in compression_types:
        metrics[comp_type] = {
            "avg_compression_ratio": f"{sum(compression_results[comp_type]['compression_ratios'])/len(compression_results[comp_type]['compression_ratios']):.2f}%",
            "total_context_length": len("\n\n".join(compression_results[comp_type]['compressed_chunks'])),
            "original_context_length": len("\n\n".join(standard_result['chunks']))
        }
    
    # Return the evaluation results, responses, and metrics
    return {
        "query": query,
        "responses": responses,
        "evaluation": evaluation,
        "metrics": metrics,
        "standard_result": standard_result,
        "compression_results": compression_results
    }


#  运行完整系统  
pdf_path = "data/AI_Information.pdf" 
query = "What are the ethical concerns surrounding the use of AI in decision-making?"  
reference_answer = """  
The use of AI in decision-making raises several ethical concerns.  
- Bias in AI models can lead to unfair or discriminatory outcomes, especially in critical areas like hiring, lending, and law enforcement.  
- Lack of transparency and explainability in AI-driven decisions makes it difficult for individuals to challenge unfair outcomes.  
- Privacy risks arise as AI systems process vast amounts of personal data, often without explicit consent.  
- The potential for job displacement due to automation raises social and economic concerns.  
- AI decision-making may also concentrate power in the hands of a few large tech companies, leading to accountability challenges.  
- Ensuring fairness, accountability, and transparency in AI systems is essential for ethical deployment.  
"""  

#使用不同的压缩技术进行评估。
# 压缩类型包括：
# 1) “选择性”压缩，保留关键细节并省略相对不重要的部分；
# 2) “摘要”压缩，提供信息的简洁版本；
# 3) “提取”压缩，从文档中逐字提取相关句子。

results = evaluate_compression(  
    pdf_path=pdf_path,  
    query=query,  
    reference_answer=reference_answer,  
    compression_types=["selective", "summary", "extraction"]  
)


"""
## 可视化压缩结果
# 让我们把压缩结果用更直观的方式展示出来，这样老板看了也会觉得咱们的工作很专业！
"""

def visualize_compression_results(evaluation_results):
    """
    可视化不同压缩技术的结果，让数据说话！
    
    参数：
        evaluation_results (Dict): 来自evaluate_compression函数的结果
    """
    # 从评估结果中提取查询和标准块
    query = evaluation_results["query"]
    standard_chunks = evaluation_results["standard_result"]["chunks"]
    
    # 打印查询
    print(f"Query: {query}")
    print("\n" + "="*80 + "\n")
    
    # 获取一个示例块进行可视化（使用第一个块）
    original_chunk = standard_chunks[0]
    
    # 遍历每种压缩类型并显示比较结果
    for comp_type in evaluation_results["compression_results"].keys():
        compressed_chunks = evaluation_results["compression_results"][comp_type]["compressed_chunks"]
        compression_ratios = evaluation_results["compression_results"][comp_type]["compression_ratios"]
        
        # 获取对应的压缩块及其压缩率
        compressed_chunk = compressed_chunks[0]
        compression_ratio = compression_ratios[0]
        
        print(f"\n=== {comp_type.upper()} COMPRESSION EXAMPLE ===\n")
        
        # 显示原始块（如果太长则截断）
        print("ORIGINAL CHUNK:")
        print("-" * 40)
        if len(original_chunk) > 800:
            print(original_chunk[:800] + "... [truncated]")
        else:
            print(original_chunk)
        print("-" * 40)
        print(f"Length: {len(original_chunk)} characters\n")
        
        # 显示压缩后的块
        print("COMPRESSED CHUNK:")
        print("-" * 40)
        print(compressed_chunk)
        print("-" * 40)
        print(f"Length: {len(compressed_chunk)} characters")
        print(f"Compression ratio: {compression_ratio:.2f}%\n")
        
        # 显示该压缩类型的整体统计信息
        avg_ratio = sum(compression_ratios) / len(compression_ratios)
        print(f"Average compression across all chunks: {avg_ratio:.2f}%")
        print(f"Total context length reduction: {evaluation_results['metrics'][comp_type]['avg_compression_ratio']}")
        print("=" * 80)
    
    # 显示压缩技术摘要表
    print("\n=== COMPRESSION SUMMARY ===\n")
    print(f"{'Technique':<15} {'Avg Ratio':<15} {'Context Length':<15} {'Original Length':<15}")
    print("-" * 60)
    
    # 打印每种压缩类型的指标
    for comp_type, metrics in evaluation_results["metrics"].items():
        print(f"{comp_type:<15} {metrics['avg_compression_ratio']:<15} {metrics['total_context_length']:<15} {metrics['original_context_length']:<15}")
