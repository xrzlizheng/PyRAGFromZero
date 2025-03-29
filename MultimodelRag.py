#!/usr/bin/env python
# coding: utf-8

# # 基于图像描述的多模态RAG系统
# 
# 在这个脚本中，我实现了一个多模态RAG系统，它能从文档中提取文本和图像，为图像生成描述，并利用这两种内容类型来回答查询。这种方法通过将视觉信息整合到知识库中，增强了传统的RAG系统。
# 
# 传统的RAG系统只能处理文本，但许多文档中包含重要的图像、图表和表格信息。通过为这些视觉元素生成描述并将它们整合到我们的检索系统中，我们可以：
# 
# - 访问图形和图表中的锁定信息
# - 理解补充文本的表格和图表
# - 创建更全面的知识库
# - 回答依赖视觉数据的问题

# ## 环境设置
# 首先导入必要的库。



import os
import io
import numpy as np
import json
import fitz
from PIL import Image
from openai import OpenAI
import base64
import re
import tempfile
import shutil


# ## 设置OpenAI API客户端
# 我们初始化OpenAI客户端以生成嵌入和响应。



# 使用基础URL和API密钥初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)


# ## 文档处理函数



def extract_content_from_pdf(pdf_path, output_dir=None):
    """
    从PDF文件中提取文本和图像。
    
    参数:
        pdf_path (str): PDF文件的路径
        output_dir (str, optional): 保存提取图像的目录
        
    返回:
        Tuple[List[Dict], List[Dict]]: 文本数据和图像数据
    """
    # 如果未提供，则为图像创建临时目录
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
        
    text_data = []  # 存储提取的文本数据的列表
    image_paths = []  # 存储提取的图像路径的列表
    
    print(f"Extracting content from {pdf_path}...")
    
    try:
        with fitz.open(pdf_path) as pdf_file:
            # 遍历PDF中的每一页
            for page_number in range(len(pdf_file)):
                page = pdf_file[page_number]
                
                # 从页面提取文本
                text = page.get_text().strip()
                if text:
                    text_data.append({
                        "content": text,
                        "metadata": {
                            "source": pdf_path,
                            "page": page_number + 1,
                            "type": "text"
                        }
                    })
                
                # 从页面提取图像
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # 图像的XREF
                    base_image = pdf_file.extract_image(xref)
                    
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # 将图像保存到输出目录
                        img_filename = f"page_{page_number+1}_img_{img_index+1}.{image_ext}"
                        img_path = os.path.join(output_dir, img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        image_paths.append({
                            "path": img_path,
                            "metadata": {
                                "source": pdf_path,
                                "page": page_number + 1,
                                "image_index": img_index + 1,
                                "type": "image"
                            }
                        })
        
        print(f"Extracted {len(text_data)} text segments and {len(image_paths)} images")
        return text_data, image_paths
    
    except Exception as e:
        print(f"Error extracting content: {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


# ## 文本内容分块



def chunk_text(text_data, chunk_size=1000, overlap=200):
    """
    将文本数据分割成重叠的块。
    
    参数:
        text_data (List[Dict]): 从PDF提取的文本数据
        chunk_size (int): 每个块的字符大小
        overlap (int): 块之间的重叠字符数
        
    返回:
        List[Dict]: 分块后的文本数据
    """
    chunked_data = []  # 初始化一个空列表来存储分块数据
    
    for item in text_data:
        text = item["content"]  # 提取文本内容
        metadata = item["metadata"]  # 提取元数据
        
        # 如果文本太短则跳过
        if len(text) < chunk_size / 2:
            chunked_data.append({
                "content": text,
                "metadata": metadata
            })
            continue
        
        # 创建带有重叠的块
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]  # 提取指定大小的块
            if chunk:  # 确保我们不添加空块
                chunks.append(chunk)
        
        # 添加每个块及其更新的元数据
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()  # 复制原始元数据
            chunk_metadata["chunk_index"] = i  # 向元数据添加块索引
            chunk_metadata["chunk_count"] = len(chunks)  # 向元数据添加总块数
            
            chunked_data.append({
                "content": chunk,  # 块文本
                "metadata": chunk_metadata  # 更新的元数据
            })
    
    print(f"Created {len(chunked_data)} text chunks")  # 打印创建的块数
    return chunked_data  # 返回分块数据列表


# ## 使用OpenAI Vision进行图像描述



def encode_image(image_path):
    """
    将图像文件编码为base64。
    
    参数:
        image_path (str): 图像文件的路径
        
    返回:
        str: Base64编码的图像
    """
    # 以二进制读取模式打开图像文件
    with open(image_path, "rb") as image_file:
        # 读取图像文件并将其编码为base64
        encoded_image = base64.b64encode(image_file.read())
        # 将base64字节解码为字符串并返回
        return encoded_image.decode('utf-8')




def generate_image_caption(image_path):
    """
    使用OpenAI的视觉能力为图像生成描述。
    
    参数:
        image_path (str): 图像文件的路径
        
    返回:
        str: 生成的描述
    """
    # 检查文件是否存在且为图像
    if not os.path.exists(image_path):
        return "Error: Image file not found"
    
    try:
        # 打开并验证图像
        Image.open(image_path)
        
        # 将图像编码为base64
        base64_image = encode_image(image_path)
        
        # 创建API请求以生成描述
        response = client.chat.completions.create(
            model="llava-hf/llava-1.5-7b-hf", # 使用llava-1.5-7b模型
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant specialized in describing images from academic papers. "
                    "Provide detailed captions for the image that capture key information. "
                    "If the image contains charts, tables, or diagrams, describe their content and purpose clearly. "
                    "Your caption should be optimized for future retrieval when people ask questions about this content."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail, focusing on its academic content:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        # 从响应中提取描述
        caption = response.choices[0].message.content
        return caption
    
    except Exception as e:
        # 如果发生异常，返回错误消息
        return f"Error generating caption: {str(e)}"




def process_images(image_paths):
    """
    处理所有图像并生成描述。
    
    参数:
        image_paths (List[Dict]): 提取的图像路径
        
    返回:
        List[Dict]: 带有描述的图像数据
    """
    image_data = []  # 初始化一个空列表来存储带有描述的图像数据
    
    print(f"Generating captions for {len(image_paths)} images...")  # 打印要处理的图像数量
    for i, img_item in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}...")  # 打印当前正在处理的图像
        img_path = img_item["path"]  # 获取图像路径
        metadata = img_item["metadata"]  # 获取图像元数据
        
        # 为图像生成描述
        caption = generate_image_caption(img_path)
        
        # 将带有描述的图像数据添加到列表中
        image_data.append({
            "content": caption,  # 生成的描述
            "metadata": metadata,  # 图像元数据
            "image_path": img_path  # 图像路径
        })
    
    return image_data  # 返回带有描述的图像数据列表


# ## 简单向量存储实现



class MultiModalVectorStore:
    """
    多模态内容的简单向量存储实现。
    """
    def __init__(self):
        # 初始化列表以存储向量、内容和元数据
        self.vectors = []
        self.contents = []
        self.metadata = []
    
    def add_item(self, content, embedding, metadata=None):
        """
        向向量存储添加项目。
        
        参数:
            content (str): 内容（文本或图像描述）
            embedding (List[float]): 嵌入向量
            metadata (Dict, optional): 附加元数据
        """
        # 将嵌入向量、内容和元数据附加到各自的列表中
        self.vectors.append(np.array(embedding))
        self.contents.append(content)
        self.metadata.append(metadata or {})
    
    def add_items(self, items, embeddings):
        """
        向向量存储添加多个项目。
        
        参数:
            items (List[Dict]): 内容项目列表
            embeddings (List[List[float]]): 嵌入向量列表
        """
        # 遍历项目和嵌入，并将每个添加到向量存储中
        for item, embedding in zip(items, embeddings):
            self.add_item(
                content=item["content"],
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
                "content": self.contents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)  # 转换为float以进行JSON序列化
            })
        
        return results


# ## 创建嵌入



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
    
    # 分批迭代输入文本
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


# ## 完整处理流程



def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    处理用于多模态RAG的文档。
    
    参数:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的字符大小
        chunk_overlap (int): 块之间的重叠字符数
        
    返回:
        Tuple[MultiModalVectorStore, Dict]: 向量存储和文档信息
    """
    # 为提取的图像创建目录
    image_dir = "extracted_images"
    os.makedirs(image_dir, exist_ok=True)
    
    # 从PDF中提取文本和图像
    text_data, image_paths = extract_content_from_pdf(pdf_path, image_dir)
    
    # 对提取的文本进行分块
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)
    
    # 处理提取的图像以生成描述
    image_data = process_images(image_paths)
    
    # 组合所有内容项目（文本块和图像描述）
    all_items = chunked_text + image_data
    
    # 提取用于嵌入的内容
    contents = [item["content"] for item in all_items]
    
    # 为所有内容创建嵌入
    print("Creating embeddings for all content...")
    embeddings = create_embeddings(contents)
    
    # 构建向量存储并添加带有嵌入的项目
    vector_store = MultiModalVectorStore()
    vector_store.add_items(all_items, embeddings)
    
    # 准备带有文本块和图像描述计数的文档信息
    doc_info = {
        "text_count": len(chunked_text),
        "image_count": len(image_data),
        "total_items": len(all_items),
    }
    
    # 打印添加项目的摘要
    print(f"Added {len(all_items)} items to vector store ({len(chunked_text)} text chunks, {len(image_data)} image captions)")
    
    # 返回向量存储和文档信息
    return vector_store, doc_info


# ## 查询处理和响应生成



def query_multimodal_rag(query, vector_store, k=5):
    """
    查询多模态RAG系统。
    
    参数:
        query (str): 用户查询
        vector_store (MultiModalVectorStore): 带有文档内容的向量存储
        k (int): 要检索的结果数量
        
    返回:
        Dict: 查询结果和生成的响应
    """
    print(f"\n=== Processing query: {query} ===\n")
    
    # 为查询生成嵌入
    query_embedding = create_embeddings(query)
    
    # 从向量存储中检索相关内容
    results = vector_store.similarity_search(query_embedding, k=k)
    
    # 分离文本和图像结果
    text_results = [r for r in results if r["metadata"].get("type") == "text"]
    image_results = [r for r in results if r["metadata"].get("type") == "image"]
    
    print(f"Retrieved {len(results)} relevant items ({len(text_results)} text, {len(image_results)} image captions)")
    
    # 使用检索到的内容生成响应
    response = generate_response(query, results)
    
    return {
        "query": query,
        "results": results,
        "response": response,
        "text_results_count": len(text_results),
        "image_results_count": len(image_results)
    }




def generate_response(query, results):
    """
    基于查询和检索结果生成响应。
    
    参数:
        query (str): 用户查询
        results (List[Dict]): 检索到的内容
        
    返回:
        str: 生成的响应
    """
    # 从检索到的结果格式化上下文
    context = ""
    
    for i, result in enumerate(results):
        # 确定内容类型（文本或图像描述）
        content_type = "Text" if result["metadata"].get("type") == "text" else "Image caption"
        # 从元数据中获取页码
        page_num = result["metadata"].get("page", "unknown")
        
        # 将内容类型和页码附加到上下文
        context += f"[{content_type} from page {page_num}]\n"
        # 将实际内容附加到上下文
        context += result["content"]
        context += "\n\n"
    
    # 指导AI助手的系统消息
    system_message = """You are an AI assistant specializing in answering questions about documents 
    that contain both text and images. You have been given relevant text passages and image captions 
    from the document. Use this information to provide a comprehensive, accurate response to the query.
    If information comes from an image or chart, mention this in your answer.
    If the retrieved information doesn't fully answer the query, acknowledge the limitations."""

    # 包含查询和格式化上下文的用户消息
    user_message = f"""Query: {query}

    Retrieved content:
    {context}

    Please answer the query based on the retrieved content.
    """
    
    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )
    
    # 返回生成的响应
    return response.choices[0].message.content


# ## 与纯文本RAG的对比评估



def build_text_only_store(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    构建纯文本向量存储以进行比较。
    
    参数:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的字符大小
        chunk_overlap (int): 块之间的重叠字符数
        
    返回:
        MultiModalVectorStore: 纯文本向量存储
    """
    # 从PDF中提取文本（重用函数但忽略图像）
    text_data, _ = extract_content_from_pdf(pdf_path, None)
    
    # 文本分块
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)
    
    # 提取用于嵌入的内容
    contents = [item["content"] for item in chunked_text]
    
    # 创建嵌入
    print("Creating embeddings for text-only content...")
    embeddings = create_embeddings(contents)
    
    # 构建向量存储
    vector_store = MultiModalVectorStore()
    vector_store.add_items(chunked_text, embeddings)
    
    print(f"Added {len(chunked_text)} text items to text-only vector store")
    return vector_store




def evaluate_multimodal_vs_textonly(pdf_path, test_queries, reference_answers=None):
    """
    比较多模态RAG与纯文本RAG。
    
    参数:
        pdf_path (str): PDF文件的路径
        test_queries (List[str]): 测试查询
        reference_answers (List[str], optional): 参考答案
        
    返回:
        Dict: 评估结果
    """
    print("=== EVALUATING MULTI-MODAL RAG VS TEXT-ONLY RAG ===\n")
    
    # 处理多模态RAG的文档
    print("\nProcessing document for multi-modal RAG...")
    mm_vector_store, mm_doc_info = process_document(pdf_path)
    
    # 构建纯文本存储
    print("\nProcessing document for text-only RAG...")
    text_vector_store = build_text_only_store(pdf_path)
    
    # 为每个查询运行评估
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n\n=== Evaluating Query {i+1}: {query} ===")
        
        # 获取参考答案（如果有）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        
        # 运行多模态RAG
        print("\nRunning multi-modal RAG...")
        mm_result = query_multimodal_rag(query, mm_vector_store)
        
        # 运行纯文本RAG
        print("\nRunning text-only RAG...")
        text_result = query_multimodal_rag(query, text_vector_store)
        
        # 比较响应
        comparison = compare_responses(query, mm_result["response"], text_result["response"], reference)
        
        # 添加到结果中
        results.append({
            "query": query,
            "multimodal_response": mm_result["response"],
            "textonly_response": text_result["response"],
            "multimodal_results": {
                "text_count": mm_result["text_results_count"],
                "image_count": mm_result["image_results_count"]
            },
            "reference_answer": reference,
            "comparison": comparison
        })
    
    # 生成总体分析
    overall_analysis = generate_overall_analysis(results)
    
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "multimodal_doc_info": mm_doc_info
    }




def compare_responses(query, mm_response, text_response, reference=None):
    """
    比较多模态和纯文本响应。
    
    参数:
        query (str): 用户查询
        mm_response (str): 多模态响应
        text_response (str): 纯文本响应
        reference (str, optional): 参考答案
        
    返回:
        str: 比较分析
    """
    # 评估者的系统提示
    system_prompt = """You are an expert evaluator comparing two RAG systems:
    1. Multi-modal RAG: Retrieves from both text and image captions
    2. Text-only RAG: Retrieves only from text

    Evaluate which response better answers the query based on:
    - Accuracy and correctness
    - Completeness of information
    - Relevance to the query
    - Unique information from visual elements (for multi-modal)"""

    # 带有查询和响应的用户提示
    user_prompt = f"""Query: {query}

    Multi-modal RAG Response:
    {mm_response}

    Text-only RAG Response:
    {text_response}
    """

    if reference:
        user_prompt += f"""
    Reference Answer:
    {reference}
    """

        user_prompt += """
    Compare these responses and explain which one better answers the query and why.
    Note any specific information that came from images in the multi-modal response.
    """

    # 使用meta-llama/Llama-3.2-3B-Instruct生成比较
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content




def generate_overall_analysis(results):
    """
    生成多模态与纯文本RAG的总体分析。
    
    参数:
        results (List[Dict]): 每个查询的评估结果
        
    返回:
        str: 总体分析
    """
    # 评估者的系统提示
    system_prompt = """You are an expert evaluator of RAG systems. Provide an overall analysis comparing 
    multi-modal RAG (text + images) versus text-only RAG based on multiple test queries.

    Focus on:
    1. Types of queries where multi-modal RAG outperforms text-only
    2. Specific advantages of incorporating image information
    3. Any disadvantages or limitations of the multi-modal approach
    4. Overall recommendation on when to use each approach"""

    # 创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Multi-modal retrieved {result['multimodal_results']['text_count']} text chunks and {result['multimodal_results']['image_count']} image captions\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    # 带有评估摘要的用户提示
    user_prompt = f"""Based on the following evaluations of multi-modal vs text-only RAG across {len(results)} queries, 
    provide an overall analysis comparing these two approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis of the relative strengths and weaknesses of multi-modal RAG 
    compared to text-only RAG, with specific attention to how image information contributed (or didn't contribute) to response quality."""

    # 使用meta-llama/Llama-3.2-3B-Instruct生成总体分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content


# ## 多模态RAG与纯文本RAG的评估



# PDF文档的路径
pdf_path = "data/attention_is_all_you_need.pdf"

# 定义针对文本和视觉内容的测试查询
test_queries = [
    "What is the BLEU score of the Transformer (base model)?",
]

# 用于评估的可选参考答案
reference_answers = [
    "The Transformer (base model) achieves a BLEU score of 27.3 on the WMT 2014 English-to-German translation task and 38.1 on the WMT 2014 English-to-French translation task.",
]

# 运行评估
evaluation_results = evaluate_multimodal_vs_textonly(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# 打印总体分析
print("\n=== OVERALL ANALYSIS ===\n")
print(evaluation_results["overall_analysis"])

