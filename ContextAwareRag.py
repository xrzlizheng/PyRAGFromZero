#!/usr/bin/env python
# coding: utf-8

"""
## 上下文增强的RAG检索
检索增强生成（RAG）通过从外部源检索相关知识来增强AI响应。传统检索方法返回孤立的文本块，可能导致不完整的答案。

为了解决这个问题，我们引入了上下文增强检索，确保检索到的信息包含相邻块以获得更好的连贯性。

步骤：
- 数据摄取：从PDF中提取文本
- 带重叠上下文的文本分块：将文本分割成重叠块以保留上下文
- 嵌入创建：将文本块转换为数值表示
- 上下文感知检索：检索相关块及其相邻块以获得更好的完整性
- 响应生成：使用语言模型基于检索到的上下文生成响应
- 评估：评估模型的响应准确性
"""

# 设置环境
# 我们首先导入必要的库
import fitz
import os
import numpy as np
import json
from openai import OpenAI

"""
## 从PDF文件中提取文本
要实现RAG，我们首先需要一个文本数据源。在这里，我们使用PyMuPDF库从PDF文件中提取文本。
"""
def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并打印前`num_chars`个字符

    参数：
    pdf_path (str): PDF文件的路径

    返回：
    str: 从PDF中提取的文本
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化空字符串以存储提取的文本

    # 遍历PDF中的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取页面
        text = page.get_text("text")  # 从页面提取文本
        all_text += text  # 将提取的文本追加到all_text字符串

    return all_text  # 返回提取的文本

"""
## 分割提取的文本
一旦我们有了提取的文本，我们将其分割成较小的重叠块以提高检索准确性。
"""
def chunk_text(text, n, overlap):
    """
    将给定文本分割成n个字符的段，带有重叠

    参数：
    text (str): 要分割的文本
    n (int): 每个块的字符数
    overlap (int): 块之间的重叠字符数

    返回：
    List[str]: 文本块列表
    """
    chunks = []  # 初始化空列表以存储块
    
    # 以(n - overlap)为步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 从索引i到i + n追加文本块到chunks列表
        chunks.append(text[i:i + n])

    return chunks  # 返回文本块列表

"""
## 设置OpenAI API客户端
我们初始化OpenAI客户端以生成嵌入和响应
"""
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)

"""
## 从PDF文件中提取和分割文本
现在，我们加载PDF，提取文本，并将其分割成块
"""
# 定义PDF文件路径
pdf_path = "data/AI_Information.pdf"

# 从PDF文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 将提取的文本分割成1000个字符的段，重叠200个字符
text_chunks = chunk_text(extracted_text, 1000, 200)

# 打印创建的文本块数
print("Number of text chunks:", len(text_chunks))

# 打印第一个文本块
print("\nFirst text chunk:")
print(text_chunks[0])

"""
## 为文本块创建嵌入
嵌入将文本转换为数值向量，从而实现高效的相似性搜索
"""
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    使用指定的OpenAI模型为给定文本创建嵌入

    参数：
    text (str): 要创建嵌入的输入文本
    model (str): 用于创建嵌入的模型。默认为"BAAI/bge-en-icl"

    返回：
    dict: 包含嵌入的OpenAI API响应
    """
    # 使用指定模型为输入文本创建嵌入
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # 返回包含嵌入的响应

# 为文本块创建嵌入
response = create_embeddings(text_chunks)

"""
## 实现上下文感知的语义搜索
我们修改检索以包含相邻块以获得更好的上下文
"""
def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度

    参数：
    vec1 (np.ndarray): 第一个向量
    vec2 (np.ndarray): 第二个向量

    返回：
    float: 两个向量之间的余弦相似度
    """
    # 计算两个向量的点积并除以它们的范数乘积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    """
    检索最相关的块及其相邻块

    参数：
    query (str): 搜索查询
    text_chunks (List[str]): 文本块列表
    embeddings (List[dict]): 块嵌入列表
    k (int): 要检索的相关块数
    context_size (int): 要包含的相邻块数

    返回：
    List[str]: 带有上下文信息的相关文本块
    """
    # 将查询转换为嵌入向量
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []

    # 计算查询与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        # 计算查询嵌入与当前块嵌入之间的余弦相似度
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        # 将索引和相似度分数存储为元组
        similarity_scores.append((i, similarity_score))

    # 按相似度分数降序排序（最高相似度优先）
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关块的索引
    top_index = similarity_scores[0][0]

    # 定义上下文包含的范围
    # 确保不低于0或超过text_chunks的长度
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)

    # 返回相关块及其相邻上下文块
    return [text_chunks[i] for i in range(start, end)]

"""
## 使用上下文检索运行查询
我们现在测试上下文增强检索
"""
# 从JSON文件加载验证数据集
with open('data/val.json') as f:
    data = json.load(f)

# 从数据集中提取第一个问题作为我们的查询
query = data[0]['question']

# 检索最相关的块及其相邻块以获得上下文
# 参数：
# - query: 我们正在搜索的问题
# - text_chunks: 我们从PDF中提取的文本块
# - response.data: 我们文本块的嵌入
# - k=1: 返回最佳匹配
# - context_size=1: 包括最佳匹配前后各1个块以获得上下文
top_chunks = context_enriched_search(query, text_chunks, response.data, k=1, context_size=1)

# 打印查询以供参考
print("Query:", query)
# 打印每个检索到的块，带有标题和分隔符
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")

"""
## 使用检索到的上下文生成响应
我们现在使用LLM生成响应
"""
# 定义AI助手的系统提示
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于系统提示和用户消息生成AI模型的响应

    参数：
    system_prompt (str): 指导AI行为的系统提示
    user_message (str): 用户的消息或查询
    model (str): 用于生成响应的模型。默认为"meta-llama/Llama-2-7B-chat-hf"

    返回：
    dict: AI模型的响应
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

# 基于top_chunks创建用户提示
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# 生成AI响应
ai_response = generate_response(system_prompt, user_prompt)

"""
## 评估AI响应
我们将AI响应与预期答案进行比较并分配分数
"""
# 定义评估系统的系统提示
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."

# 通过组合用户查询、AI响应、真实响应和评估系统提示创建评估提示
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估响应
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

# 打印评估响应
print(evaluation_response.choices[0].message.content)

