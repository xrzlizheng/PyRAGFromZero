#!/usr/bin/env python
# coding: utf-8

"""
## 评估简单RAG中的块大小选择

选择正确的块大小对于提高检索增强生成（RAG）管道中的检索准确性至关重要。我们的目标是在检索性能和响应质量之间取得平衡。

本节通过以下步骤评估不同的块大小：

1. 从PDF中提取文本
2. 将文本分割成不同大小的块
3. 为每个块创建嵌入
4. 检索与查询相关的块
5. 使用检索到的块生成响应
6. 评估忠实度和相关性
7. 比较不同块大小的结果
"""

# 设置环境
# 我们首先导入必要的库
import fitz
import os
import numpy as np
import json
from openai import OpenAI

"""
## 设置OpenAI API客户端
我们初始化OpenAI客户端以生成嵌入和响应
"""
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)

"""
## 从PDF中提取文本
首先，我们将从`AI_Information.pdf`文件中提取文本
"""
def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本

    参数：
    pdf_path (str): PDF文件的路径

    返回：
    str: 从PDF中提取的文本
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化空字符串以存储提取的文本
    
    # 遍历PDF中的每一页
    for page in mypdf:
        # 从当前页提取文本并添加空格
        all_text += page.get_text("text") + " "

    # 返回提取的文本，去除首尾空白
    return all_text.strip()

# 定义PDF文件路径
pdf_path = "data/AI_Information.pdf"

# 从PDF文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 打印提取文本的前500个字符
print(extracted_text[:500])

"""
## 分割提取的文本
为了提高检索效果，我们将提取的文本分割成不同大小的重叠块
"""
def chunk_text(text, n, overlap):
    """
    将文本分割成重叠的块

    参数：
    text (str): 要分割的文本
    n (int): 每个块的字符数
    overlap (int): 块之间的重叠字符数

    返回：
    List[str]: 文本块列表
    """
    chunks = []  # 初始化空列表以存储块
    for i in range(0, len(text), n - overlap):
        # 从当前索引到索引+块大小追加文本块
        chunks.append(text[i:i + n])
    
    return chunks  # 返回文本块列表

# 定义要评估的不同块大小
chunk_sizes = [128, 256, 512]

# 创建字典以存储每个块大小的文本块
text_chunks_dict = {size: chunk_text(extracted_text, size, size // 5) for size in chunk_sizes}

# 打印为每个块大小创建的块数
for size, chunks in text_chunks_dict.items():
    print(f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")

"""
## 为文本块创建嵌入
嵌入将文本转换为数值表示以进行相似性搜索
"""
from tqdm import tqdm

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为文本列表生成嵌入

    参数：
    texts (List[str]): 输入文本列表
    model (str): 嵌入模型

    返回：
    List[np.ndarray]: 数值嵌入列表
    """
    # 使用指定模型创建嵌入
    response = client.embeddings.create(model=model, input=texts)
    # 将响应转换为numpy数组列表并返回
    return [np.array(embedding.embedding) for embedding in response.data]

# 为每个块大小生成嵌入
chunk_embeddings_dict = {size: create_embeddings(chunks) for size, chunks in tqdm(text_chunks_dict.items(), desc="Generating Embeddings")}

"""
## 执行语义搜索
我们使用余弦相似度来查找与用户查询最相关的文本块
"""
def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度

    参数：
    vec1 (np.ndarray): 第一个向量
    vec2 (np.ndarray): 第二个向量

    返回：
    float: 余弦相似度得分
    """
    # 计算两个向量的点积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_relevant_chunks(query, text_chunks, chunk_embeddings, k=5):
    """
    检索最相关的top-k文本块
    
    参数：
    query (str): 用户查询
    text_chunks (List[str]): 文本块列表
    chunk_embeddings (List[np.ndarray]): 文本块的嵌入
    k (int): 要返回的top块数
    
    返回：
    List[str]: 最相关的文本块
    """
    # 为查询生成嵌入 - 将查询作为列表传递并获取第一项
    query_embedding = create_embeddings([query])[0]
    
    # 计算查询嵌入与每个块嵌入之间的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    
    # 获取最相似top-k块的索引
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # 返回最相关的top-k文本块
    return [text_chunks[i] for i in top_indices]

# 从JSON文件加载验证数据
with open('data/val.json') as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[3]['question']

# 为每个块大小检索相关块
retrieved_chunks_dict = {size: retrieve_relevant_chunks(query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}

# 打印块大小为256的检索块
print(retrieved_chunks_dict[256])

"""
## 基于检索到的块生成响应
让我们基于检索到的文本为块大小`256`生成响应
"""
# 定义AI助手的系统提示
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(query, system_prompt, retrieved_chunks, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于检索到的块生成AI响应

    参数：
    query (str): 用户查询
    retrieved_chunks (List[str]): 检索到的文本块列表
    model (str): AI模型

    返回：
    str: AI生成的响应
    """
    # 将检索到的块组合成单个上下文字符串
    context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    # 通过组合上下文和查询创建用户提示
    user_prompt = f"{context}\n\nQuestion: {query}"

    # 使用指定模型生成AI响应
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回AI响应的内容
    return response.choices[0].message.content

# 为每个块大小生成AI响应
ai_responses_dict = {size: generate_response(query, system_prompt, retrieved_chunks_dict[size]) for size in chunk_sizes}

# 打印块大小为256的响应
print(ai_responses_dict[256])

"""
## 评估AI响应
我们使用强大的LLM根据忠实度和相关性对响应进行评分
"""
# 定义评估评分系统常量
SCORE_FULL = 1.0     # 完全匹配或完全满意
SCORE_PARTIAL = 0.5  # 部分匹配或有些满意
SCORE_NONE = 0.0     # 不匹配或不满意

# 定义严格的评估提示模板
FAITHFULNESS_PROMPT_TEMPLATE = """
Evaluate the faithfulness of the AI response compared to the true answer.
User Query: {question}
AI Response: {response}
True Answer: {true_answer}

Faithfulness measures how well the AI response aligns with facts in the true answer, without hallucinations.

INSTRUCTIONS:
- Score STRICTLY using only these values:
    * {full} = Completely faithful, no contradictions with true answer
    * {partial} = Partially faithful, minor contradictions
    * {none} = Not faithful, major contradictions or hallucinations
- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
"""

RELEVANCY_PROMPT_TEMPLATE = """
Evaluate the relevancy of the AI response to the user query.
User Query: {question}
AI Response: {response}

Relevancy measures how well the response addresses the user's question.

INSTRUCTIONS:
- Score STRICTLY using only these values:
    * {full} = Completely relevant, directly addresses the query
    * {partial} = Partially relevant, addresses some aspects
    * {none} = Not relevant, fails to address the query
- Return ONLY the numerical score ({full}, {partial}, or {none}) with no explanation or additional text.
"""

def evaluate_response(question, response, true_answer):
    """
    根据忠实度和相关性评估AI生成响应的质量

    参数：
    question (str): 用户的原始问题
    response (str): 正在评估的AI生成响应
    true_answer (str): 用作真实答案的正确回答

    返回：
    Tuple[float, float]: 包含(忠实度得分, 相关性得分)的元组
                                            每个得分为：1.0（完全），0.5（部分），或0.0（无）
    """
    # 格式化评估提示
    faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            question=question, 
            response=response, 
            true_answer=true_answer,
            full=SCORE_FULL,
            partial=SCORE_PARTIAL,
            none=SCORE_NONE
    )
    
    relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
            question=question, 
            response=response,
            full=SCORE_FULL,
            partial=SCORE_PARTIAL,
            none=SCORE_NONE
    )

    # 从模型请求忠实度评估
    faithfulness_response = client.chat.completions.create(
           model="meta-llama/Llama-3.2-3B-Instruct",
            temperature=0,
            messages=[
                    {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
                    {"role": "user", "content": faithfulness_prompt}
            ]
    )
    
    # 从模型请求相关性评估
    relevancy_response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            temperature=0,
            messages=[
                    {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
                    {"role": "user", "content": relevancy_prompt}
            ]
    )
    
    # 提取分数并处理潜在的解析错误
    try:
            faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
    except ValueError:
            print("Warning: Could not parse faithfulness score, defaulting to 0")
            faithfulness_score = 0.0
            
    try:
            relevancy_score = float(relevancy_response.choices[0].message.content.strip())
    except ValueError:
            print("Warning: Could not parse relevancy score, defaulting to 0")
            relevancy_score = 0.0

    return faithfulness_score, relevancy_score

# 第一个验证数据的真实答案
true_answer = data[3]['ideal_answer']

# 评估块大小为256和128的响应
faithfulness, relevancy = evaluate_response(query, ai_responses_dict[256], true_answer)
faithfulness2, relevancy2 = evaluate_response(query, ai_responses_dict[128], true_answer)

# 打印评估分数
print(f"Faithfulness Score (Chunk Size 256): {faithfulness}")
print(f"Relevancy Score (Chunk Size 256): {relevancy}")

print(f"\n")

print(f"Faithfulness Score (Chunk Size 128): {faithfulness2}")
print(f"Relevancy Score (Chunk Size 128): {relevancy2}")

