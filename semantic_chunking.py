"""
## 语义分块技术简介
文本分块是检索增强生成（RAG）中的关键步骤，通过将大段文本分割成有意义的片段来提高检索准确性。
与固定长度分块不同，语义分块基于句子之间的内容相似性进行分割。

### 断点方法：
- **百分位数**：找到所有相似度差异的第 X 百分位数，并在下降幅度大于该值的地方分割块。
- **标准差**：在相似度下降超过平均值 X 个标准差的地方分割。
- **四分位距（IQR）**：使用四分位距离（Q3 - Q1）确定分割点。

本代码实现了使用百分位数方法进行语义分块，并在示例文本上评估其性能。
"""

"""
## 环境设置
我们首先导入必要的库。
"""

import fitz
import os
import numpy as np
import json
from openai import OpenAI

"""
## 从 PDF 文件中提取文本
要实现 RAG，我们首先需要文本数据源。这里我们使用 PyMuPDF 库从 PDF 文件中提取文本。
"""

def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取文本。

    参数：
    pdf_path (str): PDF 文件路径。

    返回：
    str: 从 PDF 中提取的文本。
    """
    # 打开 PDF 文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化空字符串以存储提取的文本
    
    # 遍历 PDF 的每一页
    for page in mypdf:
        # 从当前页面提取文本并添加空格
        all_text += page.get_text("text") + " "

    # 返回提取的文本，去除首尾空白
    return all_text.strip()

# 定义 PDF 文件路径
pdf_path = "data/AI_Information.pdf"

# 从 PDF 文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 打印提取文本的前 500 个字符
print(extracted_text[:500])

"""
## 设置 OpenAI API 客户端
我们初始化 OpenAI 客户端以生成嵌入和响应。
"""

# 使用基础 URL 和 API 密钥初始化 OpenAI 客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取 API 密钥
)

"""
## 创建句子级嵌入
我们将文本分割成句子并生成嵌入。
"""

def get_embedding(text, model="BAAI/bge-en-icl"):
    """
    使用 OpenAI 为给定文本创建嵌入。

    参数：
    text (str): 输入文本。
    model (str): 嵌入模型名称。

    返回：
    np.ndarray: 嵌入向量。
    """
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding)

# 将文本分割成句子（基本分割）
sentences = extracted_text.split(". ")

# 为每个句子生成嵌入
embeddings = [get_embedding(sentence) for sentence in sentences]

print(f"Generated {len(embeddings)} sentence embeddings.")

"""
## 计算相似度差异
我们计算连续句子之间的余弦相似度。
"""

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    参数：
    vec1 (np.ndarray): 第一个向量。
    vec2 (np.ndarray): 第二个向量。

    返回：
    float: 余弦相似度。
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 计算连续句子之间的相似度
similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

"""
## 实现语义分块
我们实现了三种不同的方法来寻找断点。
"""

def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    基于相似度下降计算分块断点。

    参数：
    similarities (List[float]): 句子之间的相似度分数列表。
    method (str): 'percentile', 'standard_deviation', 或 'interquartile'。
    threshold (float): 阈值（'percentile' 的百分位数，'standard_deviation' 的标准差）。

    返回：
    List[int]: 应该进行分块分割的索引。
    """
    # 根据选择的方法确定阈值
    if method == "percentile":
        # 计算相似度分数的第 X 百分位数
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # 计算相似度分数的平均值和标准差
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # 将阈值设置为平均值减去 X 个标准差
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # 计算第一和第三四分位数（Q1 和 Q3）
        q1, q3 = np.percentile(similarities, [25, 75])
        # 使用 IQR 规则设置异常值的阈值
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # 如果提供了无效方法，则引发错误
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # 识别相似度低于阈值值的索引
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

# 使用百分位数方法计算断点，阈值为 90
breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)

"""
## 将文本分割成语义块
我们基于计算的断点分割文本。
"""

def split_into_chunks(sentences, breakpoints):
    """
    将句子分割成语义块。

    参数：
    sentences (List[str]): 句子列表。
    breakpoints (List[int]): 应该进行分块的索引。

    返回：
    List[str]: 文本块列表。
    """
    chunks = []  # 初始化空列表以存储块
    start = 0  # 初始化起始索引

    # 遍历每个断点以创建块
    for bp in breakpoints:
        # 将句子从起始索引到当前断点追加为块
        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        start = bp + 1  # 将起始索引更新为断点后的下一个句子

    # 将剩余句子追加为最后一个块
    chunks.append(". ".join(sentences[start:]))
    return chunks  # 返回块列表

# 使用 split_into_chunks 函数创建块
text_chunks = split_into_chunks(sentences, breakpoints)

# 打印创建的块数
print(f"Number of semantic chunks: {len(text_chunks)}")

# 打印第一个块以验证结果
print("\nFirst text chunk:")
print(text_chunks[0])

"""
## 为语义块创建嵌入
我们为每个块创建嵌入以便后续检索。
"""

def create_embeddings(text_chunks):
    """
    为每个文本块创建嵌入。

    参数：
    text_chunks (List[str]): 文本块列表。

    返回：
    List[np.ndarray]: 嵌入向量列表。
    """
    # 使用 get_embedding 函数为每个文本块生成嵌入
    return [get_embedding(chunk) for chunk in text_chunks]

# 使用 create_embeddings 函数创建块嵌入
chunk_embeddings = create_embeddings(text_chunks)

"""
## 执行语义搜索
我们实现余弦相似度来检索最相关的块。
"""

def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    """
    查找与查询最相关的文本块。

    参数：
    query (str): 搜索查询。
    text_chunks (List[str]): 文本块列表。
    chunk_embeddings (List[np.ndarray]): 块嵌入列表。
    k (int): 要返回的顶部结果数。

    返回：
    List[str]: 最相关的 k 个块。
    """
    # 为查询生成嵌入
    query_embedding = get_embedding(query)
    
    # 计算查询嵌入与每个块嵌入之间的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    
    # 获取最相似的前 k 个块的索引
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # 返回最相关的前 k 个文本块
    return [text_chunks[i] for i in top_indices]

# 从 JSON 文件中加载验证数据
with open('data/val.json') as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[0]['question']

# 获取前 2 个相关块
top_chunks = semantic_search(query, text_chunks, chunk_embeddings, k=2)

# 打印查询
print(f"Query: {query}")

# 打印最相关的前 2 个文本块
for i, chunk in enumerate(top_chunks):
    print(f"Context {i+1}:\n{chunk}\n{'='*40}")

"""
## 基于检索到的块生成响应
"""

# 定义 AI 助手的系统提示
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于系统提示和用户消息生成 AI 模型的响应。

    参数：
    system_prompt (str): 指导 AI 行为的系统提示。
    user_message (str): 用户的消息或查询。
    model (str): 用于生成响应的模型。默认为 "meta-llama/Llama-2-7B-chat-hf"。

    返回：
    dict: AI 模型的响应。
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

# 基于顶部块创建用户提示
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# 生成 AI 响应
ai_response = generate_response(system_prompt, user_prompt)

"""
## 评估 AI 响应
我们将 AI 响应与预期答案进行比较并分配分数。
"""

# 定义评估系统的系统提示
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."

# 通过组合用户查询、AI 响应、真实响应和评估系统提示创建评估提示
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估响应
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

# 打印评估响应
print(evaluation_response.choices[0].message.content)

