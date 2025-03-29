#!/usr/bin/env python
# coding: utf-8

"""
# 文档增强RAG与问题生成

一种通过问题生成进行文档增强的RAG方法。通过为每个文本块生成相关问题，我们改进了检索过程，从而让语言模型能给出更好的回答。

在这个实现中，我们遵循以下步骤：
1. 数据提取：从PDF文件中提取文本
2. 分块：将文本分割成可管理的块
3. 问题生成：为每个块生成相关问题
4. 嵌入创建：为块和生成的问题创建嵌入
5. 向量存储：使用NumPy构建简单的向量存储
6. 语义搜索：为用户查询检索相关块和问题
7. 响应生成：基于检索到的内容生成答案
8. 评估：评估生成回答的质量
"""

# 设置环境
# 首先导入必要的库，就像准备一顿大餐前要先准备好食材一样

import fitz
import os
import numpy as np
import json
from openai import OpenAI
import re
from tqdm import tqdm

"""
## 从PDF文件中提取文本
要实现RAG，我们首先需要文本数据源。这里我们使用PyMuPDF库从PDF文件中提取文本。
"""

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并打印前`num_chars`个字符。

    参数：
    pdf_path (str): PDF文件路径

    返回：
    str: 从PDF中提取的文本
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
提取文本后，我们将其分割成较小的、有重叠的块，以提高检索准确性。
"""

def chunk_text(text, n, overlap):
    """
    将给定文本分割成n个字符的段，并带有重叠。

    参数：
    text (str): 要分块的文本
    n (int): 每个块的字符数
    overlap (int): 块之间的重叠字符数

    返回：
    List[str]: 文本块列表
    """
    chunks = []  # 初始化空列表来存储块
    
    # 以(n - overlap)为步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 将索引i到i + n的文本块追加到chunks列表中
        chunks.append(text[i:i + n])

    return chunks  # 返回文本块列表

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
## 为文本块生成问题
这是对简单RAG的关键增强。我们生成可以由每个文本块回答的问题。
"""

def generate_questions(text_chunk, num_questions=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    从给定的文本块生成相关问题。

    参数：
    text_chunk (str): 要生成问题的文本块
    num_questions (int): 要生成的问题数量
    model (str): 用于问题生成的模型

    返回：
    List[str]: 生成的问题列表
    """
    # 定义系统提示以指导AI的行为
    system_prompt = "You are an expert at generating relevant questions from text. Create concise questions that can be answered using only the provided text. Focus on key information and concepts."
    
    # 定义用户提示，包含文本块和要生成的问题数量
    user_prompt = f"""
    Based on the following text, generate {num_questions} different questions that can be answered using only this text:

    {text_chunk}
    
    Format your response as a numbered list of questions only, with no additional text.
    """
    
    # 使用OpenAI API生成问题
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # 从响应中提取并清理问题
    questions_text = response.choices[0].message.content.strip()
    questions = []
    
    # 使用正则表达式模式匹配提取问题
    for line in questions_text.split('\n'):
        # 移除编号并清理空白
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)
    
    return questions

"""
## 为文本创建嵌入
我们为文本块和生成的问题生成嵌入。
"""

def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    使用指定的OpenAI模型为给定文本创建嵌入。

    参数：
    text (str): 要创建嵌入的输入文本
    model (str): 用于创建嵌入的模型

    返回：
    dict: 包含嵌入的OpenAI API响应
    """
    # 使用指定模型为输入文本创建嵌入
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # 返回包含嵌入的响应

"""
## 构建简单的向量存储
我们将使用NumPy实现一个简单的向量存储。
"""

class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储
    """
    def __init__(self):
        """
        初始化向量存储
        """
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储添加项目

        参数：
        text (str): 原始文本
        embedding (List[float]): 嵌入向量
        metadata (dict, 可选): 附加元数据
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def similarity_search(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目

        参数：
        query_embedding (List[float]): 查询嵌入向量
        k (int): 要返回的结果数量

        返回：
        List[Dict]: 包含文本和元数据的top k最相似项目
        """
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
        
        # 返回top k结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        
        return results

"""
## 使用问题增强处理文档
现在我们将所有内容整合在一起，处理文档，生成问题，并构建增强的向量存储。
"""

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, questions_per_chunk=5):
    """
    使用问题增强处理文档

    参数：
    pdf_path (str): PDF文件路径
    chunk_size (int): 每个文本块的字符数
    chunk_overlap (int): 块之间的重叠字符数
    questions_per_chunk (int): 每个块生成的问题数量

    返回：
    Tuple[List[str], SimpleVectorStore]: 文本块和向量存储
    """
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    print("Chunking text...")
    text_chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(text_chunks)} text chunks")
    
    vector_store = SimpleVectorStore()
    
    print("Processing chunks and generating questions...")
    for i, chunk in enumerate(tqdm(text_chunks, desc="Processing Chunks")):
        # 为块本身创建嵌入
        chunk_embedding_response = create_embeddings(chunk)
        chunk_embedding = chunk_embedding_response.data[0].embedding
        
        # 将块添加到向量存储
        vector_store.add_item(
            text=chunk,
            embedding=chunk_embedding,
            metadata={"type": "chunk", "index": i}
        )
        
        # 为这个块生成问题
        questions = generate_questions(chunk, num_questions=questions_per_chunk)
        
        # 为每个问题创建嵌入并添加到向量存储
        for j, question in enumerate(questions):
            question_embedding_response = create_embeddings(question)
            question_embedding = question_embedding_response.data[0].embedding
            
            # 将问题添加到向量存储
            vector_store.add_item(
                text=question,
                embedding=question_embedding,
                metadata={"type": "question", "chunk_index": i, "original_chunk": chunk}
            )
    
    return text_chunks, vector_store

"""
## 提取和处理文档
"""

# 定义PDF文件路径
pdf_path = "data/AI_Information.pdf"

# 处理文档（提取文本，创建块，生成问题，构建向量存储）
text_chunks, vector_store = process_document(
    pdf_path, 
    chunk_size=1000, 
    chunk_overlap=200, 
    questions_per_chunk=3
)

print(f"Vector store contains {len(vector_store.texts)} items")

"""
## 执行语义搜索
我们实现了一个类似于简单RAG实现的语义搜索函数，但适应了我们增强的向量存储。
"""

def semantic_search(query, vector_store, k=5):
    """
    使用查询和向量存储执行语义搜索

    参数：
    query (str): 搜索查询
    vector_store (SimpleVectorStore): 要搜索的向量存储
    k (int): 要返回的结果数量

    返回：
    List[Dict]: top k最相关项目
    """
    # 为查询创建嵌入
    query_embedding_response = create_embeddings(query)
    query_embedding = query_embedding_response.data[0].embedding
    
    # 搜索向量存储
    results = vector_store.similarity_search(query_embedding, k=k)
    
    return results

"""
## 在增强的向量存储上运行查询
"""

# 从JSON文件加载验证数据
with open('data/val.json') as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[0]['question']

# 执行语义搜索以查找相关内容
search_results = semantic_search(query, vector_store, k=5)

print("Query:", query)
print("\nSearch Results:")

# 按类型组织结果
chunk_results = []
question_results = []

for result in search_results:
    if result["metadata"]["type"] == "chunk":
        chunk_results.append(result)
    else:
        question_results.append(result)

# 首先打印块结果
print("\nRelevant Document Chunks:")
for i, result in enumerate(chunk_results):
    print(f"Context {i + 1} (similarity: {result['similarity']:.4f}):")
    print(result["text"][:300] + "...")
    print("=====================================")

# 然后打印匹配的问题
print("\nMatched Questions:")
for i, result in enumerate(question_results):
    print(f"Question {i + 1} (similarity: {result['similarity']:.4f}):")
    print(result["text"])
    chunk_idx = result["metadata"]["chunk_index"]
    print(f"From chunk {chunk_idx}")
    print("=====================================")

"""
## 为响应生成准备上下文
现在我们通过组合来自相关块和问题的信息来准备上下文。
"""

def prepare_context(search_results):
    """
    从搜索结果中准备统一的上下文以生成响应

    参数：
    search_results (List[Dict]): 语义搜索结果

    返回：
    str: 组合的上下文字符串
    """
    # 从结果中提取唯一块引用
    chunk_indices = set()
    context_chunks = []
    
    # 首先添加直接块匹配
    for result in search_results:
        if result["metadata"]["type"] == "chunk":
            chunk_indices.add(result["metadata"]["index"])
            context_chunks.append(f"Chunk {result['metadata']['index']}:\n{result['text']}")
    
    # 然后添加问题引用的块
    for result in search_results:
        if result["metadata"]["type"] == "question":
            chunk_idx = result["metadata"]["chunk_index"]
            if chunk_idx not in chunk_indices:
                chunk_indices.add(chunk_idx)
                context_chunks.append(f"Chunk {chunk_idx} (referenced by question '{result['text']}'):\n{result['metadata']['original_chunk']}")
    
    # 组合所有上下文块
    full_context = "\n\n".join(context_chunks)
    return full_context

"""
## 基于检索到的块生成响应
"""

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于查询和上下文生成响应

    参数：
    query (str): 用户的问题
    context (str): 从向量存储中检索到的上下文信息
    model (str): 用于响应生成的模型

    返回：
    str: 生成的响应
    """
    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"
    
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please answer the question based only on the context provided above. Be concise and accurate.
    """
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content

"""
## 生成并显示响应
"""

# 从搜索结果中准备上下文
context = prepare_context(search_results)

# 生成响应
response_text = generate_response(query, context)

print("\nQuery:", query)
print("\nResponse:")
print(response_text)

"""
## 评估AI响应
我们将AI响应与预期答案进行比较并打分。
"""

def evaluate_response(query, response, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    将AI响应与参考答案进行比较评估
    
    参数：
    query (str): 用户的问题
    response (str): AI生成的响应
    reference_answer (str): 参考/理想答案
    model (str): 用于评估的模型
    
    返回：
    str: 评估反馈
    """
    # 定义评估系统的系统提示
    evaluate_system_prompt = """You are an intelligent evaluation system tasked with assessing AI responses.
            
        Compare the AI assistant's response to the true/reference answer, and evaluate based on:
        1. Factual correctness - Does the response contain accurate information?
        2. Completeness - Does it cover all important aspects from the reference?
        3. Relevance - Does it directly address the question?

        Assign a score from 0 to 1:
        - 1.0: Perfect match in content and meaning
        - 0.8: Very good, with minor omissions/differences
        - 0.6: Good, covers main points but misses some details
        - 0.4: Partial answer with significant omissions
        - 0.2: Minimal relevant information
        - 0.0: Incorrect or irrelevant

        Provide your score with justification.
    """
            
    # 创建评估提示
    evaluation_prompt = f"""
        User Query: {query}

        AI Response:
        {response}

        Reference Answer:
        {reference_answer}

        Please evaluate the AI response against the reference answer.
    """
    
    # 生成评估
    eval_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": evaluate_system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]
    )
    
    return eval_response.choices[0].message.content

"""
## 运行评估
"""

# 从验证数据中获取参考答案
reference_answer = data[0]['ideal_answer']

# 评估响应
evaluation = evaluate_response(query, response_text, reference_answer)

print("\nEvaluation:")
print(evaluation)

"""
## 从PDF文件中提取和分块文本
现在，我们加载PDF，提取文本，并将其分割成块。
"""

# 定义PDF文件路径
pdf_path = "data/AI_Information.pdf"

# 从PDF文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 将提取的文本分割成1000个字符的段，重叠200个字符
text_chunks = chunk_text(extracted_text, 1000, 200)

# 打印创建的文本块数量
print("Number of text chunks:", len(text_chunks))

# 打印第一个文本块
print("\nFirst text chunk:")
print(text_chunks[0])

"""
## 为文本块创建嵌入
嵌入将文本转换为数值向量，从而实现高效的相似性搜索。
"""



# 为文本块创建嵌入
response = create_embeddings(text_chunks)

"""
## 执行语义搜索
我们实现余弦相似度来查找与用户查询最相关的文本块。
"""

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    参数：
    vec1 (np.ndarray): 第一个向量
    vec2 (np.ndarray): 第二个向量

    返回：
    float: 两个向量之间的余弦相似度
    """
    # 计算两个向量的点积并除以它们的范数的乘积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    使用给定的查询和嵌入对文本块执行语义搜索。

    参数：
    query (str): 语义搜索的查询
    text_chunks (List[str]): 要搜索的文本块列表
    embeddings (List[dict]): 文本块的嵌入列表
    k (int): 要返回的top相关文本块数量。默认为5。

    返回：
    List[str]: 基于查询的top k最相关文本块列表
    """
    # 为查询创建嵌入
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # 初始化列表来存储相似度分数

    # 计算查询嵌入与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # 追加索引和相似度分数

    # 按相似度分数降序排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # 获取top k最相似文本块的索引
    top_indices = [index for index, _ in similarity_scores[:k]]
    # 返回top k最相关的文本块
    return [text_chunks[index] for index in top_indices]

"""
## 在提取的块上运行查询
"""

# 从JSON文件加载验证数据
with open('data/val.json') as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[0]['question']

# 执行语义搜索以查找与查询最相关的2个文本块
top_chunks = semantic_search(query, text_chunks, response.data, k=2)

# 打印查询
print("查询:", query)

# 打印top 2最相关的文本块
for i, chunk in enumerate(top_chunks):
    print(f"上下文 {i + 1}:\n{chunk}\n=====================================")

"""
## 基于检索到的块生成响应
"""

# 定义AI助手的系统提示
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于系统提示和用户消息生成AI模型的响应。

    参数：
    system_prompt (str): 指导AI行为的系统提示
    user_message (str): 用户的消息或查询
    model (str): 用于生成响应的模型。默认为"meta-llama/Llama-2-7B-chat-hf"。

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

# 基于top chunks创建用户提示
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# 生成AI响应
ai_response = generate_response(system_prompt, user_prompt)

"""
## 评估AI响应
我们将AI响应与预期答案进行比较并打分。
"""

# 定义评估系统的系统提示
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."

# 通过组合用户查询、AI响应、真实响应和评估系统提示来创建评估提示
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估响应
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)


print(evaluation_response.choices[0].message.content)

