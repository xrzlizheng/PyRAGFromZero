#!/usr/bin/env python
# coding: utf-8

"""
# 增强RAG系统的查询转换技术

本脚本实现了三种查询转换技术，无需依赖LangChain等专业库，通过修改用户查询来显著提高检索信息的相关性和全面性。

## 主要转换技术

1. **查询重写**：使查询更具体、更详细，提高搜索精度
2. 后退提示：生成更广泛的查询以检索有用的上下文信息
3. 子查询分解：将复杂查询分解为更简单的组件以实现全面检索
"""

import fitz
import os
import numpy as np
import json
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)

def rewrite_query(original_query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    重写查询使其更具体、更详细，以提高检索效果
    
    参数：
        original_query (str): 原始用户查询
        model (str): 用于查询重写的模型
        
    返回：
        str: 重写后的查询
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific, detailed, and likely to retrieve relevant information."
    
    # Define the user prompt with the original query to be rewritten
    user_prompt = f"""
    Rewrite the following query to make it more specific and detailed. Include relevant terms and concepts that might help in retrieving accurate information.
    
    Original query: {original_query}
    
    Rewritten query:
    """
    
    # Generate the rewritten query using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,  # Low temperature for deterministic output
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the rewritten query, stripping any leading/trailing whitespace
    return response.choices[0].message.content.strip()


def generate_step_back_query(original_query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    生成更通用的"后退"查询以检索更广泛的上下文
    
    参数：
        original_query (str): 原始用户查询
        model (str): 用于生成后退查询的模型
        
    返回：
        str: 后退查询
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are an AI assistant specialized in search strategies. Your task is to generate broader, more general versions of specific queries to retrieve relevant background information."
    
    # Define the user prompt with the original query to be generalized
    user_prompt = f"""
    Generate a broader, more general version of the following query that could help retrieve useful background information.
    
    Original query: {original_query}
    
    Step-back query:
    """
    
    # Generate the step-back query using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,  # Slightly higher temperature for some variation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the step-back query, stripping any leading/trailing whitespace
    return response.choices[0].message.content.strip()


def decompose_query(original_query, num_subqueries=4, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    将复杂查询分解为更简单的子查询
    
    参数：
        original_query (str): 原始复杂查询
        num_subqueries (int): 要生成的子查询数量
        model (str): 用于查询分解的模型
        
    返回：
        List[str]: 更简单的子查询列表
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are an AI assistant specialized in breaking down complex questions. Your task is to decompose complex queries into simpler sub-questions that, when answered together, address the original query."
    
    # Define the user prompt with the original query to be decomposed
    user_prompt = f"""
    Break down the following complex query into {num_subqueries} simpler sub-queries. Each sub-query should focus on a different aspect of the original question.
    
    Original query: {original_query}
    
    Generate {num_subqueries} sub-queries, one per line, in this format:
    1. [First sub-query]
    2. [Second sub-query]
    And so on...
    """
    
    # Generate the sub-queries using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # Slightly higher temperature for some variation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Process the response to extract sub-queries
    content = response.choices[0].message.content.strip()
    
    # Extract numbered queries using simple parsing
    lines = content.split("\n")
    sub_queries = []
    
    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            # Remove the number and leading space
            query = line.strip()
            query = query[query.find(".")+1:].strip()
            sub_queries.append(query)
    
    return sub_queries


class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储
    """
    def __init__(self):
        """
        初始化向量存储
        """
        self.vectors = []  # 存储嵌入向量的列表
        self.texts = []  # 存储原始文本的列表
        self.metadata = []  # 存储每个文本元数据的列表
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储添加项目

        参数：
        text (str): 原始文本
        embedding (List[float]): 嵌入向量
        metadata (dict, 可选): 附加元数据
        """
        self.vectors.append(np.array(embedding))  # Convert embedding to numpy array and add to vectors list
        self.texts.append(text)  # Add the original text to texts list
        self.metadata.append(metadata or {})  # Add metadata to metadata list, use empty dict if None
    
    def similarity_search(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目

        参数：
        query_embedding (List[float]): 查询嵌入向量
        k (int): 要返回的结果数量

        返回：
        List[Dict]: 包含文本和元数据的最相似项目
        """
        if not self.vectors:
            return []  # Return empty list if no vectors are stored
        
        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)
        
        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Compute cosine similarity between query vector and stored vector
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # Append index and similarity score
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Add the corresponding text
                "metadata": self.metadata[idx],  # Add the corresponding metadata
                "similarity": score  # Add the similarity score
            })
        
        return results  # Return the list of top k similar items


def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    使用指定的OpenAI模型为给定文本创建嵌入

    参数：
    text (str): 要创建嵌入的输入文本
    model (str): 用于创建嵌入的模型

    返回：
    List[float]: 嵌入向量
    """
    # Handle both string and list inputs by converting string input to a list
    input_text = text if isinstance(text, list) else [text]
    
    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=model,
        input=input_text
    )
    
    # If input was a string, return just the first embedding
    if isinstance(text, str):
        return response.data[0].embedding
    
    # Otherwise, return all embeddings as a list of vectors
    return [item.embedding for item in response.data]


def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本

    参数：
    pdf_path (str): PDF文件路径

    返回：
    str: 从PDF中提取的文本
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string

    return all_text  # Return the extracted text


def chunk_text(text, n=1000, overlap=200):
    """
    将给定文本分块为n个字符的段，带有重叠

    参数：
    text (str): 要分块的文本
    n (int): 每个块中的字符数
    overlap (int): 块之间的重叠字符数

    返回：
    List[str]: 文本块列表
    """
    chunks = []  # Initialize an empty list to store the chunks
    
    # Loop through the text with a step size of (n - overlap)
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from index i to i + n to the chunks list
        chunks.append(text[i:i + n])

    return chunks  # Return the list of text chunks


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    处理RAG的文档

    参数：
    pdf_path (str): PDF文件路径
    chunk_size (int): 每个块的字符大小
    chunk_overlap (int): 块之间的重叠字符数

    返回：
    SimpleVectorStore: 包含文档块及其嵌入的向量存储
    """
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")
    
    print("Creating embeddings for chunks...")
    # Create embeddings for all chunks at once for efficiency
    chunk_embeddings = create_embeddings(chunks)
    
    # Create vector store
    store = SimpleVectorStore()
    
    # Add chunks to vector store
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    
    print(f"Added {len(chunks)} chunks to the vector store")
    return store


def transformed_search(query, vector_store, transformation_type, top_k=3):
    """
    使用转换后的查询进行搜索
    
    参数：
        query (str): 原始查询
        vector_store (SimpleVectorStore): 要搜索的向量存储
        transformation_type (str): 转换类型（'rewrite', 'step_back', 或 'decompose'）
        top_k (int): 要返回的结果数量
        
    返回：
        List[Dict]: 搜索结果
    """
    print(f"Transformation type: {transformation_type}")
    print(f"Original query: {query}")
    
    results = []
    
    if transformation_type == "rewrite":
        # Query rewriting
        transformed_query = rewrite_query(query)
        print(f"Rewritten query: {transformed_query}")
        
        # Create embedding for transformed query
        query_embedding = create_embeddings(transformed_query)
        
        # Search with rewritten query
        results = vector_store.similarity_search(query_embedding, k=top_k)
        
    elif transformation_type == "step_back":
        # Step-back prompting
        transformed_query = generate_step_back_query(query)
        print(f"Step-back query: {transformed_query}")
        
        # Create embedding for transformed query
        query_embedding = create_embeddings(transformed_query)
        
        # Search with step-back query
        results = vector_store.similarity_search(query_embedding, k=top_k)
        
    elif transformation_type == "decompose":
        # Sub-query decomposition
        sub_queries = decompose_query(query)
        print("Decomposed into sub-queries:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"{i}. {sub_q}")
        
        # Create embeddings for all sub-queries
        sub_query_embeddings = create_embeddings(sub_queries)
        
        # Search with each sub-query and combine results
        all_results = []
        for i, embedding in enumerate(sub_query_embeddings):
            sub_results = vector_store.similarity_search(embedding, k=2)  # Get fewer results per sub-query
            all_results.extend(sub_results)
        
        # Remove duplicates (keep highest similarity score)
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["similarity"] > seen_texts[text]["similarity"]:
                seen_texts[text] = result
        
        # Sort by similarity and take top_k
        results = sorted(seen_texts.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]
        
    else:
        # Regular search without transformation
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
    
    return results


def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于查询和检索到的上下文生成响应
    
    参数：
        query (str): 用户查询
        context (str): 检索到的上下文
        model (str): 用于生成响应的模型
        
    返回：
        str: 生成的响应
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."
    
    # Define the user prompt with the context and query
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
    """
    
    # Generate the response using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # Low temperature for deterministic output
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the generated response, stripping any leading/trailing whitespace
    return response.choices[0].message.content.strip()


def rag_with_query_transformation(pdf_path, query, transformation_type=None):
    """
    运行带有可选查询转换的完整RAG管道
    
    参数：
        pdf_path (str): PDF文档路径
        query (str): 用户查询
        transformation_type (str): 转换类型（None, 'rewrite', 'step_back', 或 'decompose'）
        
    返回：
        Dict: 包含查询、转换查询、上下文和响应的结果
    """
    # Process the document to create a vector store
    vector_store = process_document(pdf_path)
    
    # Apply query transformation and search
    if transformation_type:
        # Perform search with transformed query
        results = transformed_search(query, vector_store, transformation_type)
    else:
        # Perform regular search without transformation
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=3)
    
    # Combine context from search results
    context = "\n\n".join([f"PASSAGE {i+1}:\n{result['text']}" for i, result in enumerate(results)])
    
    # Generate response based on the query and combined context
    response = generate_response(query, context)
    
    # Return the results including original query, transformation type, context, and response
    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }


def compare_responses(results, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    比较不同查询转换技术生成的响应
    
    参数：
        results (Dict): 不同转换技术的结果
        reference_answer (str): 用于比较的参考答案
        model (str): 用于评估的模型
    """
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = """You are an expert evaluator of RAG systems. 
    Your task is to compare different responses generated using various query transformation techniques 
    and determine which technique produced the best response compared to the reference answer."""
    
    # Prepare the comparison text with the reference answer and responses from each technique
    comparison_text = f"""Reference Answer: {reference_answer}\n\n"""
    
    for technique, result in results.items():
        comparison_text += f"{technique.capitalize()} Query Response:\n{result['response']}\n\n"
    
    # Define the user prompt with the comparison text
    user_prompt = f"""
    {comparison_text}
    
    Compare the responses generated by different query transformation techniques to the reference answer.
    
    For each technique (original, rewrite, step_back, decompose):
    1. Score the response from 1-10 based on accuracy, completeness, and relevance
    2. Identify strengths and weaknesses
    
    Then rank the techniques from best to worst and explain which technique performed best overall and why.
    """
    
    # Generate the evaluation response using the specified model
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Print the evaluation results
    print("\n===== EVALUATION RESULTS =====")
    print(response.choices[0].message.content)
    print("=============================")


def evaluate_transformations(pdf_path, query, reference_answer=None):
    """
    评估同一查询的不同转换技术
    
    参数：
        pdf_path (str): PDF文档路径
        query (str): 要评估的查询
        reference_answer (str): 用于比较的可选参考答案
        
    返回：
        Dict: 评估结果
    """
    # Define the transformation techniques to evaluate
    transformation_types = [None, "rewrite", "step_back", "decompose"]
    results = {}
    
    # Run RAG with each transformation technique
    for transformation_type in transformation_types:
        type_name = transformation_type if transformation_type else "original"
        print(f"\n===== Running RAG with {type_name} query =====")
        
        # Get the result for the current transformation type
        result = rag_with_query_transformation(pdf_path, query, transformation_type)
        results[type_name] = result
        
        # Print the response for the current transformation type
        print(f"Response with {type_name} query:")
        print(result["response"])
        print("=" * 50)
    
    # Compare results if a reference answer is provided
    if reference_answer:
        compare_responses(results, reference_answer)
    
    return results


# 从JSON文件加载验证数据
with open('data/val.json') as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[0]['question']

# 从验证数据中提取参考答案
reference_answer = data[0]['ideal_answer']

# PDF路径
pdf_path = "data/AI_Information.pdf"

# 运行评估
evaluation_results = evaluate_transformations(pdf_path, query, reference_answer)

