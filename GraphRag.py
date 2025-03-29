#!/usr/bin/env python
# coding: utf-8

"""
# 图形增强检索生成（Graph RAG）：基于图的检索增强生成

在本代码中，我实现了图形增强检索生成（Graph RAG）- 一种通过将知识组织为连接图而非平面文档集合来增强传统RAG系统的技术。这使系统能够导航相关概念并检索比标准向量相似度方法更具上下文相关性的信息。

图形增强检索生成的主要优势：

- 保留信息片段之间的关系
- 能够通过连接的概念进行遍历以找到相关上下文
- 改进复杂多部分查询的处理
- 通过可视化知识路径提供更好的可解释性
"""

"""
## 环境设置
首先导入必要的库。
"""

import os
import numpy as np
import json
import fitz  # PyMuPDF
from openai import OpenAI
from typing import List, Dict, Tuple, Any
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import re
from PIL import Image
import io

"""
## 设置OpenAI API客户端
初始化OpenAI客户端以生成嵌入和响应。
"""

# 使用基础URL和API密钥初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)

"""
## 文档处理函数
"""

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容。
    
    参数:
        pdf_path (str): PDF文件的路径
        
    返回:
        str: 提取的文本内容
    """
    print(f"Extracting text from {pdf_path}...")  # 打印正在处理的PDF路径
    pdf_document = fitz.open(pdf_path)  # 使用PyMuPDF打开PDF文件
    text = ""  # 初始化一个空字符串来存储提取的文本
    
    # 遍历PDF中的每一页
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]  # 获取页面对象
        text += page.get_text()  # 从页面提取文本并添加到文本字符串中
    
    return text  # 返回提取的文本内容

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    将文本分割成重叠的块。
    
    参数:
        text (str): 要分块的输入文本
        chunk_size (int): 每个块的字符大小
        overlap (int): 块之间的重叠字符数
        
    返回:
        List[Dict]: 带有元数据的块列表
    """
    chunks = []  # 初始化一个空列表来存储块
    
    # 以(chunk_size - overlap)的步长遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        # 从当前位置提取文本块
        chunk_text = text[i:i + chunk_size]
        
        # 确保我们不添加空块
        if chunk_text:
            # 将块及其元数据添加到列表中
            chunks.append({
                "text": chunk_text,  # 文本块
                "index": len(chunks),  # 块的索引
                "start_pos": i,  # 块在原始文本中的起始位置
                "end_pos": i + len(chunk_text)  # 块在原始文本中的结束位置
            })
    
    # 打印创建的块数量
    print(f"Created {len(chunks)} text chunks")
    
    return chunks  # 返回块列表

"""
## 创建嵌入向量
"""

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为给定文本创建嵌入向量。
    
    参数:
        texts (List[str]): 输入文本
        model (str): 嵌入模型名称
        
    返回:
        List[List[float]]: 嵌入向量
    """
    # 处理空输入
    if not texts:
        return []
        
    # 如有需要，分批处理（OpenAI API限制）
    batch_size = 100
    all_embeddings = []
    
    # 分批遍历输入文本
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

"""
## 知识图谱构建
"""

def extract_concepts(text):
    """
    使用OpenAI的API从文本中提取关键概念。
    
    参数:
        text (str): 要提取概念的文本
        
    返回:
        List[str]: 概念列表
    """
    # 系统消息，指导模型要做什么
    system_message = """Extract key concepts and entities from the provided text.
Return ONLY a list of 5-10 key terms, entities, or concepts that are most important in this text.
Format your response as a JSON array of strings."""

    # 向OpenAI API发出请求
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Extract key concepts from:\n\n{text[:3000]}"}  # API限制
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    try:
        # 从响应中解析概念
        concepts_json = json.loads(response.choices[0].message.content)
        concepts = concepts_json.get("concepts", [])
        if not concepts and "concepts" not in concepts_json:
            # 尝试获取响应中的任何数组
            for key, value in concepts_json.items():
                if isinstance(value, list):
                    concepts = value
                    break
        return concepts
    except (json.JSONDecodeError, AttributeError):
        # 如果JSON解析失败，使用备用方法
        content = response.choices[0].message.content
        # 尝试提取任何看起来像列表的内容
        matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
        if matches:
            items = re.findall(r'"([^"]*)"', matches[0])
            return items
        return []

def build_knowledge_graph(chunks):
    """
    从文本块构建知识图谱。
    
    参数:
        chunks (List[Dict]): 带有元数据的文本块列表
        
    返回:
        Tuple[nx.Graph, List[np.ndarray]]: 知识图谱和块嵌入
    """
    print("Building knowledge graph...")
    
    # 创建图
    graph = nx.Graph()
    
    # 提取块文本
    texts = [chunk["text"] for chunk in chunks]
    
    # 为所有块创建嵌入
    print("Creating embeddings for chunks...")
    embeddings = create_embeddings(texts)
    
    # 向图中添加节点
    print("Adding nodes to the graph...")
    for i, chunk in enumerate(chunks):
        # 从块中提取概念
        print(f"Extracting concepts for chunk {i+1}/{len(chunks)}...")
        concepts = extract_concepts(chunk["text"])
        
        # 添加带属性的节点
        graph.add_node(i, 
                      text=chunk["text"], 
                      concepts=concepts,
                      embedding=embeddings[i])
    
    # 基于共享概念连接节点
    print("Creating edges between nodes...")
    for i in range(len(chunks)):
        node_concepts = set(graph.nodes[i]["concepts"])
        
        for j in range(i + 1, len(chunks)):
            # 计算概念重叠
            other_concepts = set(graph.nodes[j]["concepts"])
            shared_concepts = node_concepts.intersection(other_concepts)
            
            # 如果它们共享概念，添加边
            if shared_concepts:
                # 使用嵌入计算语义相似度
                similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                
                # 基于概念重叠和语义相似度计算边权重
                concept_score = len(shared_concepts) / min(len(node_concepts), len(other_concepts))
                edge_weight = 0.7 * similarity + 0.3 * concept_score
                
                # 只添加具有显著关系的边
                if edge_weight > 0.6:
                    graph.add_edge(i, j, 
                                  weight=edge_weight,
                                  similarity=similarity,
                                  shared_concepts=list(shared_concepts))
    
    print(f"Knowledge graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph, embeddings

"""
## 图遍历和查询处理
"""

def traverse_graph(query, graph, embeddings, top_k=5, max_depth=3):
    """
    遍历知识图谱以找到与查询相关的信息。
    
    参数:
        query (str): 用户的问题
        graph (nx.Graph): 知识图谱
        embeddings (List): 节点嵌入列表
        top_k (int): 要考虑的初始节点数量
        max_depth (int): 最大遍历深度
        
    返回:
        List[Dict]: 从图遍历中获取的相关信息
    """
    print(f"Traversing graph for query: {query}")
    
    # 获取查询嵌入
    query_embedding = create_embeddings(query)
    
    # 计算查询与所有节点之间的相似度
    similarities = []
    for i, node_embedding in enumerate(embeddings):
        similarity = np.dot(query_embedding, node_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        similarities.append((i, similarity))
    
    # 按相似度排序（降序）
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 获取相似度最高的top-k个节点作为起点
    starting_nodes = [node for node, _ in similarities[:top_k]]
    print(f"Starting traversal from {len(starting_nodes)} nodes")
    
    # 初始化遍历
    visited = set()  # 用于跟踪已访问节点的集合
    traversal_path = []  # 用于存储遍历路径的列表
    results = []  # 用于存储结果的列表
    
    # 使用优先队列进行遍历
    queue = []
    for node in starting_nodes:
        heapq.heappush(queue, (-similarities[node][1], node))  # 负值用于最大堆
    
    # 使用修改后的带优先级的广度优先搜索遍历图
    while queue and len(results) < (top_k * 3):  # 将结果限制为top_k * 3
        _, node = heapq.heappop(queue)
        
        if node in visited:
            continue
        
        # 标记为已访问
        visited.add(node)
        traversal_path.append(node)
        
        # 将当前节点的文本添加到结果中
        results.append({
            "text": graph.nodes[node]["text"],
            "concepts": graph.nodes[node]["concepts"],
            "node_id": node
        })
        
        # 如果我们还没有达到最大深度，则探索邻居
        if len(traversal_path) < max_depth:
            neighbors = [(neighbor, graph[node][neighbor]["weight"]) 
                        for neighbor in graph.neighbors(node)
                        if neighbor not in visited]
            
            # 根据边权重将邻居添加到队列中
            for neighbor, weight in sorted(neighbors, key=lambda x: x[1], reverse=True):
                heapq.heappush(queue, (-weight, neighbor))
    
    print(f"Graph traversal found {len(results)} relevant chunks")
    return results, traversal_path

"""
## 响应生成
"""

def generate_response(query, context_chunks):
    """
    使用检索到的上下文生成响应。
    
    参数:
        query (str): 用户的问题
        context_chunks (List[Dict]): 从图遍历中获取的相关块
        
    返回:
        str: 生成的响应
    """
    # 从上下文中的每个块提取文本
    context_texts = [chunk["text"] for chunk in context_chunks]
    
    # 将提取的文本组合成单个上下文字符串，用"---"分隔
    combined_context = "\n\n---\n\n".join(context_texts)
    
    # 定义上下文的最大允许长度（OpenAI限制）
    max_context = 14000
    
    # 如果组合上下文超过最大长度，则截断
    if len(combined_context) > max_context:
        combined_context = combined_context[:max_context] + "... [truncated]"
    
    # 定义系统消息以指导AI助手
    system_message = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the information is not in the context, say so. Refer to specific parts of the context in your answer when possible."""

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定要使用的模型
        messages=[
            {"role": "system", "content": system_message},  # 指导助手的系统消息
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {query}"}  # 带有上下文和查询的用户消息
        ],
        temperature=0.2  # 设置响应生成的温度
    )
    
    # 返回生成的响应内容
    return response.choices[0].message.content

"""
## 可视化
"""

def visualize_graph_traversal(graph, traversal_path):
    """
    可视化知识图谱和遍历路径。
    
    参数:
        graph (nx.Graph): 知识图谱
        traversal_path (List): 按遍历顺序排列的节点列表
    """
    plt.figure(figsize=(12, 10))  # 设置图形大小
    
    # 定义节点颜色，默认为浅蓝色
    node_color = ['lightblue'] * graph.number_of_nodes()
    
    # 以浅绿色突出显示遍历路径节点
    for node in traversal_path:
        node_color[node] = 'lightgreen'
    
    # 以绿色突出显示起始节点，以红色突出显示结束节点
    if traversal_path:
        node_color[traversal_path[0]] = 'green'
        node_color[traversal_path[-1]] = 'red'
    
    # 使用弹簧布局为所有节点创建位置
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    
    # 绘制图形节点
    nx.draw_networkx_nodes(graph, pos, node_color=node_color, node_size=500, alpha=0.8)
    
    # 绘制边，宽度与权重成比例
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=weight*2, alpha=0.6)
    
    # 用红色虚线绘制遍历路径
    traversal_edges = [(traversal_path[i], traversal_path[i+1]) 
                      for i in range(len(traversal_path)-1)]
    
    nx.draw_networkx_edges(graph, pos, edgelist=traversal_edges, 
                          width=3, alpha=0.8, edge_color='red', 
                          style='dashed', arrows=True)
    
    # 为每个节点添加带有第一个概念的标签
    labels = {}
    for node in graph.nodes():
        concepts = graph.nodes[node]['concepts']
        label = concepts[0] if concepts else f"Node {node}"
        labels[node] = f"{node}: {label}"
    
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    
    plt.title("Knowledge Graph with Traversal Path")  # 设置图表标题
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()  # 调整布局
    plt.show()  # 显示图表

"""
## 完整的图形增强检索生成管道
"""

def graph_rag_pipeline(pdf_path, query, chunk_size=1000, chunk_overlap=200, top_k=3):
    """
    从文档到答案的完整图形增强检索生成管道。
    
    参数:
        pdf_path (str): PDF文档的路径
        query (str): 用户的问题
        chunk_size (int): 文本块的大小
        chunk_overlap (int): 块之间的重叠
        top_k (int): 要考虑进行遍历的顶部节点数量
        
    返回:
        Dict: 包括答案和图形可视化数据的结果
    """
    # 从PDF文档中提取文本
    text = extract_text_from_pdf(pdf_path)
    
    # 将提取的文本分割成重叠的块
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # 从文本块构建知识图谱
    graph, embeddings = build_knowledge_graph(chunks)
    
    # 遍历知识图谱以找到与查询相关的信息
    relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings, top_k)
    
    # 基于查询和相关块生成响应
    response = generate_response(query, relevant_chunks)
    
    # 可视化图遍历路径
    visualize_graph_traversal(graph, traversal_path)
    
    # 返回查询、响应、相关块、遍历路径和图
    return {
        "query": query,
        "response": response,
        "relevant_chunks": relevant_chunks,
        "traversal_path": traversal_path,
        "graph": graph
    }

"""
## 评估函数
"""

def evaluate_graph_rag(pdf_path, test_queries, reference_answers=None):
    """
    在多个测试查询上评估图形增强检索生成。
    
    参数:
        pdf_path (str): PDF文档的路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 用于比较的参考答案
        
    返回:
        Dict: 评估结果
    """
    # 从PDF中提取文本
    text = extract_text_from_pdf(pdf_path)
    
    # 将文本分割成块
    chunks = chunk_text(text)
    
    # 构建知识图谱（对所有查询只做一次）
    graph, embeddings = build_knowledge_graph(chunks)
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        print(f"Query: {query}")
        
        # 遍历图以找到相关信息
        relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings)
        
        # 生成响应
        response = generate_response(query, relevant_chunks)
        
        # 如果可用，与参考答案进行比较
        reference = None
        comparison = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
            comparison = compare_with_reference(response, reference, query)
        
        # 将当前查询的结果添加到结果中
        results.append({
            "query": query,
            "response": response,
            "reference_answer": reference,
            "comparison": comparison,
            "traversal_path_length": len(traversal_path),
            "relevant_chunks_count": len(relevant_chunks)
        })
        
        # 显示结果
        print(f"\nResponse: {response}\n")
        if comparison:
            print(f"Comparison: {comparison}\n")
    
    # 返回评估结果和图统计信息
    return {
        "results": results,
        "graph_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        }
    }

def compare_with_reference(response, reference, query):
    """
    将生成的响应与参考答案进行比较。
    
    参数:
        response (str): 生成的响应
        reference (str): 参考答案
        query (str): 原始查询
        
    返回:
        str: 比较分析
    """
    # 系统消息，指导模型如何比较响应
    system_message = """Compare the AI-generated response with the reference answer.
Evaluate based on: correctness, completeness, and relevance to the query.
Provide a brief analysis (2-3 sentences) of how well the generated response matches the reference."""

    # 构建包含查询、AI生成的响应和参考答案的提示
    prompt = f"""
Query: {query}

AI-generated response:
{response}

Reference answer:
{reference}

How well does the AI response match the reference?
"""

    # 向OpenAI API发出请求以生成比较分析
    comparison = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},  # 指导助手的系统消息
            {"role": "user", "content": prompt}  # 带有提示的用户消息
        ],
        temperature=0.0  # 设置响应生成的温度
    )
    
    # 返回生成的比较分析
    return comparison.choices[0].message.content

"""
## 在样本PDF文档上评估图形增强检索生成
"""

# 包含AI信息的PDF文档的路径
pdf_path = "data/AI_Information.pdf"

# 定义用于测试图形增强检索生成的AI相关查询
query = "What are the key applications of transformers in natural language processing?"

# 执行图形增强检索生成管道以处理文档并回答查询
results = graph_rag_pipeline(pdf_path, query)

# 打印从图形增强检索生成系统生成的响应
print("\n=== ANSWER ===")
print(results["response"])

# 定义测试查询和用于正式评估的参考答案
test_queries = [
    "How do transformers handle sequential data compared to RNNs?"
]

# 用于评估目的的参考答案
reference_answers = [
    "Transformers handle sequential data differently from RNNs by using self-attention mechanisms instead of recurrent connections. This allows transformers to process all tokens in parallel rather than sequentially, capturing long-range dependencies more efficiently and enabling better parallelization during training. Unlike RNNs, transformers don't suffer from vanishing gradient problems with long sequences."
]

# 使用测试查询运行图形增强检索生成系统的正式评估
evaluation = evaluate_graph_rag(pdf_path, test_queries, reference_answers)

# 打印评估摘要统计信息
print("\n=== EVALUATION SUMMARY ===")
print(f"Graph nodes: {evaluation['graph_stats']['nodes']}")
print(f"Graph edges: {evaluation['graph_stats']['edges']}")
for i, result in enumerate(evaluation['results']):
    print(f"\nQuery {i+1}: {result['query']}")
    print(f"Path length: {result['traversal_path_length']}")
    print(f"Chunks used: {result['relevant_chunks_count']}")

