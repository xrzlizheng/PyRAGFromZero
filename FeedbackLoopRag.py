#!/usr/bin/env python
# coding: utf-8

"""
# 带反馈循环的RAG系统

在这个脚本中，我实现了一个带有反馈循环机制的RAG系统，它能够随着时间不断改进。通过收集和整合用户反馈，我们的系统能够在每次交互中提供更相关、更高质量的回答。

传统的RAG系统是静态的 - 它们仅基于嵌入相似度检索信息。而通过反馈循环，我们创建了一个动态系统，它能够：

- 记住什么有效（什么无效）
- 随着时间调整文档相关性分数
- 将成功的问答对整合到知识库中
- 在每次用户交互中变得更智能
"""

# 设置环境
# 首先导入必要的库
import fitz
import os
import numpy as np
import json
from openai import OpenAI
from datetime import datetime


"""
# 从PDF文件中提取文本
为了实现RAG，我们首先需要文本数据源。在这个例子中，我们使用PyMuPDF库从PDF文件中提取文本。
"""
def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本。
    
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
# 对提取的文本进行分块
一旦我们提取了文本，我们将其分成更小的、重叠的块以提高检索准确性。
"""
def chunk_text(text, n, overlap):
    """
    将给定文本分成具有重叠部分的n个字符的段落。
    
    参数:
    text (str): 要分块的文本。
    n (int): 每个块中的字符数。
    overlap (int): 块之间重叠的字符数。
    
    返回:
    List[str]: 文本块列表。
    """
    chunks = []  # 初始化一个空列表来存储块
    
    # 以(n - overlap)的步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 将从索引i到i + n的文本块添加到chunks列表中
        chunks.append(text[i:i + n])
    
    return chunks  # 返回文本块列表


"""
# 设置OpenAI API客户端
我们初始化OpenAI客户端以生成嵌入和响应。
"""
# 使用基础URL和API密钥初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)


"""
# 简单向量存储实现
我们将创建一个基本的向量存储来管理文档块及其嵌入。
"""
class SimpleVectorStore:
    """
    使用NumPy的简单向量存储实现。
    
    该类提供了一个内存中的存储和检索系统，用于
    嵌入向量及其对应的文本块和元数据。
    它支持使用余弦相似度的基本相似度搜索功能。
    """
    def __init__(self):
        """
        使用向量、文本和元数据的空列表初始化向量存储。
        
        向量存储维护三个并行列表：
        - vectors: 嵌入向量的NumPy数组
        - texts: 对应每个向量的原始文本块
        - metadata: 每个项目的可选元数据
        """
        self.vectors = []  # 用于存储嵌入向量的列表
        self.texts = []    # 用于存储原始文本块的列表
        self.metadata = [] # 用于存储每个文本块的元数据的列表
    
    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加项目。

        参数:
            text (str): 要存储的原始文本块。
            embedding (List[float]): 表示文本的嵌入向量。
            metadata (dict, optional): 文本块的附加元数据，
                                      如来源、时间戳或相关性分数。
        """
        self.vectors.append(np.array(embedding))  # 转换并存储嵌入
        self.texts.append(text)                   # 存储原始文本
        self.metadata.append(metadata or {})      # 存储元数据（如果为None则为空字典）
    
    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        使用余弦相似度找到与查询嵌入最相似的项目。

        参数:
            query_embedding (List[float]): 用于与存储的向量比较的查询嵌入向量。
            k (int): 返回的最相似结果的数量。
            filter_func (callable, optional): 基于元数据过滤结果的函数。
                                             接受元数据字典作为输入并返回布尔值。

        返回:
            List[Dict]: 前k个最相似的项目，每个包含：
                - text: 原始文本
                - metadata: 关联的元数据
                - similarity: 原始余弦相似度分数
                - relevance_score: 基于元数据的相关性或计算的相似度
                
        注意: 如果没有存储向量或没有通过过滤器，则返回空列表。
        """
        if not self.vectors:
            return []  # 如果向量存储为空，则返回空列表
        
        # 将查询嵌入转换为numpy数组以进行向量操作
        query_vector = np.array(query_embedding)
        
        # 计算查询与每个存储向量之间的余弦相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 跳过不符合过滤条件的项目
            if filter_func and not filter_func(self.metadata[i]):
                continue
                
            # 计算余弦相似度：点积 / (范数1 * 范数2)
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 存储索引和相似度分数
        
        # 按相似度分数降序排序结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 为前k个匹配构建结果字典
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score,
                # 如果元数据中有相关性分数则使用，否则使用相似度
                "relevance_score": self.metadata[idx].get("relevance_score", score)
            })
        
        return results


"""
# 创建嵌入
"""
def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    为给定文本创建嵌入。

    参数:
    text (str or List[str]): 要为其创建嵌入的输入文本。
    model (str): 用于创建嵌入的模型。

    返回:
    List[float] or List[List[float]]: 嵌入向量。
    """
    # 将单个字符串转换为列表以进行统一处理
    input_text = text if isinstance(text, list) else [text]
    
    # 调用OpenAI API为所有输入文本生成嵌入
    response = client.embeddings.create(
        model=model,
        input=input_text
    )
    
    # 对于单个字符串输入，仅返回第一个嵌入向量
    if isinstance(text, str):
        return response.data[0].embedding
    
    # 对于列表输入，返回所有嵌入向量的列表
    return [item.embedding for item in response.data]


"""
# 反馈系统函数
现在我们将实现核心反馈系统组件。
"""
def get_user_feedback(query, response, relevance, quality, comments=""):
    """
    将用户反馈格式化为字典。
    
    参数:
        query (str): 用户的查询
        response (str): 系统的响应
        relevance (int): 相关性分数 (1-5)
        quality (int): 质量分数 (1-5)
        comments (str): 可选的反馈评论
        
    返回:
        Dict: 格式化的反馈
    """
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }


def store_feedback(feedback, feedback_file="feedback_data.json"):
    """
    将反馈存储在JSON文件中。
    
    参数:
        feedback (Dict): 反馈数据
        feedback_file (str): 反馈文件的路径
    """
    with open(feedback_file, "a") as f:
        json.dump(feedback, f)
        f.write("\n")


def load_feedback_data(feedback_file="feedback_data.json"):
    """
    从文件加载反馈数据。
    
    参数:
        feedback_file (str): 反馈文件的路径
        
    返回:
        List[Dict]: 反馈条目列表
    """
    feedback_data = []
    try:
        with open(feedback_file, "r") as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("No feedback data file found. Starting with empty feedback.")
    
    return feedback_data


"""
# 具有反馈感知的文档处理
"""
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    处理用于带反馈循环的RAG（检索增强生成）的文档。
    此函数处理完整的文档处理流程：
    1. 从PDF中提取文本
    2. 文本分块并重叠
    3. 为块创建嵌入
    4. 在向量数据库中存储带有元数据的内容

    参数:
    pdf_path (str): 要处理的PDF文件的路径。
    chunk_size (int): 每个文本块的字符大小。
    chunk_overlap (int): 连续块之间重叠的字符数。

    返回:
    Tuple[List[str], SimpleVectorStore]: 包含以下内容的元组：
        - 文档块列表
        - 填充了嵌入和元数据的向量存储
    """
    # 步骤1：从PDF文档中提取原始文本内容
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # 步骤2：将文本分成可管理的、重叠的块，以更好地保留上下文
    print("Chunking text...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} text chunks")
    
    # 步骤3：为每个文本块生成向量嵌入
    print("Creating embeddings for chunks...")
    chunk_embeddings = create_embeddings(chunks)
    
    # 步骤4：初始化向量数据库以存储块及其嵌入
    store = SimpleVectorStore()
    
    # 步骤5：将每个块及其嵌入添加到向量存储中
    # 包括用于基于反馈的改进的元数据
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={
                "index": i,                # 原始文档中的位置
                "source": pdf_path,        # 源文档路径
                "relevance_score": 1.0,    # 初始相关性分数（将通过反馈更新）
                "feedback_count": 0        # 此块收到的反馈计数器
            }
        )
    
    print(f"Added {len(chunks)} chunks to the vector store")
    return chunks, store


"""
# 基于反馈的相关性调整
"""
def assess_feedback_relevance(query, doc_text, feedback):
    """
    使用LLM评估过去的反馈条目是否与当前查询和文档相关。
    
    此函数通过将当前查询、过去的查询+反馈和文档内容发送给LLM
    进行相关性评估，帮助确定哪些过去的反馈应该影响当前检索。
    
    参数:
        query (str): 需要信息检索的当前用户查询
        doc_text (str): 正在评估的文档的文本内容
        feedback (Dict): 包含'query'和'response'键的先前反馈数据
        
    返回:
        bool: 如果反馈被认为与当前查询/文档相关则为True，否则为False
    """
    # 定义系统提示，指示LLM只做二元相关性判断
    system_prompt = """You are an AI system that determines if a past feedback is relevant to a current query and document.
    Answer with ONLY 'yes' or 'no'. Your job is strictly to determine relevance, not to provide explanations."""

    # 构建用户提示，包含当前查询、过去的反馈数据和截断的文档内容
    user_prompt = f"""
    Current query: {query}
    Past query that received feedback: {feedback['query']}
    Document content: {doc_text[:500]}... [truncated]
    Past response that received feedback: {feedback['response'][:500]}... [truncated]

    Is this past feedback relevant to the current query and document? (yes/no)
    """

    # 使用零温度调用LLM API以获得确定性输出
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 使用temperature=0获得一致、确定性的响应
    )
    
    # 提取并规范化响应以确定相关性
    answer = response.choices[0].message.content.strip().lower()
    return 'yes' in answer  # 如果答案包含'yes'则返回True


def adjust_relevance_scores(query, results, feedback_data):
    """
    基于历史反馈调整文档相关性分数以提高检索质量。
    
    此函数分析过去的用户反馈，动态调整检索文档的相关性分数。
    它识别与当前查询上下文相关的反馈，基于相关性评级计算分数修饰符，
    并相应地重新排序结果。
    
    参数:
        query (str): 当前用户查询
        results (List[Dict]): 带有原始相似度分数的检索文档
        feedback_data (List[Dict]): 包含用户评级的历史反馈
        
    返回:
        List[Dict]: 具有调整后相关性分数的结果，按新分数排序
    """
    # 如果没有可用的反馈数据，则返回未更改的原始结果
    if not feedback_data:
        return results
    
    print("Adjusting relevance scores based on feedback history...")
    
    # 处理每个检索的文档
    for i, result in enumerate(results):
        document_text = result["text"]
        relevant_feedback = []
        
        # 通过查询LLM评估每个历史反馈项的相关性，
        # 为这个特定的文档和查询组合找到相关反馈
        for feedback in feedback_data:
            is_relevant = assess_feedback_relevance(query, document_text, feedback)
            if is_relevant:
                relevant_feedback.append(feedback)
        
        # 如果存在相关反馈，则应用分数调整
        if relevant_feedback:
            # 从所有适用的反馈条目计算平均相关性评级
            # 反馈相关性在1-5的范围内（1=不相关，5=高度相关）
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            
            # 将平均相关性转换为0.5-1.5范围内的分数修饰符
            # - 低于3/5的分数将减少原始相似度（修饰符 < 1.0）
            # - 高于3/5的分数将增加原始相似度（修饰符 > 1.0）
            modifier = 0.5 + (avg_relevance / 5.0)
            
            # 将修饰符应用于原始相似度分数
            original_score = result["similarity"]
            adjusted_score = original_score * modifier
            
            # 使用新分数和反馈元数据更新结果字典
            result["original_similarity"] = original_score  # 保留原始分数
            result["similarity"] = adjusted_score           # 更新主要分数
            result["relevance_score"] = adjusted_score      # 更新相关性分数
            result["feedback_applied"] = True               # 标记已应用反馈
            result["feedback_count"] = len(relevant_feedback)  # 使用的反馈条目数量
            
            # 记录调整详情
            print(f"  Document {i+1}: Adjusted score from {original_score:.4f} to {adjusted_score:.4f} based on {len(relevant_feedback)} feedback(s)")
    
    # 按调整后的分数重新排序结果，确保更高质量的匹配首先出现
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results


"""
# 使用反馈微调我们的索引
"""
def fine_tune_index(current_store, chunks, feedback_data):
    """
    使用高质量反馈增强向量存储以随时间提高检索质量。
    
    此函数通过以下方式实现持续学习过程：
    1. 识别高质量反馈（高评级的问答对）
    2. 从成功的交互中创建新的检索项目
    3. 将这些添加到向量存储中，并提高相关性权重
    
    参数:
        current_store (SimpleVectorStore): 包含原始文档块的当前向量存储
        chunks (List[str]): 原始文档文本块 
        feedback_data (List[Dict]): 具有相关性和质量评级的历史用户反馈
        
    返回:
        SimpleVectorStore: 包含原始块和反馈衍生内容的增强向量存储
    """
    print("Fine-tuning index with high-quality feedback...")
    
    # 仅过滤高质量响应（相关性和质量均评为4或5）
    # 这确保我们只从最成功的交互中学习
    good_feedback = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]
    
    if not good_feedback:
        print("No high-quality feedback found for fine-tuning.")
        return current_store  # 如果没有好的反馈，则返回未更改的原始存储
    
    # 初始化将包含原始和增强内容的新存储
    new_store = SimpleVectorStore()
    
    # 首先传输所有原始文档块及其现有元数据
    for i in range(len(current_store.texts)):
        new_store.add_item(
            text=current_store.texts[i],
            embedding=current_store.vectors[i],
            metadata=current_store.metadata[i].copy()  # 使用副本以防止引用问题
        )
    
    # 从好的反馈中创建并添加增强内容
    for feedback in good_feedback:
        # 格式化一个新文档，结合问题及其高质量答案
        # 这创建了直接解决用户查询的可检索内容
        enhanced_text = f"Question: {feedback['query']}\nAnswer: {feedback['response']}"
        
        # 为这个新的合成文档生成嵌入向量
        embedding = create_embeddings(enhanced_text)
        
        # 添加到向量存储中，带有标识其来源和重要性的特殊元数据
        new_store.add_item(
            text=enhanced_text,
            embedding=embedding,
            metadata={
                "type": "feedback_enhanced",  # 标记为源自反馈
                "query": feedback["query"],   # 存储原始查询以供参考
                "relevance_score": 1.2,       # 提高初始相关性以优先考虑这些项目
                "feedback_count": 1,          # 跟踪反馈整合
                "original_feedback": feedback # 保留完整的反馈记录
            }
        )
        
        print(f"Added enhanced content from feedback: {feedback['query'][:50]}...")
    
    # 记录有关增强的摘要统计信息
    print(f"Fine-tuned index now has {len(new_store.texts)} items (original: {len(chunks)})")
    return new_store


"""
# 带反馈循环的完整RAG流程
"""
def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于查询和上下文生成响应。
    
    参数:
        query (str): 用户查询
        context (str): 从检索文档中获取的上下文文本
        model (str): 要使用的LLM模型
        
    返回:
        str: 生成的响应
    """
    # 定义系统提示以指导AI的行为
    system_prompt = """You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."""
    
    # 通过组合上下文和查询创建用户提示
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Please provide a comprehensive answer based only on the context above.
    """
    
    # 调用OpenAI API基于系统和用户提示生成响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 使用temperature=0获得一致、确定性的响应
    )
    
    # 返回生成的响应内容
    return response.choices[0].message.content


def rag_with_feedback_loop(query, vector_store, feedback_data, k=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    包含反馈循环的完整RAG流程。
    
    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 带有文档块的向量存储
        feedback_data (List[Dict]): 反馈历史
        k (int): 要检索的文档数量
        model (str): 用于响应生成的LLM模型
        
    返回:
        Dict: 包括查询、检索文档和响应的结果
    """
    print(f"\n=== Processing query with feedback-enhanced RAG ===")
    print(f"Query: {query}")
    
    # 步骤1：创建查询嵌入
    query_embedding = create_embeddings(query)
    
    # 步骤2：基于查询嵌入执行初始检索
    results = vector_store.similarity_search(query_embedding, k=k)
    
    # 步骤3：基于反馈调整检索文档的相关性分数
    adjusted_results = adjust_relevance_scores(query, results, feedback_data)
    
    # 步骤4：从调整后的结果中提取文本以构建上下文
    retrieved_texts = [result["text"] for result in adjusted_results]
    
    # 步骤5：通过连接检索的文本构建响应生成的上下文
    context = "\n\n---\n\n".join(retrieved_texts)
    
    # 步骤6：使用上下文和查询生成响应
    print("Generating response...")
    response = generate_response(query, context, model)
    
    # 步骤7：编译最终结果
    result = {
        "query": query,
        "retrieved_documents": adjusted_results,
        "response": response
    }
    
    print("\n=== Response ===")
    print(response)
    
    return result


"""
# 完整工作流程：从初始设置到反馈收集
"""
def full_rag_workflow(pdf_path, query, feedback_data=None, feedback_file="feedback_data.json", fine_tune=False):
    """
    执行带有反馈整合的完整RAG工作流程，以实现持续改进。
    
    此函数协调整个检索增强生成过程：
    1. 加载历史反馈数据
    2. 处理和分块文档
    3. 可选地使用先前反馈微调向量索引
    4. 使用反馈调整的相关性分数执行检索和生成
    5. 收集新的用户反馈以供未来改进
    6. 存储反馈以实现系统随时间学习
    
    参数:
        pdf_path (str): 要处理的PDF文档的路径
        query (str): 用户的自然语言查询
        feedback_data (List[Dict], optional): 预加载的反馈数据，如果为None则从文件加载
        feedback_file (str): 存储反馈历史的JSON文件的路径
        fine_tune (bool): 是否使用成功的过去问答对增强索引
        
    返回:
        Dict: 包含响应和检索元数据的结果
    """
    # 步骤1：如果未明确提供，则加载用于相关性调整的历史反馈
    if feedback_data is None:
        feedback_data = load_feedback_data(feedback_file)
        print(f"Loaded {len(feedback_data)} feedback entries from {feedback_file}")
    
    # 步骤2：通过提取、分块和嵌入流程处理文档
    chunks, vector_store = process_document(pdf_path)
    
    # 步骤3：通过整合高质量的过去交互微调向量索引
    # 这从成功的问答对中创建增强的可检索内容
    if fine_tune and feedback_data:
        vector_store = fine_tune_index(vector_store, chunks, feedback_data)
    
    # 步骤4：执行具有反馈感知检索的核心RAG
    # 注意：这依赖于应该在其他地方定义的rag_with_feedback_loop函数
    result = rag_with_feedback_loop(query, vector_store, feedback_data)
    
    # 步骤5：收集用户反馈以改进未来性能
    print("\n=== Would you like to provide feedback on this response? ===")
    print("Rate relevance (1-5, with 5 being most relevant):")
    relevance = input()
    
    print("Rate quality (1-5, with 5 being highest quality):")
    quality = input()
    
    print("Any comments? (optional, press Enter to skip)")
    comments = input()
    
    # 步骤6：将反馈格式化为结构化数据
    feedback = get_user_feedback(
        query=query,
        response=result["response"],
        relevance=int(relevance),
        quality=int(quality),
        comments=comments
    )
    
    # 步骤7：持久化存储反馈以实现系统持续学习
    store_feedback(feedback, feedback_file)
    print("Feedback recorded. Thank you!")
    
    return result


"""
# 评估我们的反馈循环
"""
def evaluate_feedback_loop(pdf_path, test_queries, reference_answers=None):
    """
    通过比较反馈整合前后的性能来评估反馈循环对RAG质量的影响。
    
    此函数运行一个受控实验，以测量整合反馈如何影响检索和生成：
    1. 第一轮：在没有反馈的情况下运行所有测试查询
    2. 基于参考答案生成合成反馈（如果提供）
    3. 第二轮：使用反馈增强的检索运行相同的查询
    4. 比较两轮之间的结果以量化反馈影响
    
    参数：
        pdf_path (str)：用作知识库的PDF文档路径
        test_queries (List[str])：评估系统性能的测试查询列表
        reference_answers (List[str], optional)：用于评估和合成反馈生成的参考/黄金标准答案
        
    返回：
        Dict：包含以下内容的评估结果：
            - round1_results：没有反馈的结果
            - round2_results：有反馈的结果
            - comparison：两轮之间的定量比较指标
    """
    print("=== Evaluating Feedback Loop Impact ===")
    
    # 仅为此评估会话创建临时反馈文件
    temp_feedback_file = "temp_evaluation_feedback.json"
    
    # 初始化反馈集合（开始时为空）
    feedback_data = []
    
    # ----------------------- 第一轮评估 -----------------------
    # 在没有任何反馈影响的情况下运行所有查询，以建立基准性能
    print("\n=== ROUND 1: NO FEEDBACK ===")
    round1_results = []
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        
        # 处理文档以创建初始向量存储
        chunks, vector_store = process_document(pdf_path)
        
        # 在没有反馈影响的情况下执行RAG（空反馈列表）
        result = rag_with_feedback_loop(query, vector_store, [])
        round1_results.append(result)
        
        # 如果有参考答案，则生成合成反馈
        # 这模拟用户反馈以训练系统
        if reference_answers and i < len(reference_answers):
            # 基于与参考答案的相似性计算合成反馈分数
            similarity_to_ref = calculate_similarity(result["response"], reference_answers[i])
            # 将相似性(0-1)转换为评分尺度(1-5)
            relevance = max(1, min(5, int(similarity_to_ref * 5)))
            quality = max(1, min(5, int(similarity_to_ref * 5)))
            
            # 创建结构化反馈条目
            feedback = get_user_feedback(
                query=query,
                response=result["response"],
                relevance=relevance,
                quality=quality,
                comments=f"Synthetic feedback based on reference similarity: {similarity_to_ref:.2f}"
            )
            
            # 添加到内存集合并持久化到临时文件
            feedback_data.append(feedback)
            store_feedback(feedback, temp_feedback_file)
    
    # ----------------------- 第二轮评估 -----------------------
    # 使用反馈整合运行相同的查询以测量改进
    print("\n=== ROUND 2: WITH FEEDBACK ===")
    round2_results = []
    
    # 处理文档并使用反馈衍生内容增强
    chunks, vector_store = process_document(pdf_path)
    vector_store = fine_tune_index(vector_store, chunks, feedback_data)
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        
        # 使用反馈影响执行RAG
        result = rag_with_feedback_loop(query, vector_store, feedback_data)
        round2_results.append(result)
    
    # ----------------------- 结果分析 -----------------------
    # 比较两轮之间的性能指标
    comparison = compare_results(test_queries, round1_results, round2_results, reference_answers)
    
    # 清理临时评估工件
    if os.path.exists(temp_feedback_file):
        os.remove(temp_feedback_file)
    
    return {
        "round1_results": round1_results,
        "round2_results": round2_results,
        "comparison": comparison
    }


"""
# 评估的辅助函数
"""
def calculate_similarity(text1, text2):
    """
    使用嵌入计算两个文本之间的语义相似性。
    
    参数：
        text1 (str)：第一个文本
        text2 (str)：第二个文本
        
    返回：
        float：0到1之间的相似性分数
    """
    # 为两个文本生成嵌入
    embedding1 = create_embeddings(text1)
    embedding2 = create_embeddings(text2)
    
    # 将嵌入转换为numpy数组
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # 计算两个向量之间的余弦相似度
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    return similarity


def compare_results(queries, round1_results, round2_results, reference_answers=None):
    """
    比较两轮RAG的结果。
    
    参数：
        queries (List[str])：测试查询
        round1_results (List[Dict])：第1轮的结果
        round2_results (List[Dict])：第2轮的结果
        reference_answers (List[str], optional)：参考答案
        
    返回：
        str：比较分析
    """
    print("\n=== COMPARING RESULTS ===")
    
    # 系统提示，指导AI的评估行为
    system_prompt = """You are an expert evaluator of RAG systems. Compare responses from two versions:
        1. Standard RAG: No feedback used
        2. Feedback-enhanced RAG: Uses a feedback loop to improve retrieval

        Analyze which version provides better responses in terms of:
        - Relevance to the query
        - Accuracy of information
        - Completeness
        - Clarity and conciseness
    """

    comparisons = []
    
    # 遍历每个查询及其来自两轮的相应结果
    for i, (query, r1, r2) in enumerate(zip(queries, round1_results, round2_results)):
        # 创建用于比较响应的提示
        comparison_prompt = f"""
        Query: {query}

        Standard RAG Response:
        {r1["response"]}

        Feedback-enhanced RAG Response:
        {r2["response"]}
        """

        # 如果有参考答案则包含
        if reference_answers and i < len(reference_answers):
            comparison_prompt += f"""
            Reference Answer:
            {reference_answers[i]}
            """

        comparison_prompt += """
        Compare these responses and explain which one is better and why.
        Focus specifically on how the feedback loop has (or hasn't) improved the response quality.
        """

        # 调用OpenAI API生成比较分析
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0
        )
        
        # 将比较分析附加到结果中
        comparisons.append({
            "query": query,
            "analysis": response.choices[0].message.content
        })
        
        # 为每个查询打印分析片段
        print(f"\nQuery {i+1}: {query}")
        print(f"Analysis: {response.choices[0].message.content[:200]}...")
    
    return comparisons


"""
# 反馈循环的评估（自定义验证查询）
"""
# AI文档路径
pdf_path = "data/AI_Information.pdf"

# 定义测试查询
test_queries = [
    "What is a neural network and how does it function?",

    #################################################################################
    ### 为减少测试目的的查询数量而注释掉的查询 ###
    
    # "Describe the process and applications of reinforcement learning.",
    # "What are the main applications of natural language processing in today's technology?",
    # "Explain the impact of overfitting in machine learning models and how it can be mitigated."
]

# 定义用于评估的参考答案
reference_answers = [
    "A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. It consists of layers of nodes, with each node representing a neuron. Neural networks function by adjusting the weights of connections between nodes based on the error of the output compared to the expected result.",

    ############################################################################################
    #### 为减少测试目的的查询数量而注释掉的参考答案 ###

#     "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. It involves exploration, exploitation, and learning from the consequences of actions. Applications include robotics, game playing, and autonomous vehicles.",
#     "The main applications of natural language processing in today's technology include machine translation, sentiment analysis, chatbots, information retrieval, text summarization, and speech recognition. NLP enables machines to understand and generate human language, facilitating human-computer interaction.",
#     "Overfitting in machine learning models occurs when a model learns the training data too well, capturing noise and outliers. This results in poor generalization to new data, as the model performs well on training data but poorly on unseen data. Mitigation techniques include cross-validation, regularization, pruning, and using more training data."
]

# 运行评估
evaluation_results = evaluate_feedback_loop(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)


########################################
# # 运行完整的RAG工作流程
########################################

# # 运行交互式示例
# print("\n\n=== INTERACTIVE EXAMPLE ===")
# print("Enter your query about AI:")
# user_query = input()

# # 加载累积的反馈
# all_feedback = load_feedback_data()

# # 运行完整工作流程
# result = full_rag_workflow(
#     pdf_path=pdf_path,
#     query=user_query,
#     feedback_data=all_feedback,
#     fine_tune=True
# )

########################################
# # 运行完整的RAG工作流程
########################################


"""
# 可视化反馈影响
"""
# 提取包含反馈影响分析的比较数据
comparisons = evaluation_results['comparison']

# 打印分析结果以可视化反馈影响
print("\n=== FEEDBACK IMPACT ANALYSIS ===\n")
for i, comparison in enumerate(comparisons):
    print(f"Query {i+1}: {comparison['query']}")
    print(f"\nAnalysis of feedback impact:")
    print(comparison['analysis'])
    print("\n" + "-"*50 + "\n")

# 此外，我们可以比较轮次之间的一些指标
round_responses = [evaluation_results[f'round{round_num}_results'] for round_num in range(1, len(evaluation_results) - 1)]
response_lengths = [[len(r["response"]) for r in round] for round in round_responses]

print("\nResponse length comparison (proxy for completeness):")
avg_lengths = [sum(lengths) / len(lengths) for lengths in response_lengths]
for round_num, avg_len in enumerate(avg_lengths, start=1):
    print(f"Round {round_num}: {avg_len:.1f} chars")

if len(avg_lengths) > 1:
    changes = [(avg_lengths[i] - avg_lengths[i-1]) / avg_lengths[i-1] * 100 for i in range(1, len(avg_lengths))]
    for round_num, change in enumerate(changes, start=2):
        print(f"Change from Round {round_num-1} to Round {round_num}: {change:.1f}%")