import os                     # 用于访问环境变量（如API密钥）
import time                   # 用于计时操作
import re                     # 用于正则表达式（文本清理）
import warnings               # 用于控制警告消息
import itertools              # 用于轻松创建参数组合
import getpass                # 用于安全地提示输入API密钥（如果未设置）

import numpy as np            # 用于向量操作的数值库
import pandas as pd           # 用于表格（DataFrames）操作的数据处理库
import faiss                  # 用于快速向量相似度搜索的库
from openai import OpenAI     # 用于Nebius API交互的客户端库
from tqdm.notebook import tqdm # 用于显示进度条的库
from sklearn.metrics.pairwise import cosine_similarity # 用于计算相似度分数

# 配置Pandas DataFrames的显示选项以提高可读性
pd.set_option('display.max_colwidth', 150) # 在表格单元格中显示更多文本内容
pd.set_option('display.max_rows', 100)     # 在表格中显示更多行
warnings.filterwarnings('ignore', category=FutureWarning) # 抑制特定的非关键警告

print("库导入成功！")

# --- NebiusAI API配置 ---
# 最好将API密钥存储为环境变量而不是硬编码
# 在此处提供您的实际密钥或将其设置为环境变量
NEBIUS_API_KEY = os.getenv('NEBIUS_API_KEY', None)  # 从环境变量加载API密钥
if NEBIUS_API_KEY is None:
    print("警告：未设置NEBIUS_API_KEY。请在环境变量中设置或直接在代码中提供。") 
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/" 
NEBIUS_EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"  # 用于将文本转换为向量嵌入的模型
NEBIUS_GENERATION_MODEL = "deepseek-ai/DeepSeek-V3"    # 用于生成最终答案的LLM
NEBIUS_EVALUATION_MODEL = "deepseek-ai/DeepSeek-V3"    # 用于评估生成答案的LLM

# --- 文本生成参数（用于RAG答案生成） ---
GENERATION_TEMPERATURE = 0.1  # 较低的值（例如0.1-0.3）使输出更加集中和确定性，适合基于事实的答案
GENERATION_MAX_TOKENS = 400   # 生成答案中的最大标记数（大致相当于单词/子词）
GENERATION_TOP_P = 0.9        # 核采样参数（温度的替代方案，通常默认值即可）

# --- 评估提示（评估器LLM的指令） ---
# 忠实度：答案是否忠实于提供的上下文？
FAITHFULNESS_PROMPT = """
System: 你是一个客观的评估者。评估AI回答与真实答案相比的忠实度，仅考虑真实答案中存在的信息作为基本事实。
忠实度衡量AI回答如何准确地反映真实答案中的信息，不添加未经支持的事实或与之矛盾。
严格使用0.0到1.0之间的浮点数进行评分，基于以下标准：
- 0.0：完全不忠实，矛盾或捏造信息。
- 0.1-0.4：忠实度低，有重大不准确或未经支持的主张。
- 0.5-0.6：部分忠实，但有明显的不准确或遗漏。
- 0.7-0.8：大部分忠实，只有轻微的不准确或措辞差异。
- 0.9：非常忠实，措辞略有不同但语义一致。
- 1.0：完全忠实，准确反映真实答案。
仅回复数字分数。

User:
Query: {question}
AI Response: {response}
True Answer: {true_answer}
Score:"""

# 相关性：答案是否直接回答用户的问题？
RELEVANCY_PROMPT = """
System: 你是一个客观的评估者。评估AI回答与特定用户查询的相关性。
相关性衡量回答如何直接回答用户的问题，避免不必要或离题的信息。
严格使用0.0到1.0之间的浮点数进行评分，基于以下标准：
- 0.0：完全不相关。
- 0.1-0.4：相关性低，涉及不同主题或错过核心问题。
- 0.5-0.6：部分相关，只回答查询的一部分或仅与主题相关。
- 0.7-0.8：大部分相关，回答查询的主要方面但可能包含少量无关细节。
- 0.9：高度相关，直接回答查询，包含最少的多余信息。
- 1.0：完全相关，直接且完整地回答所问的具体问题。
仅回复数字分数。

User:
Query: {question}
AI Response: {response}
Score:"""

# --- 要调整的参数（实验变量） ---
CHUNK_SIZES_TO_TEST = [150, 250]    # 要实验的块大小（以词为单位）列表
CHUNK_OVERLAPS_TO_TEST = [30, 50]   # 要实验的块重叠（以词为单位）列表
RETRIEVAL_TOP_K_TO_TEST = [3, 5]   # 要测试的'k'值（要检索的块数）列表

# --- 重排序配置（仅用于重排序策略） ---
RERANK_RETRIEVAL_MULTIPLIER = 3 # 用于模拟重排序：初始检索K * multiplier个块

# --- 验证API密钥 --- 
print("--- 配置检查 --- ")
print(f"尝试从环境变量'NEBIUS_API_KEY'加载Nebius API密钥...")
if not NEBIUS_API_KEY:
    print("在环境变量中未找到Nebius API密钥。")
    # 如果未找到密钥，安全地提示用户
    NEBIUS_API_KEY = getpass.getpass("请输入您的Nebius API密钥：")
else:
    print("成功从环境变量加载Nebius API密钥。")

# 打印关键设置的摘要以供验证
print(f"模型：嵌入='{NEBIUS_EMBEDDING_MODEL}'，生成='{NEBIUS_GENERATION_MODEL}'，评估='{NEBIUS_EVALUATION_MODEL}'")
print(f"要测试的块大小：{CHUNK_SIZES_TO_TEST}")
print(f"要测试的重叠：{CHUNK_OVERLAPS_TO_TEST}")
print(f"要测试的Top-K值：{RETRIEVAL_TOP_K_TO_TEST}")
print(f"生成温度：{GENERATION_TEMPERATURE}，最大标记数：{GENERATION_MAX_TOKENS}")
print("配置就绪。")
print("-" * 25) 

def chunk_text(text, chunk_size, chunk_overlap):
    """将单个文本文档分割成基于词数的重叠块。

    参数：
        text (str): 要分块的输入文本。
        chunk_size (int): 每个块的目标词数。
        chunk_overlap (int): 连续块之间重叠的词数。

    返回：
        list[str]: 文本块列表。
    """
    words = text.split()      # 将文本分割成单个单词列表
    total_words = len(words) # 计算文本中的总词数
    chunks = []             # 初始化一个空列表来存储生成的块
    start_index = 0         # 初始化第一个块的起始词索引

    # --- 输入验证 ---
    # 确保chunk_size是正整数。
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        print(f"  警告：无效的chunk_size（{chunk_size}）。必须是正整数。返回整个文本作为一个块。")
        return [text]
    # 确保chunk_overlap是非负整数且小于chunk_size。
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        print(f"  警告：无效的chunk_overlap（{chunk_overlap}）。必须是非负整数。将重叠设置为0。")
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        # 如果重叠太大，将其调整为chunk_size的合理分数（例如1/3）
        # 这可以防止无限循环或无意义的分块。
        adjusted_overlap = chunk_size // 3
        print(f"  警告：chunk_overlap（{chunk_overlap}）>= chunk_size（{chunk_size}）。将重叠调整为{adjusted_overlap}。")
        chunk_overlap = adjusted_overlap

    # --- 分块循环 ---
    # 只要start_index在文本范围内就继续分块
    while start_index < total_words:
        # 确定当前块的结束索引。
        # 它是（start + chunk_size）和总词数中的较小值。
        end_index = min(start_index + chunk_size, total_words)
        
        # 提取当前块的词并将它们重新连接成单个字符串。
        current_chunk_text = " ".join(words[start_index:end_index])
        chunks.append(current_chunk_text) # 将生成的块添加到列表中
        
        # 计算*下一个*块的起始索引。
        # 向前移动（chunk_size - chunk_overlap）个词。
        next_start_index = start_index + chunk_size - chunk_overlap
        
        # --- 安全检查 ---
        # 检查1：如果重叠导致没有进展，防止无限循环。
        # 如果chunk_size非常小或重叠相对于chunk_size非常大，就会发生这种情况。
        if next_start_index <= start_index:
            if end_index == total_words: # 如果已经到达末尾，可以安全地退出。
                break
            else: 
                # 通过至少向前移动一个词来强制进展。
                print(f"  警告：分块逻辑卡住（start={start_index}, next_start={next_start_index}）。强制进展。")
                next_start_index = start_index + 1 
                
        # 检查2：如果计算的下一个起始索引已经达到或超过总词数，我们就完成了。
        if next_start_index >= total_words:
            break
            
        # 将start_index移动到计算的位置以进行下一次迭代。
        start_index = next_start_index
        
    return chunks # 返回文本块列表

# --- 输入数据：知识源和我们的问题 ---
# 我们的知识库：关于可再生能源的文本文档列表
corpus_texts = [
    "太阳能使用光伏板或CSP系统。光伏直接将阳光转换为电能。CSP使用镜子加热流体驱动涡轮机。它很清洁但随天气/时间变化。存储（电池）对一致性至关重要。", # 文档0
    "风能使用风力发电场的涡轮机。它可持续且运营成本低。风速变化，选址可能具有挑战性（视觉/噪音）。海上风电更强且更稳定。", # 文档1
    "水力发电使用流动的水，通常通过水坝旋转涡轮机。可靠的大规模发电，具有防洪/蓄水效益。大型水坝会破坏生态系统并迁移社区。径流式发电较小，破坏性较小。", # 文档2
    "地热能使用地球的热量通过蒸汽/热水驱动涡轮机。24/7稳定发电，占地面积小。初始钻井成本高，地点受地理限制。", # 文档3
    "生物质能来自有机物（木材、作物、废物）。直接燃烧或转化为生物燃料。利用废物，提供可调度电力。需要可持续来源。燃烧释放排放（如果通过再生长平衡则为碳中和）。" # 文档4
]

# 我们将向RAG系统提出的问题
test_query = "比较太阳能和水力发电的一致性和环境影响。"

# !!! 重要：'真实答案'必须只能从上面的corpus_texts中得出 !!!
# 这是我们用于评估的基本事实。
true_answer_for_query = "太阳能的一致性随天气和时间变化，需要电池等存储。水力发电通常可靠，但大型水坝对生态系统和社区有重大环境影响，与太阳能的主要影响是面板用地不同。"

print(f"已加载{len(corpus_texts)}个文档到我们的语料库中。")
print(f"测试查询：'{test_query}'")
print(f"用于评估的参考（真实）答案：'{true_answer_for_query}'")
print("输入数据就绪。")
print("-" * 25) 

def calculate_cosine_similarity(text1, text2, client, embedding_model):
    """计算两个文本嵌入之间的余弦相似度。

    参数：
        text1 (str): 第一个文本字符串。
        text2 (str): 第二个文本字符串。
        client (OpenAI): 初始化的Nebius AI客户端。
        embedding_model (str): 要使用的嵌入模型名称。

    返回：
        float: 余弦相似度分数（在0.0和1.0之间），如果发生错误则返回0.0。
    """
    if not client:
        print("  错误：Nebius客户端不可用于相似度计算。")
        return 0.0
    if not text1 or not text2:
        # 处理一个或两个文本可能为空或None的情况
        return 0.0
        
    try:
        # 如果可能，在单个API调用中为两个文本生成嵌入
        response = client.embeddings.create(model=embedding_model, input=[text1, text2])
        
        # 提取嵌入向量
        embedding1 = np.array(response.data[0].embedding)
        embedding2 = np.array(response.data[1].embedding)
        
        # 将向量重塑为cosine_similarity期望的2D数组
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)
        
        # 使用scikit-learn计算余弦相似度
        # cosine_similarity返回一个2D数组，例如[[similarity]]，所以我们提取值。
        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
        
        # 为了安全/一致性，将分数限制在0.0和1.0之间
        return max(0.0, min(1.0, similarity_score))
        
    except Exception as e:
        print(f"  计算余弦相似度时出错：{e}")
        return 0.0 # 在发生任何API或计算错误时返回0.0

# --- 快速测试 ---
print("定义'calculate_cosine_similarity'函数。")
if client: # 仅在客户端初始化时运行测试
    test_sim = calculate_cosine_similarity("苹果", "橙子", client, NEBIUS_EMBEDDING_MODEL)
    print(f"测试相似度函数：'苹果'和'橙子'之间的相似度 = {test_sim:.2f}")
else:
    print("由于Nebius客户端未初始化，跳过相似度函数测试。")
print("-" * 25)

# --- Nebius AI客户端设置 ---
client = None # 全局初始化client变量为None

print("尝试初始化Nebius AI客户端...")
try:
    # 在创建客户端之前检查API密钥是否实际可用
    if not NEBIUS_API_KEY:
        raise ValueError("缺少Nebius API密钥。无法初始化客户端。")
        
    # 创建OpenAI客户端对象，配置为Nebius API。
    client = OpenAI(
        api_key=NEBIUS_API_KEY,     # 传递之前加载的API密钥
        base_url=NEBIUS_BASE_URL  # 指定Nebius API端点
    )
    
    # 可选：添加快速测试调用以验证客户端连接，
    # 例如，列出模型（如果支持且需要）。这可能会产生费用。
    # try:
    #     client.models.list() 
    #     print("通过列出模型验证了客户端连接。")
    # except Exception as test_e:
    #     print(f"警告：无法通过测试调用验证客户端连接：{test_e}")
    
    print("Nebius AI客户端初始化成功。准备进行API调用。")
    
except Exception as e:
    # 捕获客户端初始化期间的任何错误（例如，无效密钥，网络问题）
    print(f"初始化Nebius AI客户端时出错：{e}")
    print("!!! 没有有效的客户端无法继续执行。请检查您的API密钥和网络连接。 !!!")
    # 如果初始化失败，将client设置回None以防止进一步尝试
    client = None 

print("客户端设置步骤完成。")
print("-" * 25) 

# 存储每个实验运行的详细结果的列表
all_results = []

# --- 用于分块/嵌入/索引的缓存变量 --- 
# 这些变量帮助我们避免在只有'top_k'改变时进行冗余计算。
last_chunk_size = -1      # 存储上一次迭代中使用的chunk_size
last_overlap = -1         # 存储上一次迭代中使用的chunk_overlap
current_index = None      # 保存活动的FAISS索引
current_chunks = []       # 保存活动设置的文本块列表
current_embeddings = None # 保存活动块的嵌入的numpy数组

# 在开始之前检查Nebius客户端是否成功初始化
if not client:
    print("停止：Nebius AI客户端未初始化。无法运行实验。")
else:
    print("=== 开始RAG实验循环 ===\n")
    
    # 创建所有可能的调优参数组合
    param_combinations = list(itertools.product(
        CHUNK_SIZES_TO_TEST,
        CHUNK_OVERLAPS_TO_TEST,
        RETRIEVAL_TOP_K_TO_TEST
    ))
    
    print(f"要测试的参数组合总数：{len(param_combinations)}")
    
    # --- 主循环 --- 
    # 遍历每个组合（chunk_size, chunk_overlap, top_k）
    # 使用tqdm显示进度条。
    for chunk_size, chunk_overlap, top_k in tqdm(param_combinations, desc="测试配置"):
        
        # --- 8.1 处理分块配置 --- 
        # 检查分块设置是否已更改，需要重新处理。
        if chunk_size != last_chunk_size or chunk_overlap != last_overlap:
            # 取消注释下面的行以在执行期间进行更详细的日志记录
            # print(f"\n--- 处理新的分块配置：大小={chunk_size}，重叠={chunk_overlap} ---")
            
            # 更新缓存变量
            last_chunk_size, last_overlap = chunk_size, chunk_overlap
            # 为新配置重置索引、块和嵌入
            current_index = None 
            current_chunks = []
            current_embeddings = None
            
            # --- 8.1a：分块 --- 
            # 将chunk_text函数应用于语料库中的每个文档
            try:
                # print("  分块文档...") # 取消注释以进行详细日志记录
                temp_chunks = []
                for doc_index, doc in enumerate(corpus_texts):
                    doc_chunks = chunk_text(doc, chunk_size, chunk_overlap)
                    if not doc_chunks:
                         print(f"  警告：使用大小={chunk_size}，重叠={chunk_overlap}为文档{doc_index}创建了0个块。跳过文档。")
                         continue
                    temp_chunks.extend(doc_chunks)
                
                current_chunks = temp_chunks
                if not current_chunks:
                    # 如果没有创建任何块（例如，由于无效设置或空语料库）
                    raise ValueError("当前配置没有创建任何块。")
                # print(f"    总共创建了{len(current_chunks)}个块。") # 取消注释以进行详细日志记录
            except Exception as e:
                 print(f"    使用大小={chunk_size}，重叠={chunk_overlap}进行分块时出错：{e}。跳过此配置。")
                 last_chunk_size, last_overlap = -1, -1 # 重置缓存状态
                 continue # 移动到下一个参数组合
            
            # --- 8.1b：嵌入 --- 
            # 使用Nebius API为所有块生成嵌入。
            # print("  生成嵌入...") # 取消注释以进行详细日志记录
            try:
                batch_size = 32 # 分批处理块以避免使API不堪重负或达到限制。
                temp_embeddings = [] # 临时列表存储嵌入向量
                
                # 分批循环处理块
                for i in range(0, len(current_chunks), batch_size):
                    batch_texts = current_chunks[i : min(i + batch_size, len(current_chunks))]
                    # 为当前批次调用Nebius API
                    response = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=batch_texts)
                    # 从API响应中提取嵌入向量
                    batch_embeddings = [item.embedding for item in response.data]
                    temp_embeddings.extend(batch_embeddings)
                    time.sleep(0.05) # 在批次之间添加小延迟以对API端点保持礼貌。
                
                # 将嵌入列表转换为单个NumPy数组
                current_embeddings = np.array(temp_embeddings)
                # 对嵌入数组进行基本验证检查
                if current_embeddings.ndim != 2 or current_embeddings.shape[0] != len(current_chunks):
                    raise ValueError(f"嵌入数组形状不匹配。期望({len(current_chunks)}, dim)，得到{current_embeddings.shape}")
                # print(f"    生成了{current_embeddings.shape[0]}个嵌入（维度：{current_embeddings.shape[1]}）。") # 取消注释以进行详细日志记录

            except Exception as e:
                print(f"    使用大小={chunk_size}，重叠={chunk_overlap}生成嵌入时出错：{e}。跳过此分块配置。")
                # 重置缓存变量以指示此分块设置失败
                last_chunk_size, last_overlap = -1, -1 
                current_chunks = []
                current_embeddings = None
                continue # 跳过到下一个参数组合
                
            # --- 8.1c：索引 --- 
            # 构建FAISS索引以进行高效的相似度搜索。
            # print("  构建FAISS搜索索引...") # 取消注释以进行详细日志记录
            try:
                embedding_dim = current_embeddings.shape[1] # 获取嵌入的维度
                # 我们使用IndexFlatL2，它使用L2（欧几里得）距离执行精确搜索。
                # 对于来自现代嵌入模型的高维向量，余弦相似度通常效果更好，
                # 但FAISS的IndexFlatIP（内积）密切相关。对于归一化嵌入（如许多BGE模型），
                # L2距离和内积/余弦相似度排名是等价的。
                current_index = faiss.IndexFlatL2(embedding_dim)
                # 将块嵌入添加到索引。FAISS需要float32数据类型。
                current_index.add(current_embeddings.astype('float32'))
                
                if current_index.ntotal == 0:
                     raise ValueError("添加向量后FAISS索引为空。没有添加任何向量。")
                # print(f"    FAISS索引准备就绪，包含{current_index.ntotal}个向量。") # 取消注释以进行详细日志记录
            except Exception as e:
                print(f"    使用大小={chunk_size}，重叠={chunk_overlap}构建FAISS索引时出错：{e}。跳过此分块配置。")
                # 重置变量以指示失败
                last_chunk_size, last_overlap = -1, -1
                current_index = None
                current_embeddings = None
                current_chunks = []
                continue # 跳过到下一个参数组合
        
        # --- 8.2 测试当前Top-K的RAG策略 --- 
        # 如果我们到达这一点，我们为当前chunk_size/overlap有一个有效的索引和块。
        
        # 检查索引和块是否实际可用（安全检查）
        if current_index is None or not current_chunks:
            print(f"    警告：大小={chunk_size}，重叠={chunk_overlap}的索引或块不可用。跳过Top-K={top_k}测试。")
            continue
            
        # --- 8.3 运行和评估单个RAG策略 --- 
        # 定义一个嵌套函数来执行核心RAG步骤（检索、生成、评估）
        # 这避免了每个策略的代码重复。
        def run_and_evaluate(strategy_name, query_to_use, k_retrieve, use_simulated_rerank=False):
            # print(f"    开始：{strategy_name}（k={k_retrieve}）...") # 取消注释以进行详细日志记录
            run_start_time = time.time() # 记录运行开始时间
            
            # 初始化一个字典来存储此特定运行的结果
            result = {
                'chunk_size': chunk_size, 'overlap': chunk_overlap, 'top_k': k_retrieve, 
                'strategy': strategy_name,
                'retrieved_indices': [], 'rewritten_query': None, 'answer': '错误：执行失败',
                'faithfulness': 0.0, 'relevancy': 0.0, 'similarity_score': 0.0, 'avg_score': 0.0, 
                'time_sec': 0.0
            }
            # 如果适用，存储重写的查询
            if strategy_name == "Query Rewrite RAG": 
                result['rewritten_query'] = query_to_use

            try:
                # --- 检索步骤 --- 
                k_for_search = k_retrieve # 初始检索的块数
                if use_simulated_rerank:
                    # 对于模拟重排序，初始检索更多候选
                    k_for_search = k_retrieve * RERANK_RETRIEVAL_MULTIPLIER
                    # print(f"      重排序：初始检索{k_for_search}个候选。") # 取消注释以进行详细日志记录
                
                # 1. 嵌入查询（原始或重写）
                query_embedding_response = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=[query_to_use])
                query_embedding = query_embedding_response.data[0].embedding
                query_vector = np.array([query_embedding]).astype('float32') # FAISS需要float32 numpy数组
                
                # 2. 在FAISS索引中执行搜索
                # 确保k不大于索引中的项目总数
                actual_k = min(k_for_search, current_index.ntotal)
                if actual_k == 0:
                    raise ValueError("索引为空或k_for_search为零，无法搜索。")
                
                # `current_index.search`返回最近邻的距离和索引
                distances, indices = current_index.search(query_vector, actual_k)
                
                # 3. 处理检索到的索引
                # 如果找到的向量少于'actual_k'，索引可以包含-1（使用IndexFlatL2不应该发生，除非k > ntotal）
                retrieved_indices_all = indices[0]
                valid_indices = retrieved_indices_all[retrieved_indices_all != -1].tolist()
                
                # 4. 应用模拟重排序（如果适用）
                # 在这个模拟中，我们只是从初始检索的集合中取前'k_retrieve'个结果。
                # 真正的重排序器会根据与查询的相关性重新评分这些'k_for_search'候选。
                if use_simulated_rerank:
                    final_indices = valid_indices[:k_retrieve]
                    # print(f"      重排序：模拟重排序后选择了{len(final_indices)}个索引。") # 取消注释以进行详细日志记录
                else:
                    final_indices = valid_indices # 使用所有有效检索的索引，最多到k_retrieve
                
                result['retrieved_indices'] = final_indices
                
                # 5. 获取与最终索引对应的实际文本块
                retrieved_chunks = [current_chunks[i] for i in final_indices]
                
                # 处理没有检索到块的情况（使用有效索引应该很少发生）
                if not retrieved_chunks:
                    print(f"      警告：为{strategy_name}（C={chunk_size}，O={chunk_overlap}，K={k_retrieve}）未找到相关块。设置答案以指示这一点。")
                    result['answer'] = "基于查询在文档中未找到相关上下文。"
                    # 保持分数为0.0，因为没有从上下文生成答案
                else:
                    # --- 生成步骤 --- 
                    # 将检索到的块组合成单个上下文字符串
                    context_str = "\n\n".join(retrieved_chunks)
                    
                    # 为生成LLM定义系统提示
                    sys_prompt_gen = "你是一个有帮助的AI助手。严格基于提供的上下文回答用户的问题。如果上下文中不包含答案，请明确说明。要简洁。"
                    
                    # 构造用户提示，包括上下文和*原始*查询
                    # 即使使用重写的查询进行检索，在这里使用原始查询生成最终答案也很重要。
                    user_prompt_gen = f"上下文：\n------\n{context_str}\n------\n\n查询：{test_query}\n\n答案："
                    
                    # 调用Nebius生成模型
                    gen_response = client.chat.completions.create(
                        model=NEBIUS_GENERATION_MODEL, 
                        messages=[
                            {"role": "system", "content": sys_prompt_gen},
                            {"role": "user", "content": user_prompt_gen}
                        ],
                        temperature=GENERATION_TEMPERATURE,
                        max_tokens=GENERATION_MAX_TOKENS,
                        top_p=GENERATION_TOP_P
                    )
                    # 提取生成的文本答案
                    generated_answer = gen_response.choices[0].message.content.strip()
                    result['answer'] = generated_answer
                    # 可选：打印生成答案的片段
                    # print(f"      生成的答案：{generated_answer[:100].replace('\n', ' ')}...") 

                    # --- 评估步骤 --- 
                    # 使用忠实度、相关性和相似度评估生成的答案
                    # print(f"      评估答案...（忠实度，相关性，相似度）") # 取消注释以进行详细日志记录
                    
                    # 准备评估调用的参数（使用低温度进行确定性评分）
                    eval_params = {'model': NEBIUS_EVALUATION_MODEL, 'temperature': 0.0, 'max_tokens': 10}
                    
                    # 1. 忠实度评估调用
                    prompt_f = FAITHFULNESS_PROMPT.format(question=test_query, response=generated_answer, true_answer=true_answer_for_query)
                    try:
                        resp_f = client.chat.completions.create(messages=[{"role": "user", "content": prompt_f}], **eval_params)
                        # 尝试解析分数，限制在0.0和1.0之间
                        result['faithfulness'] = max(0.0, min(1.0, float(resp_f.choices[0].message.content.strip())))
                    except Exception as eval_e:
                        print(f"      警告：{strategy_name}的忠实度分数解析错误 - {eval_e}。分数设置为0.0")
                        result['faithfulness'] = 0.0

                    # 2. 相关性评估调用
                    prompt_r = RELEVANCY_PROMPT.format(question=test_query, response=generated_answer)
                    try:
                        resp_r = client.chat.completions.create(messages=[{"role": "user", "content": prompt_r}], **eval_params)
                        # 尝试解析分数，限制在0.0和1.0之间
                        result['relevancy'] = max(0.0, min(1.0, float(resp_r.choices[0].message.content.strip())))
                    except Exception as eval_e:
                        print(f"      警告：{strategy_name}的相关性分数解析错误 - {eval_e}。分数设置为0.0")
                        result['relevancy'] = 0.0
                    
                    # 3. 相似度分数计算
                    result['similarity_score'] = calculate_cosine_similarity(
                        generated_answer, 
                        true_answer_for_query, 
                        client, 
                        NEBIUS_EMBEDDING_MODEL
                    )
                    
                    # 4. 计算平均分数（忠实度，相关性，相似度）
                    result['avg_score'] = (result['faithfulness'] + result['relevancy'] + result['similarity_score']) / 3.0
            
            except Exception as e:
                # 捕获检索/生成/评估过程中的任何意外错误
                error_message = f"执行{strategy_name}（C={chunk_size}，O={chunk_overlap}，K={k_retrieve}）时出错：{str(e)[:200]}..."
                print(f"    {error_message}")
                result['answer'] = error_message # 在答案字段中存储错误
                # 确保分数保持在默认错误状态（0.0）
                result['faithfulness'] = 0.0
                result['relevancy'] = 0.0
                result['similarity_score'] = 0.0
                result['avg_score'] = 0.0
            
            # 记录此运行的总时间
            run_end_time = time.time()
            result['time_sec'] = run_end_time - run_start_time
            
            # 打印此运行的摘要行（对监控进度有用）
            print(f"    完成：{strategy_name}（C={chunk_size}，O={chunk_overlap}，K={k_retrieve}）。平均分数={result['avg_score']:.2f}，时间={result['time_sec']:.2f}秒")
            return result
        # --- run_and_evaluate嵌套函数结束 ---

        # --- 使用run_and_evaluate函数执行RAG策略 --- 
        
        # 策略1：简单RAG（使用原始查询进行检索）
        result_simple = run_and_evaluate("Simple RAG", test_query, top_k)
        all_results.append(result_simple)

        # 策略2：查询重写RAG 
        rewritten_q = test_query # 如果重写失败，默认为原始查询
        try:
             # print("    尝试为重写RAG重写查询...") # 取消注释以进行详细日志记录
             # 为查询重写任务定义提示
             sys_prompt_rw = "你是一个专家查询优化器。重写用户的查询以使其适合向量数据库检索。关注关键实体、概念和关系。删除对话性废话。仅输出重写的查询文本。"
             user_prompt_rw = f"原始查询：{test_query}\n\n重写的查询："
             
             # 调用LLM重写查询
             resp_rw = client.chat.completions.create(
                 model=NEBIUS_GENERATION_MODEL, # 也可以使用生成模型完成此任务
                 messages=[
                     {"role": "system", "content": sys_prompt_rw},
                     {"role": "user", "content": user_prompt_rw}
                 ],
                 temperature=0.1, # 低温度以进行集中重写
                 max_tokens=100, 
                 top_p=0.9
             )
             # 清理LLM的响应以仅获取查询文本
             candidate_q = resp_rw.choices[0].message.content.strip()
             # 删除潜在的"重写的查询："或"查询："前缀
             candidate_q = re.sub(r'^(重写的查询：|查询：)\s*', '', candidate_q, flags=re.IGNORECASE).strip('"')
             
             # 仅当重写的查询合理不同且不太短时才使用它
             if candidate_q and len(candidate_q) > 5 and candidate_q.lower() != test_query.lower(): 
                 rewritten_q = candidate_q
                 # print(f"      使用重写的查询：'{rewritten_q}'") # 取消注释以进行详细日志记录
             # else: 
                 # print("      重写失败、太短或与原始相同。使用原始查询。") # 取消注释以进行详细日志记录
        except Exception as e:
             print(f"    警告：查询重写时出错：{e}。使用原始查询。")
             rewritten_q = test_query # 出错时回退到原始查询
             
        # 使用（可能）重写的查询进行检索来运行评估
        result_rewrite = run_and_evaluate("Query Rewrite RAG", rewritten_q, top_k)
        all_results.append(result_rewrite)

        # 策略3：重排序RAG（模拟）
        # 使用原始查询进行检索，但模拟重排序过程
        result_rerank = run_and_evaluate("Rerank RAG (Simulated)", test_query, top_k, use_simulated_rerank=True)
        all_results.append(result_rerank)

    print("\n=== RAG实验循环完成 ===")
    print("-" * 25)

# --- 9. 结果分析和可视化 ---
print("\n=== 开始结果分析 ===")

# 将结果转换为DataFrame以便于分析
df_results = pd.DataFrame(all_results)

# 基本统计信息
print("\n基本统计信息：")
print(f"总实验运行次数：{len(df_results)}")
print(f"成功运行次数：{len(df_results[df_results['avg_score'] > 0])}")
print(f"失败运行次数：{len(df_results[df_results['avg_score'] == 0])}")

# 按策略分组的平均分数
print("\n按策略分组的平均分数：")
strategy_scores = df_results.groupby('strategy')['avg_score'].agg(['mean', 'std', 'min', 'max'])
print(strategy_scores)

# 找出最佳配置
best_config = df_results.loc[df_results['avg_score'].idxmax()]
print("\n最佳配置：")
print(f"策略：{best_config['strategy']}")
print(f"分块大小：{best_config['chunk_size']}")
print(f"重叠度：{best_config['overlap']}")
print(f"Top-K：{best_config['top_k']}")
print(f"平均分数：{best_config['avg_score']:.3f}")
print(f"忠实度：{best_config['faithfulness']:.3f}")
print(f"相关性：{best_config['relevancy']:.3f}")
print(f"相似度：{best_config['similarity_score']:.3f}")
print(f"运行时间：{best_config['time_sec']:.2f}秒")

# 保存结果到CSV文件
output_file = "rag_experiment_results.csv"
df_results.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n结果已保存到：{output_file}")

# 可视化结果
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 1. 策略性能对比
    plt.subplot(2, 2, 1)
    sns.boxplot(x='strategy', y='avg_score', data=df_results)
    plt.title('不同RAG策略的性能对比')
    plt.xticks(rotation=45)
    plt.ylabel('平均分数')
    
    # 2. 分块大小vs性能
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df_results, x='chunk_size', y='avg_score', hue='strategy')
    plt.title('分块大小对性能的影响')
    plt.xlabel('分块大小')
    plt.ylabel('平均分数')
    
    # 3. 重叠度vs性能
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df_results, x='overlap', y='avg_score', hue='strategy')
    plt.title('重叠度对性能的影响')
    plt.xlabel('重叠度')
    plt.ylabel('平均分数')
    
    # 4. Top-K vs性能
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df_results, x='top_k', y='avg_score', hue='strategy')
    plt.title('Top-K对性能的影响')
    plt.xlabel('Top-K')
    plt.ylabel('平均分数')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('rag_experiment_results.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存到：rag_experiment_results.png")
    
except Exception as e:
    print(f"生成可视化时出错：{e}")

print("\n=== 结果分析完成 ===")
print("-" * 25)