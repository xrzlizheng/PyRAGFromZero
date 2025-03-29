# 所有RAG技术：让算法工程师不再秃头 ✨

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-31016/) [![Nebius AI](https://img.shields.io/badge/Nebius%20AI-API-brightgreen)](https://cloud.nebius.ai/services/llm-embedding) [![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/) [![CSDN](https://img.shields.io/badge/CSDN-Blog-black?logo=csdn)](https://lizheng.blog.csdn.net/)

作为一名算法工程师，你是否经常在深夜对着电脑屏幕发呆，思考着如何让RAG系统更智能？别担心，这个仓库就是为你准备的！我们采用清晰、实用的方法来实现**检索增强生成（RAG）**，将先进的技术分解为简单易懂的实现。这里不使用`LangChain`或`FAISS`等框架，而是使用熟悉的Python库如`openai`、`numpy`、`matplotlib`等。

我们的目标是提供可读、可修改且具有教育意义的代码，让你在享受咖啡的同时也能理解RAG的精髓。通过关注基础知识，本项目有助于揭开RAG的神秘面纱，让你在下次团队会议中也能侃侃而谈，而不是只能默默点头。

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Nebius AI](https://img.shields.io/badge/Nebius%20AI-API-brightgreen)](https://cloud.nebius.ai/services/llm-embedding) [![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/) [![CSDN](https://img.shields.io/badge/CSDN-Blog-black?logo=csdn)](https://lizheng.blog.csdn.net/)

本仓库采用清晰、实用的方法来实现**检索增强生成（RAG）**，将先进的技术分解为简单易懂的实现。这里不使用`LangChain`或`FAISS`等框架，而是使用熟悉的Python库如`openai`、`numpy`、`matplotlib`等。

目标是提供可读、可修改且具有教育意义的代码。通过关注基础知识，本项目有助于揭开RAG的神秘面纱，使其更容易理解。


## 🚀 内容概览：从菜鸟到大神的进阶之路

本仓库包含一系列Jupyter Notebook，每个笔记本专注于一种特定的RAG技术。每个笔记本提供：

*   技术的简要说明。
*   从零开始的逐步实现。
*   带有内联注释的清晰代码示例。
*   评估和比较以展示技术的有效性。
*   可视化结果。



## 💡 核心概念

*   **嵌入：**  捕捉语义的文本数值表示。我们使用Nebius AI的嵌入API，在许多笔记本中也使用`BAAI/bge-en-icl`嵌入模型。

*   **向量存储：**  用于存储和搜索嵌入的简单数据库。我们使用NumPy创建了自己的`SimpleVectorStore`类，以实现高效的相似性计算。

*   **余弦相似度：**  两个向量之间相似度的度量。值越高表示相似度越高。

*   **分块：**  将文本分割为更小、更易管理的部分。我们探索了各种分块策略。

*   **检索：**  为给定查询找到最相关的文本分块的过程。

*   **生成：**  使用大型语言模型（LLM）根据检索到的上下文和用户查询创建响应。我们通过Nebius AI的API使用`meta-llama/Llama-3.2-3B-Instruct`模型。

*   **评估：**  通过将RAG系统的响应与参考答案进行比较或使用LLM评分相关性来评估其质量。

## 🤝 贡献

欢迎贡献！# PyRAGFromZero
