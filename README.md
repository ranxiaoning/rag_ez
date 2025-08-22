# 《射雕英雄传》RAG问答系统

这是一个基于RAG（检索增强生成）技术的问答系统，专门针对金庸武侠小说《射雕英雄传》的内容进行智能问答。

## 项目概述

本项目结合了现代自然语言处理技术和向量数据库，实现了对《射雕英雄传》小说内容的精准检索和智能问答。系统首先将小说文本分割成小块，然后使用嵌入模型转换为向量并存储在Milvus向量数据库中。当用户提问时，系统会检索相关文本片段，并使用大型语言模型生成准确回答。

## 功能特点

- **文本分割**: 使用递归字符文本分割器，针对中文文本特点优化分割策略
- **向量检索**: 使用Qwen3-Embedding-0.6B模型生成文本嵌入，Milvus向量数据库进行高效相似性搜索
- **重排序**: 使用mxbai-rerank-large-v2模型对检索结果进行精排序
- **智能问答**: 基于Qwen3-8B模型生成自然、准确的回答

## 技术栈

- **文本处理**: LangChain RecursiveCharacterTextSplitter
- **向量化**: SentenceTransformers, Qwen3-Embedding-0.6B
- **向量数据库**: Milvus
- **重排序**: CrossEncoder, mxbai-rerank-large-v2
- **语言模型**: Qwen3-8B

## 安装步骤

### 环境要求

- Python 3.8+
- PyTorch
- Transformers >= 4.51.0
- Sentence-Transformers >= 2.7.0
- Pymilvus
- CUDA（如使用GPU加速）

### 依赖安装

```bash
pip install torch transformers sentence-transformers pymilvus
```

### 模型下载

需要下载以下模型并放置在指定目录：

1. **Qwen3-Embedding-0.6B** → `../models/Qwen3-Embedding-0.6B/`
2. **mxbai-rerank-large-v2** → `../models/mxbai-rerank-large-v2/`
3. **Qwen3-8B** → `../models/Qwen3-8B/`

### Milvus数据库安装

请参考[Milvus官方文档](https://milvus.io/docs)安装并启动Milvus服务。

## 使用方法

### 1. 准备文本数据

将《射雕英雄传》文本文件放置在`../source-txt/sdyxz.txt`

### 2. 文本分割与向量化

运行文本分割脚本，将文本分割成小块并生成向量：

```bash
python text_splitter.py
```

### 3. 构建向量数据库

运行Milvus数据导入脚本：

```bash
python milvus_import.py
```

### 4. 运行问答系统

启动问答系统：

```bash
python qa_system.py
```

### 5. 提问示例

系统启动后，可以输入关于《射雕英雄传》的问题，例如：
- "郭靖的师父是谁？"
- "黄蓉为什么会喜欢郭靖？"
- "华山论剑的结果如何？"

## 项目结构

```
project/
├── source-txt/
│   └── sdyxz.txt              # 原始文本文件
├── split-txt/                 # 分割后的文本块
├── models/                    # 模型文件
│   ├── Qwen3-Embedding-0.6B/
│   ├── mxbai-rerank-large-v2/
│   └── Qwen3-8B/
├── text_splitter.py           # 文本分割脚本
├── milvus_import.py          # Milvus数据导入
├── search.py                 # 检索与重排序模块
└── qa_system.py              # 问答系统主程序
```

## 配置说明

### 文本分割参数

项目使用以下参数进行文本分割：
- `chunk_size=300`: 每个文本块的理想大小
- `chunk_overlap=50`: 块之间的重叠字符数
- 针对中文优化的分隔符优先级

### Milvus配置

- 集合名称: `shediao_heroes_collection`
- 向量维度: 1024 (适配Qwen3-Embedding-0.6B)
- 索引类型: IVF_FLAT
- 距离度量: L2

## 性能优化建议

1. **GPU加速**: 确保CU环境正确配置，以加速模型推理
2. **内存管理**: 处理大量数据时注意内存使用，适时释放集合
3. **索引优化**: 根据数据量调整Milvus的nlist和nprobe参数
4. **批量处理**: 对大量文本进行批量编码以提高效率

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 致谢

- 感谢金庸先生创作的经典武侠小说
- 感谢开源社区提供的各类NLP工具和模型
- 感谢Milvus团队提供的优秀向量数据库解决方案

---

如有问题或建议，请通过Issue提交反馈。
