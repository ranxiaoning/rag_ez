# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0
import numpy as np
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# 0. 连接服务器（同上）
connections.connect(alias="default", host="localhost", port="19530")

# 1. 定义集合的 Schema
# 我们需要存储：向量、文本内容、元数据（如章节、人物）
dimension = 1024  # 根据你选择的Embedding模型确定，例如Qwen3-Embedding-0.6B是1024维

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), # 主键，自增ID
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension) ,   # 向量字段
    FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=65535),     # 存储原始文本
    #FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=200),          # 元数据：章节
    #FieldSchema(name="characters", dtype=DataType.VARCHAR, max_length=500),       # 元数据：人物（用逗号分隔的字符串存储）
]

# 定义集合Schema，并描述集合
from pymilvus import utility


schema = CollectionSchema(
    fields=fields,
    description="《射雕英雄传》小说文本分片集合"
)

# 2. 创建集合
collection_name = "shediao_heroes_collection"
# 在创建集合之前，先检查并删除已存在的集合
# if utility.has_collection(collection_name):
#     utility.drop_collection(collection_name)

collection = Collection(
    name=collection_name,
    schema=schema,
    consistency_level="Strong" # 一致性级别
)

print(f"集合 '{collection_name}' 创建成功！")

# 3. 创建索引（这是高效向量搜索的关键）
# 在插入数据前或后创建索引都可以，但之后需要手动加载集合到内存
index_params = {
    "index_type": "IVF_FLAT",  # 一种高效的近似搜索索引
    "metric_type": "L2",       # 使用L2距离（欧氏距离）作为相似度度量。如果你的向量是归一化的，也可以用"IP"（内积）
    "params": {"nlist": 1024}, # 索引参数：聚类中心数量
}

# 为`embedding`字段创建索引
collection.create_index(
    field_name="embedding",
    index_params=index_params
)
print("索引创建成功！")

# 4. 准备并插入模拟数据
# 假设我们已经有了3个文本分片及其元数据和向量
from sentence_transformers import SentenceTransformer
import torch
# Load the model
# 检查是否有可用的 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载模型并指定设备
model = SentenceTransformer("../models/Qwen3-Embedding-0.6B", device=device)

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

documents = []
for i in range(4275):
    with open(f'../split-txt/{i}.txt','r') as f:
        documents.append(f.read())
document_embeddings = model.encode(documents)


num_entities = 4275
# 模拟文本和元数据


# 组织要插入的数据，必须与Schema中字段的顺序一致（除了自增主键id）
data_to_insert = [
    document_embeddings,    # 对应 embedding 字段
    documents,         # 对应 text_chunk 字段
    #fake_chapters,      # 对应 chapter 字段
    #fake_character_lists # 对应 characters 字段
]

# 插入数据
insert_result = collection.insert(data_to_insert)
print(f"数据插入成功！插入的ID为: {insert_result.primary_keys}")

# 5. 将集合加载到内存（非常重要！搜索前必须做这一步）
collection.load()
print("集合已加载至内存，准备搜索。")

# 6. 执行向量搜索
# 模拟一个查询向量（比如，查询“关于郭靖吃饭的情节”）
query_vector = np.random.random((1, dimension)).astype(np.float32) # 请替换为你的真实查询向量
search_params = {"metric_type": "L2", "params": {"nprobe": 10}} # 搜索参数：搜索的聚类中心数量

# 定义搜索时返回的字段
output_fields = ["text_chunk",
                 #"chapter",
                 #"characters"
                 ]

# 执行搜索，返回最相似的10个结果
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=10,
    output_fields=output_fields
)

# 7. 处理并打印搜索结果
print("\n=== 搜索结果 ===")
for i, hits in enumerate(results):
    print(f"对于第 {i+1} 个查询向量：")
    for j, hit in enumerate(hits):
        print(f"  结果 {j+1}: ")
        print(f"    文本: {hit.entity.get('text_chunk')}")
        print(f"    章节: {hit.entity.get('chapter')}")
        print(f"    人物: {hit.entity.get('characters')}")
        print(f"    距离: {hit.distance:.4f} (距离越小越相似)")
        print("  ---")

# 8. （可选）完成后释放内存
collection.release()
