# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0
#milvus-server
import numpy as np
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
def search_vector(query):
    # 0. 连接服务器（同上）
    connections.connect(alias="default", host="localhost", port="19530")



    # 定义集合Schema，并描述集合
    from pymilvus import utility
    collection_name = "shediao_heroes_collection"
    collection = Collection(collection_name)

    print(f"集合 '{collection_name}' 寻找成功！")

    # 3. 创建索引（这是高效向量搜索的关键）
    # 在插入数据前或后创建索引都可以，但之后需要手动加载集合到内存
    index_params = {
        "index_type": "IVF_FLAT",  # 一种高效的近似搜索索引
        "metric_type": "L2",       # 使用L2距离（欧氏距离）作为相似度度量。如果你的向量是归一化的，也可以用"IP"（内积）
        "params": {"nlist": 1024}, # 索引参数：聚类中心数量
    }





    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # together with setting `padding_side` to "left":
    # model = SentenceTransformer(
    #     "Qwen/Qwen3-Embedding-0.6B",
    #     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
    #     tokenizer_kwargs={"padding_side": "left"},
    # )






    # 5. 将集合加载到内存（非常重要！搜索前必须做这一步）
    collection.load()
    print("集合已加载至内存，准备搜索。")

    # 6. 执行向量搜索
    # 模拟一个查询向量（比如，查询“关于郭靖吃饭的情节”）
    from sentence_transformers import SentenceTransformer
    import torch
    # Load the model
    # 检查是否有可用的 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载模型并指定设备
    model = SentenceTransformer("../models/Qwen3-Embedding-0.6B", device=device)
    query=query
    query_vector = model.encode(query, prompt_name="query") # 请替换为你的真实查询向量
    print(121212121,query_vector)
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
        limit=20,
        output_fields=output_fields
    )

    # 7. 处理并打印搜索结果
    result=[]
    print(f"\n=== 搜索结果  {query}===")
    for i, hits in enumerate(results):
        print(f"对于第 {i+1} 个查询向量：")
        for j, hit in enumerate(hits):
            print(f"  结果 {j+1}: ")
            result.append(hit.entity.get('text_chunk'))
            print(f"    文本: {hit.entity.get('text_chunk')}")
            print(f"    章节: {hit.entity.get('chapter')}")
            print(f"    人物: {hit.entity.get('characters')}")
            print(f"    距离: {hit.distance:.4f} (距离越小越相似)")
            print("  ---")
    # 8. （可选）完成后释放内存
    collection.release()
    return result

def cross_encoder(query,result):
    from sentence_transformers import CrossEncoder

    # 初始化模型
    model = CrossEncoder('../models/mxbai-rerank-large-v2',max_length=512)
    sentence_pairs = []
    # 准备数据：列表形式，每个元素是 [query, document] 对
    for re in result:
        sentence_pairs.append([query])
        sentence_pairs[-1].append(re)

    # 批量预测分数
    scores=[]
    for sentence_pair in sentence_pairs:
        score = model.predict([sentence_pair])[0]  # 每次只处理一个句子对
        scores.append(score)

    # 组合结果并排序
    results = list(zip(result, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    print("Sentence-Transformers 重排序结果:")
    for i, (doc, score) in enumerate(results[:5]):
        print(f"{i + 1}. 分数: {score:.4f}")
        print(f"   文本: {doc}...")
        print()
    return result[:3]
if __name__ == "__main__":
    query="郭靖的爹是谁"
    cross_encoder(query,search_vector([query]))