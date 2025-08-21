from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('../source-txt/sdyxz.txt','r',encoding='utf-8') as f:
    txt=f.read()
# 2. 创建递归文本分割器，并配置适合中文的参数
text_splitter = RecursiveCharacterTextSplitter(
    # 这是一个非常重要的参数！
    # 它定义了分片时优先使用的分隔符列表。列表顺序决定了分割的优先级。
    separators=[
        "\n\n",        # 优先级1: 双换行符 (章节内的自然场景切换)
        "\n",          # 优先级2: 单换行符 (段落结束)
        "。", "！", "？", # 优先级3: 中文句末标点 (句子结束)
        "；",           # 优先级4: 分号 (可选，根据文本情况决定)
        "，",           # 优先级5: 逗号 (尽量不在逗号处切分，这是最后的手段)
        " ",            # 优先级6: 空格 (对于中文，这个优先级应该放很低)
        "",             # 优先级7: 所有字符 (最终保底，按字符切)
    ],
    chunk_size=300,      # 每个分片的理想最大长度（以字符为单位）
    chunk_overlap=50,   # 分片之间重叠的字符数
    length_function=len, # 计算长度的方法，对于中文，直接用`len`即可
    is_separator_regex=False, # 分隔符是普通字符串，不是正则表达式
    keep_separator=True,     # 分割后保留分隔符，有助于维持上下文
    add_start_index=True,   # 可选：记录分片在原文中的起始位置，用于调试
)

# 3. 执行分片
chunks = text_splitter.create_documents([txt])

# 4. 打印结果
print(f"原始文本长度: {len(txt)} 字符")
print(f"分割后得到 {len(chunks)} 个分片\n")
print("="*50)

for i, chunk in enumerate(chunks):
    with open(f'../split-txt/{i}.txt','w',encoding='utf-8') as f:
        f.write(chunk.page_content)
    print(f"分片 {i+1} (长度: {len(chunk.page_content)} 字符):")
    print(chunk.page_content)
    print("-" * 40)