import transformers
import torch
from serach import search_vector,cross_encoder


from transformers import AutoModelForCausalLM, AutoTokenizer
query = '郭靖的师父是谁'
rag_knowledge=cross_encoder(query,search_vector([query]))
model_name = "../models/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    #device_map={'cuda':0}
    device_map="auto"
)

# prepare the model input
rag_word=''
for i in rag_knowledge:
    rag_word+='根据以下前置信息:'+i
rag_word=''
messages = [
    {"role": "user", "content": f"{rag_word}回答以下问题：{query}"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print('==========================================')
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
print('问题:',query)
print("思考文本:", thinking_content)
print("回答文本:", content)
#
# model_id = "../models/llama38b"
#
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )
# query='郭靖的爹是谁'
# rag_knowledge=cross_encoder(query,search_vector([query]))
# rag_word=''
# for i in rag_knowledge:
#     rag_word+='根据以下前置信息:'+i
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": f"{rag_word}回答以下问题：{query}"},
# ]
#
# outputs = pipeline(
#     messages,
#     max_new_tokens=256,
# )
# print(outputs[0]["generated_text"][-1])
