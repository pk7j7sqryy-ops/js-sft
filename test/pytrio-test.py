import json

from openai import OpenAI

BASE_URL = "https://pytrio.com/api/openai/v1"
MODEL_PATH = "trio://run_3v8ibvfcnlxk/sampler_weights/rag-alignment-sft-step-126"  # 权重路径或基模名称

api_key = "trio_gVzAIRL5rVdraJqNGe4PeOZtywGhXsRp8jg6U"  # 你的 PyTRIO API Key

client = OpenAI(
    base_url=BASE_URL,
    api_key=api_key,
)
with open("/Users/sunqiyu/AiLLM/document/all_project/all_project/09_llmprj/prj_train/xiaoan_brain/eval/rag-alignment/data/train/train-1116.jsonl","r",encoding="utf-8") as f:
    line = f.readline()  # 只读取第一行，读完就停

line=json.loads(line)
assistant_content = line["messages"][0]['content']#next(msg["messages"][0]['contente'] for msg in line if msg["messages"][0]['role'] == "role")
response = client.chat.completions.create(
    model=MODEL_PATH,
    messages=[{"role": "user", "content": f"{assistant_content}"}],
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

print(response.choices[0])
print(f"{response.choices[0].message.content}")