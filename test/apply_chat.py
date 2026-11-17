from transformers import AutoTokenizer

# ==============================================
# 说明：只下载tokenizer配置文件，不下载4B模型权重！
# 依赖：pip install transformers
# 国内建议配置HF镜像加速，否则第一次拉tokenizer很慢
# ==============================================
# huggingface仓库ID，直接在线加载tokenizer
HF_REPO_ID = "Qwen/Qwen2.5-3B-Instruct"

# 加载tokenizer（只加载词表+chat_template，不加载模型参数）
tokenizer = AutoTokenizer.from_pretrained(
    HF_REPO_ID,
    trust_remote_code=False,  # Qwen系列必须开启
    # local_files_only=False,  # 默认False，在线拉取
    # cache_dir="./hf_cache"  # 可选：指定缓存目录，避免重复下载tokenizer
)

# -------------------------- 模拟SFT一条对话样本 --------------------------
# 标准对话结构：user提问 + assistant标准答案
messages = [
    {"role": "user", "content": "系统从哪类服务器同步时间？"},
    {"role": "assistant", "content": "没有可信和可选的主体，请明确以后再次提问!!"}
]
print("【原始messages对话列表】")
print(messages, "\n")


# =============================================================================
# ✅ 就是你问的这一行核心代码：推理prompt构造
# messages[:-1]
#    切片：丢掉最后一条assistant消息，只保留前面历史（这里只有user）
# tokenize=False
#    不转token id，返回拼接好的原始字符串文本（方便肉眼核对模板）
# add_generation_prompt=True
#    自动在末尾追加：<|im_start|>assistant\n
#    作用：告诉模型接下来要输出assistant回答，推理时用！
# =============================================================================
prompt_text = tokenizer.apply_chat_template(
    messages[:-1],
    tokenize=True,
    add_generation_prompt=False#messages 最后是 user + `add_generation_prompt=True`：追加 `<|im_start|>assistant\n`（推理用）
)
# <|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
print("=" * 70)
print("【① 推理prompt｜messages[:-1] + add_generation_prompt=True】")
print("prompt_text（repr，能看见换行符）：")
print(repr(prompt_text))#<im_start>system
print("prompt_text原样：")
# <|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
# 系统从哪类服务器同步时间？<|im_end|>
# <|im_start|>assistant

# <|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
# 系统从哪类服务器同步时间？<|im_end|>
print(prompt_text)
print("👉 用途：推理时喂给模型，模型从这里开始续写回答\n")


# =============================================================================
# 对比版本A：add_generation_prompt=False，不会自动追加assistant头
# =============================================================================
prompt_no_gen = tokenizer.apply_chat_template(
    messages[:-1],
    tokenize=False,
    add_generation_prompt=False
)
print("=" * 70)
print("【② 对比｜messages[:-1] + add_generation_prompt=False】")
print(repr(prompt_no_gen))
print("👉 用途：一般不用，自己手动拼接模板时才用\n")


# =============================================================================
# 对比版本B：SFT训练【完整整条对话】（重点！和你loss mask强绑定）
# 传入全部messages，不切片[:-1]，add_generation_prompt=False
# 输出：user + assistant回答 + 自动补 <|im_end|>
# 这就是训练时 model_input_text 的原始文本
# =============================================================================
full_chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
print("=" * 70)
print("【③ SFT训练完整文本｜全部messages，不加generation prompt】")
print(repr(full_chat_text))
print("👉 用途：dataset里拼接整条对话，后续转input_ids、做shift label、loss mask\n")


# =============================================================================
# 转token id（只做编码，不跑模型）
# =============================================================================
# 推理prompt编码
prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
print("=" * 70)
print("【推理prompt token ids】")
print(prompt_ids)
print(f"长度：{len(prompt_ids)}")

# 训练完整文本编码 → 对应你训练日志里 model_input_token_ids
full_input_ids = tokenizer(full_chat_text, return_tensors="pt").input_ids[0].tolist()
print("\n【训练完整文本token ids（model_input_token_ids）】")
print(full_input_ids)
print(f"长度：{len(full_input_ids)}")


# =============================================================================
# 🚨 【重点：修复你之前的mask错位bug的演示】
# 规则回顾：
# target_token_ids = full_input_ids[1:] + [eos_token_id]
# raw_mask：和full_input_ids等长，prompt区域0，assistant回答区域1
# loss_weights = raw_mask[1:] + [0.0]  # 必须跟着target一起右移！！
# =============================================================================
print("\n" + "="*70)
print("==== 移位label + loss_mask 对齐规则演示 ====")
eos_id = tokenizer.eos_token_id
target_token_ids = full_input_ids[1:] + [eos_id]
print('eos_id',eos_id)
print('full_input_ids',full_input_ids)
print('target_token_ids',target_token_ids)


print(f"full_input_ids长度={len(full_input_ids)}")
print(f"target_token_ids长度={len(target_token_ids)}")
print("⚠️ 核心坑：loss_weights不能直接拿原始mask，必须 raw_mask[1:] + [0.0]")
