"""使用 PyTRIO 对 JSONL 对话数据执行 LoRA SFT。"""

from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import swanlab
from dotenv import dotenv_values
from loguru import logger as log
from tqdm import tqdm

if TYPE_CHECKING:
    import pytrio as trio


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


@dataclass(frozen=True)
class TrainingExample:
    """单条监督微调数据：PyTRIO Datum、完整 token 序列和 assistant loss mask。"""

    datum: trio.Datum
    tokens: list[int]
    weights: np.ndarray


@dataclass(frozen=True)
class TrainingConfig:
    """训练运行所需配置；所有可配置字段均来自项目 `.env`。"""

    # PyTRIO 远程训练使用的基础模型标识。
    model_name_or_path: str
    # 微调方式；当前脚本仅支持 lora。
    finetuning_type: str
    # LoRA 低秩矩阵的秩，传给 PyTRIO 创建训练客户端。
    lora_rank: int
    # PyTRIO 保存的完整训练 state 路径；为空时从基础模型开始。
    resume_state_path: str | None
    # Adam 优化器的基础学习率。
    learning_rate: float
    # 完整遍历训练集的次数。
    num_train_epochs: int
    # 每次提交给 PyTRIO 计算梯度的训练样本数。
    per_device_train_batch_size: int
    # 累计多少个训练批次的梯度后执行一次参数更新。
    gradient_accumulation_steps: int
    # 学习率调度方式：constant、none 或 cosine。
    lr_scheduler_type: str
    # 学习率线性预热占总参数更新步数的比例。
    warmup_ratio: float
    # Dev 验证间隔占总参数更新步数的比例。
    eval_step_ratio: float
    #梯度裁剪
    grad_clip_norm:float
    #权重衰减
    weight_decay:float
    # 每条对话允许的最大 token 数；过长时优先截断 prompt。
    cutoff_len: int
    # 训练集 JSONL 路径；只用于参数更新。
    train_data_file: Path
    # Dev 集 JSONL 路径；只计算验证 loss，不参与反向传播。
    val_data_file: Path
    # 每轮保存的 LoRA 权重名称前缀。
    weights_name: str
    # Dev loss 连续多少次验证不下降后早停；0 表示关闭早停。
    early_stopping_patience: int
    # LoRA 缩放系数；当前 PyTRIO 示例接口未使用，仅作为配置元数据。
    lora_alpha: str | None
    # LoRA 目标模块；当前 PyTRIO 示例接口未使用，仅作为配置元数据。
    lora_target: str | None
    # 数据集标识；脚本直接读取文件，此字段当前仅作为配置元数据。
    dataset: str | None
    # 验证集划分比例；训练与 Dev 已分开存储，此字段不会重新划分数据。
    val_size: str | None


def step_1_load_config(env_file: Path) -> TrainingConfig:
    """输入：`.env` 文件路径 `Path`；输出：完成类型转换的 `TrainingConfig`。

    读取训练和 Dev 配置，但不读取或访问 Holdout 文件路径。
    """
    values = dotenv_values(env_file)

    def required(key: str) -> str:
        value = values.get(key)
        if value is None or not value.strip():
            raise ValueError(f"缺少必需环境变量 {key}: {env_file}")
        return value.strip()

    return TrainingConfig(
        model_name_or_path=required("MODEL_NAME_OR_PATH"),
        finetuning_type=required("FINETUNING_TYPE").lower(),
        lora_rank=int(required("LORA_RANK")),
        resume_state_path=(values.get("RESUME_STATE_PATH") or "").strip() or None,
        learning_rate=float(required("LEARNING_RATE")),
        num_train_epochs=int(required("NUM_TRAIN_EPOCHS")),
        per_device_train_batch_size=int(required("PER_DEVICE_TRAIN_BATCH_SIZE")),
        gradient_accumulation_steps=int(required("GRADIENT_ACCUMULATION_STEPS")),
        lr_scheduler_type=required("LR_SCHEDULER_TYPE").lower(),
        warmup_ratio=float(required("WARMUP_RATIO")),
        eval_step_ratio=float(required("EVAL_STEP_RATIO")),
        cutoff_len=int(required("CUTOFF_LEN")),
        train_data_file=Path(required("TRAIN_DATA_FILE")).expanduser(),
        val_data_file=Path(required("VAL_DATA_FILE")).expanduser(),
        weights_name=required("WEIGHTS_NAME"),
        early_stopping_patience=int(required("EARLY_STOPPING_PATIENCE")),
        lora_alpha=values.get("LORA_ALPHA"),
        lora_target=values.get("LORA_TARGET"),
        dataset=values.get("DATASET"),
        val_size=values.get("VAL_SIZE"),
        grad_clip_norm=float(required("GRAD_CLIP_NORM")),
        weight_decay=float(required("WEIGHT_DECAY")),
    )


def step_2_validate_config(config: TrainingConfig) -> None:
    """输入：`TrainingConfig`；输出：校验通过时返回 `None`，否则抛出异常。

    提前拒绝当前脚本及 PyTRIO SFT 接口无法兑现的训练设置，避免静默忽略。
    """
    if config.finetuning_type != "lora":
        raise ValueError("FINETUNING_TYPE 目前必须为 lora")
    if config.lora_rank <= 0 or config.learning_rate <= 0:
        raise ValueError("LORA_RANK 和 LEARNING_RATE 必须大于 0")
    if config.num_train_epochs <= 0 or config.per_device_train_batch_size <= 0:
        raise ValueError("训练轮数和 batch size 必须大于 0")
    if config.cutoff_len < 2:
        raise ValueError("CUTOFF_LEN 必须至少为 2")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("GRADIENT_ACCUMULATION_STEPS 必须大于 0")
    if config.lr_scheduler_type not in {"constant", "none", "cosine"}:
        raise ValueError("LR_SCHEDULER_TYPE 只支持 constant、none 或 cosine")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("WARMUP_RATIO 必须在 [0, 1) 范围内")
    if not 0 < config.eval_step_ratio <= 1:
        raise ValueError("EVAL_STEP_RATIO 必须在 (0, 1] 范围内")
    if config.early_stopping_patience < 0:
        raise ValueError("EARLY_STOPPING_PATIENCE 不能为负数")
    for label, path in (
        ("TRAIN_DATA_FILE", config.train_data_file),
        ("VAL_DATA_FILE", config.val_data_file),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 指向的文件不存在：{path}")
    if config.train_data_file.resolve() == config.val_data_file.resolve():
        raise ValueError("训练集和 Dev 集必须是不同文件")

    unsupported = {
        "LORA_ALPHA": config.lora_alpha,
        "LORA_TARGET": config.lora_target,
        "DATASET": config.dataset,
        "VAL_SIZE": config.val_size,
    }
    configured = [f"{key}={value!r}" for key, value in unsupported.items() if value]
    if configured:
        warnings.warn(
            "以下 .env 参数仅作为配置元数据，不由当前 PyTRIO API 控制："
            + ", ".join(configured),
            stacklevel=2,
        )


def step_3_load_jsonl(path: Path) -> list[list[dict[str, str]]]:
    """读取逐行 JSONL 或多行格式化 JSON，并验证 messages 结构。"""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    conversations: list[list[dict[str, str]]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        try:
            record, offset = decoder.raw_decode(text, offset)
        except json.JSONDecodeError as error:
            line_number = text.count("\n", 0, error.pos) + 1
            raise ValueError(f"{path}:{line_number} JSON 格式错误：{error}") from error
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"{path}:每条记录必须包含至少两条 messages")
        validated: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"{path}:message 必须为对象")
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ValueError(f"{path}:message role/content 不合法")
            validated.append({"role": role, "content": content})
        if validated[-1]["role"] != "assistant":
            raise ValueError(f"{path}:每条记录最后一条消息必须是 assistant")
        conversations.append(validated)
    if not conversations:
        raise ValueError(f"数据集为空：{path}")
    return conversations


def step_4_build_example(
    messages: list[dict[str, str]], tokenizer: Any, cutoff_len: int
) -> TrainingExample:
    """输入：消息列表、PyTRIO tokenizer、最大 token 数；输出：`TrainingExample`。

    使用 chat template 编码上下文，只对最后一条 assistant 回复及 EOS 计算 loss；
    超长时保留完整 completion 和最近的 prompt token，保证序列不超过 cutoff。
    """
    import pytrio as trio

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_tokens = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        completion_tokens.append(eos_token_id)
    if not completion_tokens:
        raise ValueError("assistant completion 编码后为空")
    if len(completion_tokens) >= cutoff_len:
        raise ValueError("assistant completion 超过 CUTOFF_LEN，无法保留 prompt；请增大长度")
    prompt_tokens = prompt_tokens[-(cutoff_len - len(completion_tokens)) :]
    if not prompt_tokens:
        raise ValueError("prompt 编码后为空，请检查 chat template 和 CUTOFF_LEN")

    tokens = [int(token) for token in prompt_tokens + completion_tokens]
    all_weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    loss_weights = np.asarray(all_weights[1:], dtype=np.float32)
    target_tokens = np.asarray(tokens[1:], dtype=np.int32)
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={"weights": loss_weights, "target_tokens": target_tokens},
    )
    return TrainingExample(datum=datum, tokens=tokens, weights=loss_weights)


def step_5_prepare_dataset(
    conversations: list[list[dict[str, str]]], tokenizer: Any, cutoff_len: int
) -> list[TrainingExample]:
    """输入：消息对话列表、tokenizer、最大长度；输出：`list[TrainingExample]`。

    将已校验的 JSONL 对话转换为 PyTRIO Datum，拒绝没有可监督 token 的数据集。
    """
    examples = [step_4_build_example(messages, tokenizer, cutoff_len) for messages in conversations]
    prepared = [example for example in examples if float(example.weights.sum()) > 0]
    if not prepared:
        raise ValueError("数据集中没有可监督的 assistant token")
    return prepared


def step_6_create_clients(
    config: TrainingConfig,
) -> tuple[Any, Any, Any]:
    """输入：`TrainingConfig`；输出：`(service_client, training_client, tokenizer)`。

    配置 state 路径时恢复训练客户端，否则从基础模型新建 LoRA 客户端。
    """
    import pytrio as trio

    service_client = trio.ServiceClient()
    print(config.resume_state_path)
    if config.resume_state_path:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            path=config.resume_state_path,
        )
        log.info(f'续训成功，权重：{config.resume_state_path}')
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name_or_path,
            rank=config.lora_rank,
        )
    tokenizer = training_client.get_tokenizer()
    return service_client, training_client, tokenizer


def step_7_learning_rate(
    config: TrainingConfig, update_index: int, total_updates: int
) -> float:
    """输入：配置、从 1 开始的更新序号及总步数 `int`；输出：本次 Adam 学习率 `float`。

    在本地计算 warmup 和 cosine 衰减，然后将学习率传给 PyTRIO 优化器。
    """
    warmup_updates = math.ceil(total_updates * config.warmup_ratio)
    if update_index <= warmup_updates:
        return config.learning_rate * update_index / warmup_updates
    if config.lr_scheduler_type == "cosine" and total_updates > warmup_updates:
        progress = (update_index - warmup_updates) / (total_updates - warmup_updates)
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    return config.learning_rate


def step_8_evaluate_dev(
    service_client: Any,
    config: TrainingConfig,
    dev_examples: list[TrainingExample],
    weights_path: str,
) -> float:
    """输入：PyTRIO service、配置、Dev 样本及权重路径；输出：Dev token 平均 loss `float`。

    通过带当前 LoRA 权重的 sampling client 计算 teacher-forced logprobs；不调用训练反传接口。
    """
    import pytrio as trio

    sampling_client = service_client.create_sampling_client(
        base_model=config.model_name_or_path,
        model_path=weights_path,
    )
    total_nll = 0.0
    total_weight = 0.0
    for example in dev_examples:
        values = sampling_client.compute_logprobs(
            trio.ModelInput.from_ints(example.tokens)
        ).result()
        shifted_logprobs = values[1:]
        if len(shifted_logprobs) != len(example.weights):
            raise ValueError("Dev logprobs 与目标 token 长度不一致")
        for logprob, weight in zip(shifted_logprobs, example.weights, strict=True):
            if weight and (logprob is None or not math.isfinite(float(logprob))):
                raise ValueError("Dev 集监督位置出现无效 logprob")
            if weight:
                total_nll -= float(logprob) * float(weight)
                total_weight += float(weight)
    if total_weight == 0:
        raise ValueError("Dev 集没有有效的 assistant token logprobs")
    return total_nll / total_weight


def step_9_tensor_data_to_numpy(value: Any) -> np.ndarray:
    """把 PyTRIO TensorData、NumPy 数组或普通列表转换为一维 NumPy 数组。

    输入：张量数据对象，例如 `TensorData`、`np.ndarray` 或 `[0.1, 0.2]`。
    输出：`np.ndarray`；例如 `TensorData.to_numpy() -> array([0.1, 0.2])`。
    """
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def step_9_train(
    service_client: Any,
    training_client: Any,
    config: TrainingConfig,
    train_examples: list[TrainingExample],
    dev_examples: list[TrainingExample],
) -> str:
    """输入：PyTRIO clients、配置、训练/Dev 样本；输出：已保存 LoRA 权重路径 `str`。

    仅训练集进入 forward_backward；按更新步数间隔保存权重并计算只读 Dev loss。
    """
    import pytrio as trio

    batches_per_epoch = math.ceil(len(train_examples) / config.per_device_train_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.num_train_epochs
    eval_steps = max(1, math.ceil(total_updates * config.eval_step_ratio))
    print(f"total_steps={total_updates} eval_steps={eval_steps}")
    global_step = 0
    best_loss = float("inf")
    best_weights_path = ""
    validations_without_improvement = 0
    stop_training = False
    rng = random.Random(42)
    for epoch in tqdm(range(config.num_train_epochs)):
        epoch_examples = train_examples.copy()
        rng.shuffle(epoch_examples)
        micro_batch_index = 0
        for start in range(0, len(epoch_examples), config.per_device_train_batch_size):
            batch = epoch_examples[start : start + config.per_device_train_batch_size]
            fwdbwd_future = training_client.forward_backward(
                [example.datum for example in batch], loss_fn="cross_entropy"
            )
            fwdbwd_result = fwdbwd_future.result()
            logprobs = np.concatenate(
                [
                    step_9_tensor_data_to_numpy(output["logprobs"])
                    for output in fwdbwd_result.loss_fn_outputs
                ]
            )
            weights = np.concatenate([example.weights for example in batch])
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                raise ValueError("训练 batch 没有 assistant loss token")
            train_loss = -float(np.dot(logprobs, weights) / weight_sum)
            micro_batch_index += 1
            should_update = (
                micro_batch_index % config.gradient_accumulation_steps == 0
                or micro_batch_index == batches_per_epoch
            )
            if should_update:
                global_step += 1
                learning_rate = step_7_learning_rate(config, global_step, total_updates)
                training_client.optim_step(
                    trio.AdamParams(learning_rate=learning_rate,grad_clip_norm=config.grad_clip_norm,weight_decay=config.weight_decay)
                ).result()
            metrics = {
                "train/loss": train_loss,
                "train/epoch": epoch + 1,
                "train/optimizer_step": global_step,
            }
            if should_update:
                metrics["train/learning_rate"] = learning_rate
            swanlab.log(metrics)
            print(
                f"epoch={epoch + 1}/{config.num_train_epochs} "
                f"step={global_step}/{total_updates} "
                f"train_loss={train_loss:.6f}"
            )
            if should_update and (global_step % eval_steps == 0 or global_step == total_updates):
                weights = training_client.save_weights_for_sampler(
                    name=f"{config.weights_name}-step-{global_step}"
                ).result()
                dev_loss = step_8_evaluate_dev(
                    service_client, config, dev_examples, str(weights.path)
                )
                swanlab.log({"dev/loss": dev_loss})
                print(f"step={global_step}/{total_updates} dev_loss={dev_loss:.6f}")
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_weights_path = str(weights.path)
                    validations_without_improvement = 0
                else:
                    validations_without_improvement += 1
                    if (
                        config.early_stopping_patience
                        and validations_without_improvement >= config.early_stopping_patience
                    ):
                        print(f"Dev loss 连续 {validations_without_improvement} 次验证未改善，提前停止")
                        stop_training = True
                        break

        state = training_client.save_state(
            name=f"{config.weights_name}-epoch-{epoch + 1}-state"
        ).result()
        print(f"epoch={epoch + 1} saved_train_state_path={state.path}")
        if stop_training:
            break
    return best_weights_path


def step_10_main() -> None:
    """输入：无；输出：完成训练、Dev 监控并打印 PyTRIO 权重路径。

    串联配置检查、训练/Dev 数据准备、PyTRIO SFT 和按步 Dev loss 监控；不触碰 Holdout。
    """
    config = step_1_load_config(ENV_FILE)
    step_2_validate_config(config)
    train_conversations = step_3_load_jsonl(config.train_data_file)
    dev_conversations = step_3_load_jsonl(config.val_data_file)
    service_client, training_client, tokenizer = step_6_create_clients(config)
    raw_messages = train_conversations[0]
    rendered_prompt = tokenizer.apply_chat_template(
        raw_messages[:-1], tokenize=False, add_generation_prompt=True
    )
    # log.info(
    #     "首条训练样本（tokenizer 转换前）：\n{}",
    #     json.dumps(
    #         {
    #             "raw_messages": raw_messages,
    #             "raw_prompt_messages": raw_messages[:-1],
    #             "raw_assistant_target": raw_messages[-1]["content"],
    #             "chat_template_prompt": rendered_prompt,
    #         },
    #         ensure_ascii=False,
    #         indent=2,
    #     ),
    # )
    train_examples = step_5_prepare_dataset(train_conversations, tokenizer, config.cutoff_len)
    dev_examples = step_5_prepare_dataset(dev_conversations, tokenizer, config.cutoff_len)
    print(f"train_examples={len(train_examples)} dev_examples={len(dev_examples)}")
    example = train_examples[0]
    input_tokens = example.tokens[:-1]
    target_tokens = example.datum.loss_fn_inputs["target_tokens"].tolist()
    loss_weights = example.weights.tolist()
    supervised_tokens = [
        token for token, weight in zip(target_tokens, loss_weights, strict=True) if weight > 0
    ]
    # log.info(
    #     "首条训练样本（编码后的模型输入、target 和 loss mask）：\n{}",
    #     json.dumps(
    #         {
    #             "model_input_token_ids": input_tokens,
    #             "model_input_text": tokenizer.decode(input_tokens, skip_special_tokens=False),
    #             "target_token_ids": target_tokens,
    #             "target_text": tokenizer.decode(target_tokens, skip_special_tokens=False),
    #             "loss_weights": loss_weights,
    #             "supervised_target_text": tokenizer.decode(
    #                 supervised_tokens, skip_special_tokens=False
    #             ),
    #         },
    #         ensure_ascii=False,
    #         indent=2,
    #     ),
    # )
    run = swanlab.init(
        project="xiaoan-brain-sft-1114",
        config={
            "model_name_or_path": config.model_name_or_path,
            "finetuning_type": config.finetuning_type,
            "lora_rank": config.lora_rank,
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "lr_scheduler_type": config.lr_scheduler_type,
            "warmup_ratio": config.warmup_ratio,
            "eval_step_ratio": config.eval_step_ratio,
            "cutoff_len": config.cutoff_len,
            "train_data_file": str(config.train_data_file),
            "val_data_file": str(config.val_data_file),
            'grad_clip_norm':config.grad_clip_norm,
            'weight_decay':config.weight_decay,
            'early_stopping_patience':config.early_stopping_patience,
        },
    )
    try:
        weights_path = step_9_train(
            service_client, training_client, config, train_examples, dev_examples
        )
    except Exception as error:
        run.finish(state="crashed", error=str(error))
        raise
    else:
        run.finish()
    print(f"saved_weights_path={weights_path}")


if __name__ == "__main__":
    step_10_main()
