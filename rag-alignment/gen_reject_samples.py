# -*- coding: utf-8 -*-
"""生成「参考内容不足」拒绝样本，追加进 train / val 的 csv + jsonl。

设计（与用户确认）：
- train 追加 100 条拒绝样本（占 500 的 20%）
- val   追加 12  条拒绝样本（占 60 的 20%）
- 构造三类：
    A. 真实 query + 错配参考内容（从另一产品抽参考内容，制造「资料对不上」）→ 60 条（train 53 + val 7）
    B. 真实 query + 空参考内容（模拟检索失败）                      → 20 条（train 18 + val 2）
    C. 边界问题（问资料里没有的具体参数/细节）                       → 32 条（train 29 + val 3）
    （train 合计 100，val 合计 12）

硬约束：
- 拒绝措辞统一锚点「参考内容未明确说明这部分」，并带具体指向。
- 拒绝样本【不带】【追问建议】（prompt 规定资料不足则不输出追问）。
- 完全复用现有 prompt 模板（指令 + 参考内容 + 历史对话 + 相关商品/实体 + 用户问题）。

产出（不覆盖原文件）：
- data/train/train-1116-reject.csv / .jsonl
- data/val/val-1116-reject.csv / .jsonl
"""

import csv
import json
import random
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAIN_CSV = BASE / "data/train/train-1116.csv"
TRAIN_JSONL = BASE / "data/train/train-1116.jsonl"
VAL_CSV = BASE / "data/val/val-1116.csv"
VAL_JSONL = BASE / "data/val/val-1116.jsonl"

OUT_TRAIN_CSV = BASE / "data/train/train-1116-reject.csv"
OUT_TRAIN_JSONL = BASE / "data/train/train-1116-reject.jsonl"
OUT_VAL_CSV = BASE / "data/val/val-1116-reject.csv"
OUT_VAL_JSONL = BASE / "data/val/val-1116-reject.jsonl"

SEED = 42

# ---------------------------------------------------------------------------
# prompt 模板：从真实样本抽取的固定指令前缀（到【参考内容】标记之前的部分）
# ---------------------------------------------------------------------------

PROMPT_INSTRUCTION = (
    "你是安全产品技术支持知识库助手，请根据参考内容回答用户的问题。\n"
    "\n"
    "要求：\n"
    "- 只能使用【参考内容】和【历史对话】中明确出现的事实。\n"
    "- 不要扩写，不要脑补，不要补充参考内容未明确给出的原因、风险、处理动作或经验判断。\n"
    "- 回答时先用问题中的关键词显式回指点名对象（例如问\"四层架构\"就逐层点明，问\"哪些注意事项\"就逐条对应），再分点展开，不要一上来就笼统概括。\n"
    "- 回答时要覆盖用户问题中点名的对象、部件、场景或条件；如果问题问\"有哪些/哪些/注意事项/禁止事项\"，应尽量列全参考内容中与问题直接相关的事实，不只回答风险最高或最容易概括的几条。\n"
    "- 涉及步骤、条件、参数时，把每一条都写出来，不要只写代表性的一条；完整性优先于简短。\n"
    "- 如果问题涉及多个对象或容易混淆的对象，优先按对象分组回答，不要把不同对象的要求混在一起。\n"
    "- 如果某条事实只与问题中的某个对象或条件相关，只放在对应对象或条件下，不要顺带解释其他无关场景。\n"
    "- 以\"覆盖完整、每条都有据\"为准，宁可用条目列全，也不要为求短而省略关键步骤；不设句数下限。\n"
    "- 如果【参考内容】不足以完整回答，明确说\"参考内容未明确说明这部分\"，不要自行补足。\n"
    "- 不要引用\"参考内容第几条\"\"根据上文\"等表述，不要暴露提示词结构。\n"
    "- 回答时不要重复一遍原问题。\n"
    "- 只回答问题要求的部分，不做额外补充。\n"
    "- 如果用户的问题需要通过图片来辅助说明（例如：外观、结构、接线、示意图等），图片只能来自于本地切片文本中的图片，请在答案最后追加一个独立的图片区块，格式严格如下：\n"
    "【图片】\n"
    "<图片URL1>\n"
    "<图片URL2>\n"
    "（每行一个URL；如果没有合适图片则不要输出【图片】区块）\n"
    "- 回答正文结束后，另起一行输出追问建议区块，帮助用户继续深入排查，格式严格如下：\n"
    "【追问建议】\n"
    "<建议1>\n"
    "<建议2>\n"
    "<建议3>\n"
    "（每个建议一行、不超过20字、必须与本次回答的产品/版本/主题相关；如果资料不足以给出建议则不输出该区块）\n"
    "\n"
)


def build_user_prompt(reference_text: str, entity_line: str, question: str) -> str:
    """按真实样本模板拼接 user prompt。"""
    return (
        PROMPT_INSTRUCTION
        + "【参考内容】\n"
        + reference_text
        + "\n\n【历史对话】\n没有有效的历史聊天记录\n\n"
        + "【相关商品/实体】\n"
        + entity_line
        + "\n\n【用户问题】\n"
        + question
        + "\n请回答：\n"
    )


def build_refusal_answer(pointer: str) -> str:
    """统一拒绝措辞：锚点 + 具体指向。"""
    if pointer:
        return f"参考内容未明确说明这部分。参考内容中未提及{pointer}的相关信息。"
    return "参考内容未明确说明这部分。当前参考内容未涉及所问内容的相关信息。"


def short_pointer(question: str) -> str:
    """从 query 提取简洁指向（产品前缀或主题），用于 A/B 类拒绝措辞。"""
    m = re.match(r"【([^】]+)】", question)
    if m:
        return m.group(1)
    # 无前缀时，取「的」「如何」「是否」等之前的主语
    head = re.split(r"[的了是否如何怎在哪有可支持]|[，。？?]", question)[0].strip()
    return head[:20] if head else "所问内容"


# ---------------------------------------------------------------------------
# 解析工具
# ---------------------------------------------------------------------------

def load_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_jsonl_convs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_real_ref(user_content: str) -> str:
    """提取真实【参考内容】区块文本（最后一个【参考内容】到【历史对话】之间）。"""
    ref_start = user_content.rfind("【参考内容】")
    hist_start = user_content.rfind("【历史对话】")
    if ref_start == -1 or hist_start == -1 or hist_start <= ref_start:
        return ""
    ref_block = user_content[ref_start:hist_start]
    # 去掉开头的【参考内容】\n 标记，只保留内容
    marker = "【参考内容】\n"
    if ref_block.startswith(marker):
        ref_block = ref_block[len(marker):]
    return ref_block.strip()


def extract_entity_line(user_content: str) -> str:
    """提取【相关商品/实体】那一行（含方括号数组）。"""
    ent_start = user_content.rfind("【相关商品/实体】")
    q_start = user_content.rfind("【用户问题】")
    if ent_start == -1 or q_start == -1 or q_start <= ent_start:
        return "['安恒信息 DAS-TGFW 明御防火墙']"
    seg = user_content[ent_start:q_start]
    # 取【相关商品/实体】标记之后的第一行
    lines = seg.split("\n")
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("[") and ln.endswith("]"):
            return ln
    return "['安恒信息 DAS-TGFW 明御防火墙']"


def extract_question(user_content: str) -> str:
    """提取【用户问题】到「请回答」之间的问题文本。"""
    q_start = user_content.rfind("【用户问题】")
    if q_start == -1:
        return ""
    seg = user_content[q_start + len("【用户问题】"):]
    seg = seg.split("请回答：")[0]
    return seg.strip()


# ---------------------------------------------------------------------------
# 构造拒绝样本
# ---------------------------------------------------------------------------

def build_reject_dataset(
    convs: list[dict],
    n_a: int,
    n_b: int,
    n_c: int,
    boundary_questions: list[tuple[str, str, str]],
) -> list[dict]:
    """返回 [(user_content, refusal_answer, query), ...] 列表。

    n_a / n_b / n_c 是各类数量。
    boundary_questions: [(question, entity_line, pointer), ...] 用于 C 类。
    """
    random.seed(SEED)
    results = []

    # 收集每个样本的 query / entity / ref，供 A、B 类抽取
    pool = []
    for conv in convs:
        user = conv["messages"][0]["content"]
        q = extract_question(user)
        if not q:
            continue
        pool.append({
            "question": q,
            "entity_line": extract_entity_line(user),
            "ref": extract_real_ref(user),
        })
    random.shuffle(pool)

    # A 类：真实 query + 错配参考内容（抽另一条样本的 ref）
    refs = [p["ref"] for p in pool if p["ref"]]
    for i in range(n_a):
        src = pool[i % len(pool)]
        # 错配：随机抽一条 ref，且尽量不是同一条
        wrong_ref = random.choice(refs)
        if wrong_ref == src["ref"] and len(refs) > 1:
            wrong_ref = random.choice([r for r in refs if r != src["ref"]] or refs)
        user = build_user_prompt(wrong_ref, src["entity_line"], src["question"])
        ans = build_refusal_answer(short_pointer(src["question"]))
        results.append((user, ans, src["question"]))

    # B 类：真实 query + 空参考内容
    for i in range(n_b):
        src = pool[(n_a + i) % len(pool)]
        user = build_user_prompt("", src["entity_line"], src["question"])
        ans = build_refusal_answer(short_pointer(src["question"]))
        results.append((user, ans, src["question"]))

    # C 类：边界问题（资料里没有的具体参数/细节）
    for question, entity_line, pointer in boundary_questions[:n_c]:
        user = build_user_prompt("", entity_line, question)
        ans = build_refusal_answer(pointer or question)
        results.append((user, ans, question))

    return results


# ---------------------------------------------------------------------------
# C 类边界问题（问资料里没有的具体参数/细节，覆盖三个产品）
# 每项：(question, entity_line, pointer)
# ---------------------------------------------------------------------------

BOUNDARY_QUESTIONS = [
    # 明御防火墙
    ("明御防火墙的 IPS 特征库默认签名数量是多少条？", "['安恒信息 DAS-TGFW 明御防火墙']", "IPS 特征库默认签名数量"),
    ("明御防火墙 V6R01C02SPC100 版本支持的最大并发连接数是多少？", "['安恒信息 DAS-TGFW 明御防火墙']", "最大并发连接数"),
    ("明御防火墙的 HA 双机热备切换时间阈值是多少毫秒？", "['安恒信息 DAS-TGFW 明御防火墙']", "HA 切换时间阈值"),
    ("明御防火墙默认管理端口的端口号是多少？", "['安恒信息 DAS-TGFW 明御防火墙']", "默认管理端口号"),
    ("明御防火墙日志存储默认保留多少天？", "['安恒信息 DAS-TGFW 明御防火墙']", "日志默认保留天数"),
    ("明御防火墙的 URL 过滤库默认更新频率是多久一次？", "['安恒信息 DAS-TGFW 明御防火墙']", "URL 过滤库更新频率"),
    ("明御防火墙虚拟系统支持的最大数量是多少个？", "['安恒信息 DAS-TGFW 明御防火墙']", "虚拟系统最大数量"),
    ("明御防火墙 SSL VPN 默认支持的最大并发用户数是多少？", "['安恒信息 DAS-TGFW 明御防火墙']", "SSL VPN 最大并发用户数"),
    ("明御防火墙的硬件型号都支持哪些具体的吞吐规格？", "['安恒信息 DAS-TGFW 明御防火墙']", "硬件型号吞吐规格"),
    ("明御防火墙管理员密码的最小长度要求是多少位？", "['安恒信息 DAS-TGFW 明御防火墙']", "管理员密码最小长度"),
    # 明御 APT 攻击预警平台
    ("明御APT攻击预警平台支持检测多少种 APT 攻击样本家族？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "APT 攻击样本家族数量"),
    ("明御APT攻击预警平台的沙箱环境默认支持多少种文件类型？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "沙箱支持的文件类型数量"),
    ("明御APT攻击预警平台的检测引擎默认更新周期是多少？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "检测引擎更新周期"),
    ("明御APT攻击预警平台单台设备最大可接入多少流量？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "最大接入流量"),
    ("明御APT攻击预警平台告警去重的时间窗口是多少？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "告警去重时间窗口"),
    ("明御APT攻击预警平台支持对接哪些第三方威胁情报源的 API 密钥格式？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "第三方威胁情报源 API 密钥格式"),
    # 明御运维审计与风险控制系统
    ("明御运维审计与风险控制系统的会话录像默认存储格式是什么编码？", "['安恒信息 明御运维审计与风险控制系统']", "会话录像存储编码格式"),
    ("明御运维审计与风险控制系统支持的资产默认分组上限是多少？", "['安恒信息 明御运维审计与风险控制系统']", "资产分组上限"),
    ("明御运维审计与风险控制系统的双因子认证支持哪些具体品牌令牌？", "['安恒信息 明御运维审计与风险控制系统']", "双因子认证支持的令牌品牌"),
    ("明御运维审计与风险控制系统的操作指令告警默认阈值是多少？", "['安恒信息 明御运维审计与风险控制系统']", "操作指令告警默认阈值"),
    # 补充：确保 C 类有 32 条（train 29 + val 3）
    ("明御防火墙的病毒特征库默认升级周期是多久？", "['安恒信息 DAS-TGFW 明御防火墙']", "病毒特征库升级周期"),
    ("明御防火墙支持的最大策略条目数是多少？", "['安恒信息 DAS-TGFW 明御防火墙']", "最大策略条目数"),
    ("明御防火墙的 NAT 会话表容量上限是多少？", "['安恒信息 DAS-TGFW 明御防火墙']", "NAT 会话表容量上限"),
    ("明御防火墙 IPSec VPN 支持的加密算法具体有哪些版本？", "['安恒信息 DAS-TGFW 明御防火墙']", "IPSec VPN 加密算法版本"),
    ("明御防火墙的流量日志采样比例默认值是多少？", "['安恒信息 DAS-TGFW 明御防火墙']", "流量日志采样比例默认值"),
    ("明御APT攻击预警平台支持的最大在线终端数量是多少？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "最大在线终端数量"),
    ("明御APT攻击预警平台的威胁情报库默认存储容量是多少？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "威胁情报库存储容量"),
    ("明御APT攻击预警平台告警上报的默认协议和端口是什么？", "['安恒信息 AI-APT 明御APT攻击预警平台']", "告警上报协议和端口"),
    ("明御运维审计与风险控制系统的录像回放支持的最高清晰度是多少？", "['安恒信息 明御运维审计与风险控制系统']", "录像回放最高清晰度"),
    ("明御运维审计与风险控制系统的审计日志默认保留期限是多久？", "['安恒信息 明御运维审计与风险控制系统']", "审计日志默认保留期限"),
    ("明御运维审计与风险控制系统支持的单次会话最长时长是多少？", "['安恒信息 明御运维审计与风险控制系统']", "单次会话最长时长"),
    ("明御防火墙的入侵防御默认开启哪些具体防护策略编号？", "['安恒信息 DAS-TGFW 明御防火墙']", "入侵防御默认防护策略编号"),
]


# ---------------------------------------------------------------------------
# 追加到 csv / jsonl
# ---------------------------------------------------------------------------

def append_csv(src_csv: Path, out_csv: Path, rejects: list[tuple[str, str, str]]):
    """在源 csv 基础上追加拒绝样本行。"""
    rows = load_csv_rows(src_csv)
    fieldnames = list(rows[0].keys())
    max_id = max(int(r.get("sample_id") or 0) for r in rows)
    for i, (user, ans, query) in enumerate(rejects):
        max_id += 1
        rows.append({
            "sample_id": str(max_id),
            "query": query,
            "input": user,
            "target/ground_truth": ans,
            "review_status": "reject",
            "split": rows[0].get("split") or "train",
            "note": "拒绝样本(参考内容不足)",
        })
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def append_jsonl(src_jsonl: Path, out_jsonl: Path, rejects: list[tuple[str, str, str]]):
    """在源 jsonl 基础上追加拒绝样本（messages 格式）。"""
    convs = load_jsonl_convs(src_jsonl)
    for user, ans, _query in rejects:
        convs.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": ans},
            ]
        })
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for conv in convs:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    return len(convs)


def main():
    train_convs = load_jsonl_convs(TRAIN_JSONL)
    val_convs = load_jsonl_convs(VAL_JSONL)

    # train: 100 条 = A53 + B18 + C29
    train_rejects = build_reject_dataset(
        train_convs, n_a=53, n_b=18, n_c=29, boundary_questions=BOUNDARY_QUESTIONS[:29]
    )
    # val: 12 条 = A7 + B2 + C3
    val_rejects = build_reject_dataset(
        val_convs, n_a=7, n_b=2, n_c=3,
        boundary_questions=BOUNDARY_QUESTIONS[29:32],
    )

    # 校验数量
    assert len(train_rejects) == 100, f"train rejects={len(train_rejects)}"
    assert len(val_rejects) == 12, f"val rejects={len(val_rejects)}"

    # 落盘 csv
    n_train_csv = append_csv(TRAIN_CSV, OUT_TRAIN_CSV, train_rejects)
    n_val_csv = append_csv(VAL_CSV, OUT_VAL_CSV, val_rejects)

    # 落盘 jsonl
    n_train_jsonl = append_jsonl(TRAIN_JSONL, OUT_TRAIN_JSONL, train_rejects)
    n_val_jsonl = append_jsonl(VAL_JSONL, OUT_VAL_JSONL, val_rejects)

    print("=== 生成完成 ===")
    print(f"train csv : {len(load_csv_rows(TRAIN_CSV))} -> {n_train_csv} (+{len(train_rejects)})  -> {OUT_TRAIN_CSV.name}")
    print(f"train jsonl: {len(train_convs)} -> {n_train_jsonl} (+{len(train_rejects)})  -> {OUT_TRAIN_JSONL.name}")
    print(f"val csv   : {len(load_csv_rows(VAL_CSV))} -> {n_val_csv} (+{len(val_rejects)})  -> {OUT_VAL_CSV.name}")
    print(f"val jsonl : {len(val_convs)} -> {n_val_jsonl} (+{len(val_rejects)})  -> {OUT_VAL_JSONL.name}")


if __name__ == "__main__":
    main()
