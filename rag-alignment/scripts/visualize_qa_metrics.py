"""Read-only CSV evaluation analysis; generate self-contained HTML and SVG charts."""
import csv
import html
import itertools
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "sft前优化结果"
FILES = ["qa_result.csv", "qa_result_20261111_125356.csv", "qa_result_20261111_195326.csv", "qa_result_full_20261111.csv"]
LABELS = ["A · 原始文件", "B · 11/11 12:53", "C · 11/11 19:53", "D · full 11/11"]
METRICS = ["faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Answer Correctness"]
CN = ["忠实度", "回答相关性", "上下文精确度", "上下文召回率", "回答正确性"]
COLORS = ["#8fa6c4", "#a899ff", "#42cbbb", "#f5bb66"]

def number(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def key(r):
    return r["question"].strip(), r["ground_truth"].strip()

def pct(a, b):
    return (b - a) / a * 100 if a else None

def fmt(v, digits=4):
    return "N/A" if v is None else f"{v:.{digits}f}"

def fp(v):
    return "N/A" if v is None else f"{v:+.2f}%"

rows = [list(csv.DictReader((DATA / f).open(encoding="utf-8-sig", newline=""))) for f in FILES]
assert all(len(r) == len(set(map(key, r))) for r in rows), "Duplicate question+gold keys"
indexed = [{key(r): r for r in run} for run in rows]
stats = []
missing = []
for i, run in enumerate(rows):
    item = {"file": FILES[i], "label": LABELS[i], "rows": len(run), "metrics": {}}
    for m in METRICS:
        vals = [number(r[m]) for r in run]
        valid = [v for v in vals if v is not None]
        assert all(0 <= v <= 1 for v in valid), (FILES[i], m)
        item["metrics"][m] = {"mean": mean(valid) if valid else None, "valid": len(valid), "missing": len(run) - len(valid), "min": min(valid) if valid else None, "max": max(valid) if valid else None}
        missing.extend({"file": FILES[i], "record": j + 1, "metric": m, "raw": run[j][m]} for j, v in enumerate(vals) if v is None)
    stats.append(item)

comparisons = []
for a, b in itertools.combinations(range(4), 2):
    for m in METRICS:
        ks = sorted(set(indexed[a]) & set(indexed[b]))
        pairs = [(number(indexed[a][k][m]), number(indexed[b][k][m])) for k in ks]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        x = mean(p[0] for p in pairs) if pairs else None
        y = mean(p[1] for p in pairs) if pairs else None
        u, v = stats[a]["metrics"][m]["mean"], stats[b]["metrics"][m]["mean"]
        comparisons.append({"from": LABELS[a], "to": LABELS[b], "metric": m, "all_valid_from": u, "all_valid_to": v, "all_valid_delta": v-u, "all_valid_change_pct": pct(u, v), "paired_n": len(pairs), "paired_from": x, "paired_to": y, "paired_delta": y-x if pairs else None, "paired_change_pct": pct(x, y) if pairs else None})

common = set.intersection(*(set(r) for r in indexed))
common_stats = {}
for m in METRICS:
    ks = [k for k in sorted(common) if all(number(r[k][m]) is not None for r in indexed)]
    common_stats[m] = {"n": len(ks), "means": [mean(number(r[k][m]) for k in ks) if ks else None for r in indexed]}

OUT.mkdir(exist_ok=True)
result = {"files": stats, "common_questions_and_gold": len(common), "four_way_complete_cases": common_stats, "comparisons": comparisons, "missing_cells": missing, "notes": ["Display order A B C D is not verified chronology; only B and C have time-of-day in filenames.", "Missing scores excluded, never filled with zero.", "Relative change is not year-over-year growth.", "No latency, TTFT or throughput fields exist."]}
(OUT / "metrics_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

BG, PANEL, TEXT, MUTED = "#0c1220", "#151f30", "#e5edf8", "#9fb0c8"
def start(w, h, title, sub):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}"><rect width="{w}" height="{h}" fill="{BG}"/><g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">', f'<text x="32" y="40" fill="{TEXT}" font-size="23" font-weight="700">{title}</text>', f'<text x="32" y="69" fill="{MUTED}" font-size="13">{sub}</text>']
def text(parts, x, y, t, color=TEXT, size=13, anchor="start"):
    parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}">{html.escape(str(t))}</text>')
def end(parts):
    return "".join(parts) + "</g></svg>"

# Each metric has its own panel to keep every label legible.
p = start(1320, 760, "四份评测 · 平均评分变化", "0–1 分，越高越好；点标注均值和有效 n，连线标注相邻展示项的相对变化率。A/B/C/D 为展示顺序，非确认时间线。")
for j, m in enumerate(METRICS):
    col, row = j % 3, j // 3
    x, y = 24 + col * 432, 96 + row * 320
    p.append(f'<rect x="{x}" y="{y}" width="416" height="302" rx="14" fill="{PANEL}"/>')
    text(p, x+20, y+29, CN[j], size=17)
    xs = [x+56+i*99 for i in range(4)]
    def yy(v): return y+235-v*160
    for tick in [0, .25, .5, .75, 1]:
        ty=yy(tick)
        p.append(f'<path d="M{x+45},{ty}H{x+378}" stroke="#2b3950" stroke-width="1"/>')
        text(p,x+36,ty+4,f"{tick:g}",MUTED,10,"end")
    vals=[s["metrics"][m]["mean"] for s in stats]
    p.append('<polyline points="'+ ' '.join(f'{xx},{yy(v)}' for xx,v in zip(xs,vals)) +'" stroke="#7f96c0" stroke-width="2.5" fill="none"/>')
    for i,(xx,v) in enumerate(zip(xs,vals)):
        p.append(f'<circle cx="{xx}" cy="{yy(v)}" r="5" fill="{COLORS[i]}"/>')
        text(p,xx,yy(v)-13,fmt(v),COLORS[i],13,"middle")
        text(p,xx,y+258,"ABCD"[i],TEXT,12,"middle")
        text(p,xx,y+278,f'n={stats[i]["metrics"][m]["valid"]}',MUTED,11,"middle")
        if i:
            text(p,(xs[i-1]+xx)/2,y+55,fp(pct(vals[i-1],v)),TEXT,11,"middle")
# Sixth panel: methodology.
x,y=888,416
p.append(f'<rect x="{x}" y="{y}" width="416" height="302" rx="14" fill="{PANEL}"/>')
text(p,x+20,y+29,"文件与口径",size=17)
for i,label in enumerate(LABELS): text(p,x+20,y+65+i*29,label,COLORS[i],14)
for z,t in enumerate(["每份 10 题，四份问题与参考答案完全匹配。", "缺失评分剔除；有效 n 不同会影响均值可比性。", "C 忠实度/回答相关性仅 4/10 有效。", "A、D 无完整时间：相邻变化不等于优化收益。"]): text(p,x+20,y+194+z*23,t,MUTED,12)
trend=end(p)
(OUT/"01_mean_trends.svg").write_text(trend,encoding="utf-8")

p=start(1320,850,"四份评测 · 指标均值与相对基线差异","A = qa_result.csv 作为展示基线（不代表最早版本）；每个条形标注评分、有效样本量、相对 A 的变化百分比。")
for i,label in enumerate(LABELS): text(p,32+i*300,102,label,COLORS[i],14)
for j,m in enumerate(METRICS):
    y=135+j*135
    text(p,30,y+19,CN[j],size=16)
    for i in range(4):
        yy=y+i*26
        v=stats[i]["metrics"][m]["mean"]
        base=stats[0]["metrics"][m]["mean"]
        p.append(f'<rect x="205" y="{yy}" width="680" height="17" rx="3" fill="{PANEL}"/><rect x="205" y="{yy}" width="{v*680}" height="17" rx="3" fill="{COLORS[i]}"/>')
        lab=f'{"ABCD"[i]}  {v:.4f}  ·  n={stats[i]["metrics"][m]["valid"]}  ·  '+("基线" if i==0 else f'{fp(pct(base,v))} / Δ {v-base:+.4f}')
        text(p,910,yy+14,lab,COLORS[i],13)
text(p,32,835,"变化率 = (目标均值 − 基线均值) / 基线均值 × 100%；Δ 为 0–1 评分绝对差。缺失不计为 0。",MUTED,13)
compare=end(p)
(OUT/"02_mean_comparison.svg").write_text(compare,encoding="utf-8")

p=start(1320,650,"固定题目配对 · 相对变化率","每项仅使用两份文件中该指标均有效的相同问题。展示 D 对 A、C 对 B；C 对 B 是唯一有明确文件名时间顺序的比较。")
for a,b,c in [(0,3,COLORS[3]),(1,2,COLORS[2])]:
    xx=32 if a==0 else 680
    text(p,xx,111,f'{LABELS[b]}  对比  {LABELS[a]}',c,17)
    for j,m in enumerate(METRICS):
        rr=next(r for r in comparisons if r["from"]==LABELS[a] and r["to"]==LABELS[b] and r["metric"]==m)
        y=150+j*90
        text(p,xx,y,CN[j],TEXT,15)
        v=rr["paired_change_pct"]
        origin=xx+285
        p.append(f'<path d="M{origin},{y+8}V{y+33}" stroke="{MUTED}"/>')
        if v is not None:
            width=abs(v)*3.8
            p.append(f'<rect x="{origin if v>=0 else origin-width}" y="{y+14}" width="{width}" height="13" rx="3" fill="{c}"/>')
        text(p,xx+590,y+26,fp(v),c,16,"end")
        text(p,xx,y+54,f'{fmt(rr["paired_from"])} → {fmt(rr["paired_to"])}  ·  Δ {rr["paired_delta"]:+.4f}  ·  配对 n={rr["paired_n"]}',MUTED,13)
text(p,32,635,"配对均值与全量有效均值可能不同；配对可减少缺失样本构成差异，仍不能证明改动与评分变化之间的因果关系。",MUTED,13)
paired=end(p)
(OUT/"03_paired_changes.svg").write_text(paired,encoding="utf-8")

# Full audit tables include every score and every pairwise comparison.
def table(headers, records):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in records)+'</tbody></table></div>'
summary_table=table(["指标"]+LABELS,[[CN[j]]+[f'{fmt(s["metrics"][m]["mean"])} (n={s["metrics"][m]["valid"]}/10)' for s in stats] for j,m in enumerate(METRICS)])
complete_table=table(["指标","四份共同有效 n"]+LABELS,[[CN[j],common_stats[m]["n"]]+[fmt(v) for v in common_stats[m]["means"]] for j,m in enumerate(METRICS)])
pair_table=table(["比较","指标","全有效均值变化","全有效相对变化","配对 n","配对基线","配对目标","配对绝对差","配对相对变化"],[[r["to"].split(" · ")[0]+" 对 "+r["from"].split(" · ")[0],CN[METRICS.index(r["metric"])],f'{r["all_valid_delta"]:+.4f}',fp(r["all_valid_change_pct"]),r["paired_n"],fmt(r["paired_from"]),fmt(r["paired_to"]),f'{r["paired_delta"]:+.4f}',fp(r["paired_change_pct"])] for r in comparisons])
raw_table=table(["文件","题号","问题"]+CN,[["ABCD"[i],j+1,r["question"]]+[fmt(number(r[m]),6) for m in METRICS] for i,run in enumerate(rows) for j,r in enumerate(run)])
missing_table=table(["文件","数据记录序号","缺失指标"],[[r["file"],r["record"],CN[METRICS.index(r["metric"])]] for r in missing])
file_table=table(["标识","实际源文件","记录数","有效评分 / 应有评分"],[["ABCD"[i],f,len(rows[i]),f'{sum(s["metrics"][m]["valid"] for m in METRICS)}/50'] for i,(f,s) in enumerate(zip(FILES,stats))])
page='''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>小安智库 · 评测指标对比</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0c1220;color:#e5edf8;font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1440px;margin:auto;padding:32px}h1{font-size:32px;margin:5px 0 12px}h2{font-size:21px;margin:28px 0 12px}p{color:#aebed4}a{color:#79dcd1}.eyebrow{color:#42cbbb;letter-spacing:2px;font-size:12px}.note{border-left:3px solid #f5bb66;padding:12px 18px;background:#1c2637;border-radius:6px}.chart{overflow:auto;margin:20px 0;background:#0c1220;border:1px solid #29374c;border-radius:16px}.chart svg{width:100%;min-width:1050px;height:auto;display:block}.table{overflow:auto;border:1px solid #29374c;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:12px 15px;border-bottom:1px solid #29374c;white-space:nowrap}th{background:#192538;color:#b9cae1}td:nth-child(3){max-width:500px;white-space:normal;min-width:100px}tr:hover{background:#182438}details{margin:20px 0}summary{cursor:pointer;color:#79dcd1}::-webkit-scrollbar{width:7px;height:7px}::-webkit-scrollbar-thumb{background:#42536c;border-radius:8px}button{background:#22364b;border:1px solid #44627c;color:#dceaf8;border-radius:7px;padding:8px 14px;cursor:pointer;margin:5px 8px 5px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{padding:18px;background:#151f30;border:1px solid #29374c;border-radius:12px}.card strong{display:block;font-size:27px}.small{font-size:13px;color:#9fb0c8}@media(max-width:700px){main{padding:16px}.cards{grid-template-columns:repeat(2,1fr)}}</style></head><body><main>
<div class="eyebrow">XIAOAN / EVALUATION ANALYTICS</div><h1>四份评测，五项质量指标</h1><p>生成于 2026-11-12 · 全量有效均值 + 固定题目配对差异 + 完整原始评分</p>
<div class="cards"><div class="card">评测文件<strong>4</strong></div><div class="card">每份问题数<strong>10</strong></div><div class="card">共同问题 / 参考答案<strong>10 / 10</strong></div><div class="card">有效评分 / 全部评分<strong>181 / 200</strong></div></div>
<p class="note">源文件实际位于 <code>eval/data/</code>，而非 <code>eval/data/raw/</code>。本次分析的是回答与检索质量评分，文件未提供响应时延、TTFT、吞吐量。不存在去年同期数据，以下百分比均为相对变化率，不称为同比。A、D 缺少完整评测时刻，折线只表达展示顺序。</p>
<h2>均值总表</h2>'''+summary_table+'''
<p>均值按每文件、每指标的有效评分计算；缺失剔除，不按 0 计。所有有效评分均处于 [0, 1]。C 的忠实度和回答相关性仅 4/10 有效，均值比较需结合有效 n。</p>
<h2>01 / 平均指标折线图</h2><button onclick="downloadSvg(0)">下载折线图 SVG</button><div class="chart">'''+trend+'''</div>
<h2>02 / 均值对比与相对基线变化</h2><button onclick="downloadSvg(1)">下载均值对比图 SVG</button><div class="chart">'''+compare+'''</div>
<h2>03 / 同题有效配对对比</h2><button onclick="downloadSvg(2)">下载配对变化图 SVG</button><div class="chart">'''+paired+'''</div>
<p>相对变化率 =（目标均值 − 基线均值）÷ 基线均值 × 100%。绝对差 Δ 保留评分单位，例如 0.50 → 0.60 是 Δ +0.10，相对 +20%，两者不可混用。配对对比逐指标取双方都有效的题目，避免直接混比不同缺失构成。</p>
<h2>四份均有评分的固定题集</h2>'''+complete_table+'''
<p class="small">每个指标取四份均有效的问题交集。同一行各轮 n 一致；不同行可能对应不同题集。缺失原因未记录，不能把空评分直接判为问答失败。</p>
<details open><summary>全部六组文件间对比 · 30 项指标差异</summary>'''+pair_table+'''</details>
<details><summary>数据完整性 · 文件与缺失明细</summary>'''+file_table+missing_table+'''</details>
<details><summary>40 条原始评分 · 展示 6 位小数（N/A 表示缺失）</summary>'''+raw_table+'''</details>
<p class="small">A: qa_result.csv；B: qa_result_20261111_125356.csv；C: qa_result_20261111_195326.csv；D: qa_result_full_20261111.csv。B → C 的先后仅依据文件名，实际执行时间尚无日志佐证。样本量仅 10，未做统计显著性推断。评分模型、提示词与运行配置未由 CSV 提供，无法归因具体优化效果。</p>
<button onclick="downloadJson()">下载完整统计 JSON</button></main><script>
const audit='''+json.dumps(result,ensure_ascii=False).replace('<','\\u003c')+''';
function save(data,type,name){const a=document.createElement('a');const u=URL.createObjectURL(new Blob([data],{type}));a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
function downloadSvg(i){save(document.querySelectorAll('.chart svg')[i].outerHTML,'image/svg+xml','qa_metrics_'+(i+1)+'.svg')}
function downloadJson(){save(JSON.stringify(audit,null,2),'application/json','metrics_summary.json')}
</script></body></html>'''
(OUT/"index.html").write_text(page,encoding="utf-8")
assert sum(s["metrics"][m]["valid"] for s in stats for m in METRICS)==181
assert len(comparisons)==30
print(json.dumps({"output":str(OUT),"common":len(common),"missing":len(missing),"paired_D_vs_A":[r for r in comparisons if r["from"]==LABELS[0] and r["to"]==LABELS[3]],"four_way_complete_cases":common_stats},ensure_ascii=False,indent=2))
