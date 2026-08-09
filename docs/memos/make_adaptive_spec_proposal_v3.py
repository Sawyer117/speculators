#!/usr/bin/env python3
"""生成 v3 立项建议书 PPT:昇腾 NPU 自适应投机解码。

v1 = 技术备忘录(调查结论);v2 = 首版立项书;v3 = 依据上游 vLLM #47808 / #48692 的
实际进展重写 —— 算法已由上游实现,项目重定义为昇腾侧适配。三版并存,不覆盖。
用法:python3 make_adaptive_spec_proposal_v3.py --out <file>.pptx
"""
from __future__ import annotations

import argparse
import copy

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x1B, 0x22, 0x2E)
MUTE = RGBColor(0x5B, 0x66, 0x77)
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
GOOD = RGBColor(0x0E, 0x7A, 0x4B)
WARN = RGBColor(0xB4, 0x53, 0x09)
RISK = RGBColor(0xA3, 0x1D, 0x2A)
BG = RGBColor(0xFF, 0xFF, 0xFF)
BAND = RGBColor(0xF2, 0xF5, 0xF9)
CJK = "微软雅黑"


def _cjk(run, name=CJK):
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    latin = rPr.find(qn("a:latin"))
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        if latin is None:
            return
        ea = copy.deepcopy(latin)
        ea.tag = qn("a:ea")
        rPr.append(ea)
    ea.set("typeface", name)


def _txbox(slide, x, y, w, h):
    tf = slide.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True
    return tf


def _p(tf, text, *, size=15, bold=False, color=INK, before=6, after=0,
       align=PP_ALIGN.LEFT, first=False, mono=False):
    para = tf.paragraphs[0] if first else tf.add_paragraph()
    para.alignment = align
    para.space_before, para.space_after = Pt(before), Pt(after)
    run = para.add_run()
    run.text = text
    run.font.size, run.font.bold, run.font.color.rgb = Pt(size), bold, color
    _cjk(run, "Consolas" if mono else CJK)
    return para


def _slide(prs, title, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    tf = _txbox(s, Inches(0.55), Inches(0.28), Inches(12.3), Inches(0.95))
    _p(tf, title, size=25, bold=True, first=True, before=0)
    if kicker:
        _p(tf, kicker, size=12, color=MUTE, before=3)
    ln = s.shapes.add_shape(1, Inches(0.55), Inches(1.20), Inches(1.3), Emu(24000))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACCENT
    ln.line.fill.background(); ln.shadow.inherit = False
    return s


def _table(slide, rows, x, y, w, col_w=None, font=11.5, row_h=0.32, colors=None, bolds=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h * len(rows))).table
    if col_w:
        tot = sum(col_w)
        for i, f in enumerate(col_w):
            t.columns[i].width = Emu(int(w * f / tot))
    t.first_row = True
    for r, row in enumerate(rows):
        t.rows[r].height = Inches(row_h)
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = BAND if r == 0 else BG
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.text_frame.word_wrap = True
            for i, line in enumerate(str(val).split("\n")):
                para = cell.text_frame.paragraphs[0] if i == 0 else cell.text_frame.add_paragraph()
                para.space_before = para.space_after = Pt(0)
                run = para.add_run()
                run.text = line
                run.font.size = Pt(font)
                run.font.bold = (r == 0) or bool(bolds and (r, c) in bolds)
                run.font.color.rgb = colors[(r, c)] if (colors and (r, c) in colors) else INK
                _cjk(run)
    return t




def build(out: str) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 封面 ──────────────────────────────────────────────────────────────────
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    tf = _txbox(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(3.2))
    _p(tf, "昇腾 NPU 自适应投机解码", size=38, bold=True, first=True, before=0)
    _p(tf, "立项建议书 v3", size=22, color=ACCENT, before=8)
    _p(tf, "按验证长度的置信度分配,替代全 batch 统一的固定 K", size=17, color=MUTE, before=12)
    _p(tf, "v3 相对 v2 的关键变化:上游 vLLM 已实现该算法(PR #47808,针对 DeepSeek-V4-Flash-DSpark),"
           "SGLang 亦已上线。本项目据此由“设计并实现”重定义为“昇腾侧适配与验证”,范围与周期同步收缩。",
       size=12, color=MUTE, before=20)

    # 2 立项理由 ──────────────────────────────────────────────────────────────
    s = _slide(prs, "一、为什么立项", "固定 K 对所有请求一视同仁,而“该验多长”的信号其实已经有了")
    _table(s, [
        ["", "现状", "问题", "本项目要解决的"],
        ["算法侧", "全 batch 统一验证长度 K", "高置信请求本可多验,低置信请求\n白白消耗 target 计算",
         "按 confidence head 的逐位接受概率\n给每请求分配 K_i"],
        ["工程侧", "固定 K 已入图,收益稳定", "K_i 每轮变化 → ragged batch,\n与设备图“形状固定”天然冲突",
         "变长与图共存,且开销可控"],
        ["生态侧", "GPU 侧上游已实现并在合入中\nNPU 侧空白", "上游方案绑定 NVIDIA 后端能力,\n昇腾无对应机制",
         "昇腾侧首个可用实现"],
    ], Inches(0.55), Inches(1.62), Inches(12.3), col_w=[0.09, 0.25, 0.33, 0.33], row_h=0.72)

    tf = _txbox(s, Inches(0.55), Inches(4.35), Inches(12.3), Inches(2.7))
    _p(tf, "核心依据:同等验证预算下,接受长度提升 7.6%(上游实测)", size=17, bold=True, color=ACCENT,
       first=True, before=0)
    _p(tf, "· vLLM PR #48692 在 Qwen3-8B + DSpark 上测得:固定 K=5 接受长度 3.445;自适应(上限 K=7、"
           "每请求有效 K=5,即同等预算)3.707 —— 同样的算力,多接受 7.6% 的 token。", size=13.5, before=8)
    _p(tf, "· 该 PR 作者同时说明:8B 模型上“测不出显著加速”,因为 verify 本来就便宜。", size=13.5, before=4)
    _p(tf, "  ⟹ 收益随 target 变贵而放大。target 越大、verify 越贵,少验一个 token 省下的越多 —— "
           "大 MoE(DSV4-Flash 284B/13B 激活)正是收益最大的场景,也正是我们已有完整链路的场景。",
       size=13.5, bold=True, before=4)
    _p(tf, "· 边界情形(非主因,但值得记录):并发继续升高时固定 K 会由正收益转为负收益。上游在 c=256 实测"
           "固定 7-token 比不开投机还慢 33%,自适应可保住收益。此时自适应从“更好”变为“必需”。",
       size=12.5, color=MUTE, before=8)

    # 3 上游现状 ──────────────────────────────────────────────────────────────
    s = _slide(prs, "二、上游已经做到哪一步 —— 我们不再需要设计算法",
               "两个 PR 是同一条线的两代;算法、调度、图支持均已成型")
    _table(s, [
        ["", "vLLM #48692", "vLLM #47808(当前主线)"],
        ["状态", "2026-08-05 关闭\n作者:改推 #47808", "OPEN,标签 ready / mrv2\nmergeable,仅待审批;8-09 仍在更新"],
        ["规模", "28 文件 +1240/-110", "40 文件 +1507/-111"],
        ["范围", "变长投机的基础设施:\n图兼容、结构化输出、logprobs、\n混合 prefill/decode 批",
         "在此之上加入按置信度的预算调度\n(本项目对标的完整形态)"],
        ["验证模型", "Qwen3-8B-FP8 + DSpark", "DeepSeek-V4-Flash-DSpark,TP=4"],
    ], Inches(0.55), Inches(1.62), Inches(12.3), col_w=[0.09, 0.42, 0.49], row_h=0.62)

    tf = _txbox(s, Inches(0.55), Inches(4.55), Inches(12.3), Inches(2.6))
    _p(tf, "#47808 的机制(昇腾侧需要对齐的就是这些)", size=16, bold=True, first=True, before=0)
    _p(tf, "· 打分:Triton kernel 按“生存概率”(逐位 confidence 的累积乘积)对所有请求的 draft slot 统一排序,"
           "跨请求竞争 —— 自信请求的第 5 位可以压过犹豫请求的第 1 位,择优录取直到全局预算用尽;", size=13, before=8)
    _p(tf, "· 成本:启动时用 dummy step profile 出成本曲线;batch 级预算在 CPU 上用双缓冲的陈旧 confidence 计算"
           "(不引入同步),per-request 分配在 GPU 上用实时值;", size=13, before=4)
    _p(tf, "· 图:decode cudagraph 改为变长 —— 按 token 网格捕获、以 max_query_len 派发、每个捕获槽非空;"
           "要求后端上报 AttentionCGSupport.ALWAYS(PR 中把两个 MLA 后端由 UNIFORM_BATCH 升级为 ALWAYS);",
       size=13, before=4)
    _p(tf, "· 限制:仅支持带 confidence head 的 DSpark;不支持 LoRA / 流水并行 / 输出 logprobs。", size=13, before=4)
    _p(tf, "另有第二份参考实现:SGLang 的 DSpark 集成(2026-07-06)采用 packed ragged varlen、"
           "graph key 仅按总 token 数,padding 浪费更小 —— 可作为图侧的备选设计。", size=12.5, color=MUTE, before=8)

    # 4 我们的差异化 ──────────────────────────────────────────────────────────
    s = _slide(prs, "三、昇腾侧的三个真实缺口 —— 这才是本项目的工作量",
               "均已在源码层面核实,不是推测")
    _table(s, [
        ["#", "缺口", "证据", "工作量判断"],
        ["1", "confidence head 在昇腾侧\n根本没有被加载",
         "vllm-ascend 的 deepseek_v4_draft.py 中\n_remap_dspark_name 对 confidence_head.*\n直接 return None;上游 main 原本同样如此,\n#47808 正是解除它的那个 PR",
         "小 —— 移植上游改法即可。\n草稿侧我们已就位:训练已产出该头,\n且已按上游区分 bias(DSV4 无 / Qwen3 有)"],
        ["2", "没有支持变长 decode 图的\n昇腾注意力后端",
         "上游依赖 AttentionCGSupport.ALWAYS;\n昇腾图模式当前按固定形状捕获",
         "★ 大 —— 本项目的主要技术风险\n与主要投入所在"],
        ["3", "预算调度 kernel 无昇腾实现",
         "上游为 Triton kernel;\n成本曲线依赖设备侧 profile",
         "中 —— 算法确定,需重写并\n在昇腾上重新标定成本曲线"],
    ], Inches(0.55), Inches(1.62), Inches(12.3), col_w=[0.04, 0.20, 0.42, 0.34], row_h=1.05)

    tf = _txbox(s, Inches(0.55), Inches(5.55), Inches(12.3), Inches(1.6))
    _p(tf, "我们的既有条件:DSV4-Flash DSpark 草稿已训练完成(5 epoch,接受长度达已发布草稿的 99.5%),"
           "训练→转换→部署→评测全链路可复现,并已具备无投机基线用于加速比测量。"
           "confidence head 训练时输入是 detach 的,可在冻结骨干上分钟级重新拟合。",
       size=13, color=MUTE, first=True, before=0)

    # 5 实施方案 ──────────────────────────────────────────────────────────────
    s = _slide(prs, "四、实施方案与里程碑", "阶段 0 由“两周量化”缩为“一天判定”—— 上游数据已给出大部分答案")
    _table(s, [
        ["阶段", "周期", "主要工作", "交付物 / 决策门"],
        ["0\n判定", "1 天",
         "在昇腾现有固定 K 链路上,把加速比从 c=48 补测到\nc=128 / c=256,看是否出现上游那样的收益衰减",
         "一张并发-加速比曲线\nG1:是否值得继续"],
        ["1\n打通", "2–3 周",
         "vllm-ascend 加载 confidence head(移植上游改法);\n草稿侧供给并校验 confidence 数值合理性",
         "端到端可读出逐位置信度\nG2:置信度与实际接受率相关"],
        ["2\n图支持", "6–8 周",
         "★ 变长 decode 图:对齐 ALWAYS 语义或采用 SGLang\n的 packed varlen;预算 kernel 昇腾实现 + 成本曲线标定",
         "动态 K 在图模式下可运行\nG3:相对固定 K 不劣化"],
        ["3\n扩展", "3–4 周",
         "推广到通用模型路径;补齐限制项;文档与复现脚本",
         "可合入的实现 + 对比数据"],
    ], Inches(0.55), Inches(1.62), Inches(12.3), col_w=[0.07, 0.08, 0.47, 0.38], row_h=0.86)

    tf = _txbox(s, Inches(0.55), Inches(5.35), Inches(12.3), Inches(1.9))
    _p(tf, "总周期约 2.5–3.5 个月(v2 为 3–4 个月;算法设计部分由上游承担后收缩)",
       size=16, bold=True, color=ACCENT, first=True, before=0)
    _p(tf, "阶段 0 只要一天:我们已有无投机基线和固定 K 的完整评测链路,只需把并发点补齐。"
           "主要不确定性已从“收益幅度”转移到“昇腾图模式能否支持变长”—— 前者上游已用数据回答,后者是阶段 2 的核心。",
       size=13, before=8)

    # 6 风险 ──────────────────────────────────────────────────────────────────
    s = _slide(prs, "五、风险与依赖", "按“是否阻断”排序")
    _table(s, [
        ["风险", "影响", "应对"],
        ["★ 昇腾无法支持变长 decode 图\n(唯一的阻断性风险)",
         "阻断阶段 2,项目退化为\n“仅 eager 模式可用”",
         "阶段 2 先做可行性验证再投入;\n备选:SGLang 式 packed varlen,\ngraph key 只按总 token 数"],
        ["昇腾侧收益幅度不及 GPU",
         "项目价值下降,但不归零\n(同预算接受长度提升仍在)",
         "阶段 0 一天内给出判定;\nG1 不达标即停"],
        ["上游 #47808 设计仍在演进",
         "移植目标变动,返工",
         "跟随主线而非分叉;\n关闭的 #48692 已说明会向 #47808 收敛"],
        ["confidence head 质量不足",
         "调度信号噪声大,收益打折",
         "输入为 detach,可在冻结骨干上\n分钟级重训,代价极低"],
    ], Inches(0.55), Inches(1.62), Inches(12.3), col_w=[0.28, 0.30, 0.42], row_h=0.95)

    # 7 决策门 ────────────────────────────────────────────────────────────────
    s = _slide(prs, "六、决策门与验收标准", "每个门都是可量化的,不靠主观判断")
    _table(s, [
        ["门", "时点", "通过标准", "不通过怎么办"],
        ["G1", "第 1 天末",
         "在某个并发点上,固定 K 的加速比相对 c=48\n出现明显衰减(或已低于无投机)",
         "停止立项。结论本身有价值:\n说明昇腾当前运行点无需自适应"],
        ["G2", "第 3 周末",
         "端到端读出逐位置信度,且与实测逐位\n接受率单调相关",
         "回到草稿侧重新拟合 confidence head\n(代价分钟级),不影响主线"],
        ["G3", "第 11 周末",
         "动态 K 在图模式下运行,端到端不劣于\n同预算的固定 K",
         "保留 eager 路径成果并如实上报;\n将图支持作为独立议题提交上游"],
    ], Inches(0.55), Inches(1.62), Inches(12.3), col_w=[0.05, 0.11, 0.46, 0.38], row_h=0.98)

    tf = _txbox(s, Inches(0.55), Inches(5.20), Inches(12.3), Inches(2.0))
    _p(tf, "最终验收", size=16, bold=True, first=True, before=0)
    _p(tf, "· 昇腾侧可用实现,与固定 K、无投机三者的对比数据齐备,复现文档可被他人独立执行;", size=13.5, before=6)
    _p(tf, "· DSV4-Flash 在动态 K 下取得不劣于固定 K 的端到端表现,并在高并发点显示优势;", size=13.5, before=4)
    _p(tf, "· 通用模型路线给出明确结论:可行方案,或阻断原因与所需外部支持。", size=13.5, before=4)
    _p(tf, "建议:先做阶段 0。一天即可拿到 G1 数据,再决定是否投入后续 2.5–3.5 个月。",
       size=16, bold=True, color=ACCENT, before=12)

    prs.save(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="adaptive-spec-project-proposal-v3.pptx")
    build(ap.parse_args().out)
