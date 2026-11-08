"""Generate the focused PP3 (3d attention-distilled scorer) draw.io diagram.

Writes architecture/pp3-dsfold-scorer/PP3-3d-attention-scorer.drawio. Edit this
script (or the .drawio file in draw.io) when the method changes; PNG/SVG exports
live in architecture/pp3-dsfold-scorer/exports/.

Example:
    uv run python scripts/pp3/make_architecture_drawio.py
"""

from pathlib import Path
from xml.sax.saxutils import quoteattr

OUT = Path(__file__).resolve().parents[2] / "architecture" / "pp3-dsfold-scorer" / "PP3-3d-attention-scorer.drawio"

FROZEN = ("#DAE8FC", "#6C8EBF")
LEARN = ("#E1D5E7", "#9673A6")
ALGO = ("#FFE6CC", "#D79B00")
TRAIN = ("#F5F5F5", "#666666")
DATA = ("#FFFFFF", "#999999")
NOTE = ("#FFF9DB", "#C9A400")

cells: list[str] = []
counter = [0]


def nid(prefix: str) -> str:
    counter[0] += 1
    return f"{prefix}{counter[0]}"


def box(x, y, w, h, html, colors=DATA, size=13, rounded=True, align="center", valign="middle", dashed=False, bold_title=None):
    i = nid("b")
    fill, stroke = colors
    style = (f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={size};align={align};verticalAlign={valign};spacing=8;arcSize=6;"
             + ("dashed=1;" if dashed else ""))
    value = (f"<b>{bold_title}</b><br>" if bold_title else "") + html
    cells.append(f'<mxCell id="{i}" value={quoteattr(value)} style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def panel(x, y, w, h, title, colors):
    fill, stroke = colors
    i = nid("p")
    style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=18;"
             "verticalAlign=top;align=left;spacingLeft=14;spacingTop=8;arcSize=2;opacity=60;")
    cells.append(f'<mxCell id="{i}" value={quoteattr("<b>" + title + "</b>")} style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def text(x, y, w, h, html, size=13, align="left"):
    i = nid("t")
    style = f"text;html=1;whiteSpace=wrap;fontSize={size};align={align};verticalAlign=top;"
    cells.append(f'<mxCell id="{i}" value={quoteattr(html)} style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(src, tgt, label="", dashed=False, color="#333333", exit=None, entry=None):
    i = nid("e")
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeWidth=2;"
             f"strokeColor={color};fontSize=12;labelBackgroundColor=#FFFFFF;" + ("dashed=1;" if dashed else ""))
    if exit:
        style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
    if entry:
        style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    cells.append(f'<mxCell id="{i}" value={quoteattr(label)} style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')


# ---------------------------------------------------------------- title + legend
text(40, 10, 2600, 50, "<b>DeSTiL — chọn token ảnh theo câu hỏi bằng score head học từ attention (head 3d_attn50k) trên Qwen3-VL-2B-Instruct</b>", 26)
text(40, 58, 2600, 30, "Nguyễn Thiện Thanh, Nguyễn Trung Hiếu, Gia Bảo — GVHD: Lê Phú Nguyên. Qwen3-VL đóng băng hoàn toàn; chỉ score head (1.993.153 tham số) được huấn luyện.", 15)
lx = 40
for label, colors in (("Qwen3-VL đóng băng", FROZEN), ("Học (score head)", LEARN), ("Thuật toán cố định, không học", ALGO),
                      ("Huấn luyện offline (một lần)", TRAIN), ("Tensor / dữ liệu", DATA)):
    box(lx, 96, 230, 34, label, colors, 13)
    lx += 250

# ---------------------------------------------------------------- panel A: inference
panel(20, 150, 2780, 720, "A. LUỒNG SUY LUẬN (mỗi câu hỏi) — ví dụ ảnh 160 token, mức L3 (r = 2/9 → K = 36)", ("#FAFAFA", "#BBBBBB"))

inp = box(50, 220, 230, 150, "Ảnh RGB (bất kỳ kích thước)<br>+ câu hỏi văn bản<br>+ mức budget r ∈ {1/18, 1/9, 2/9, 1/3, 1/2}", DATA, 13, bold_title="Đầu vào")
proc = box(330, 220, 250, 150, "resize bội số 32 px<br>patch 16 px → lưới H×W patch<br>prompt chat có<br>&lt;|image_pad|&gt; × N", FROZEN, 13, bold_title="Processor Qwen3-VL")
vit = box(630, 200, 290, 190, "24 lớp, hidden 1024, 407M tham số<br>PatchMerger 2×2 → 1 token ≈ 32×32 px<br><br>nhánh chính → <b>v</b> [N × 2048]<br>DeepStack từ lớp 5, 11, 17 → <b>f⁽¹⁾,f⁽²⁾,f⁽³⁾</b> [N × 2048]", FROZEN, 13, bold_title="Vision Encoder (ViT)")
emb = box(630, 430, 290, 110, "embed_tokens (bảng 151.936 × 2048)<br>→ embedding câu hỏi <b>e</b> [N_q × 2048]", FROZEN, 13, bold_title="Embedding từ của LLM")

feat = box(980, 200, 330, 340,
           "<div style='text-align:left'>• PCA cố định (fit 1 lần trên 1000 ảnh train):<br>&nbsp;&nbsp;v 2048→256; mỗi mức DeepStack 2048→64;<br>&nbsp;&nbsp;câu hỏi 2048→256<br>"
           "• 2 điểm DSFold: z_rel (liên quan câu hỏi),<br>&nbsp;&nbsp;z_ms (khác biệt giữa các mức)<br>"
           "• vị trí Fourier của ô lưới (16 số)<br>• log r (budget)<br><br>"
           "ghép → <b>u_i</b> = [256 ; 3×64 ; 2 ; 16 ; 1] = <b>467 số / token</b><br>câu hỏi → <b>w</b> [N_q × 256]</div>",
           ALGO, 13, bold_title="S1 — Đặc trưng token (không học)")

head = box(1360, 200, 360, 340,
           "<div style='text-align:left'>① Linear 467→256 → h [N×256]<br>"
           "② <b>Self-attention</b> giữa các token ảnh<br>&nbsp;&nbsp;(token biết ngữ cảnh toàn ảnh)<br>"
           "③ Câu hỏi: + vị trí từ, 1 lớp Transformer<br>"
           "④ <b>Cross-attention</b> token → từ câu hỏi<br>"
           "⑤ FFN 256→1024→256, LayerNorm<br>"
           "⑥ a_i = Linear(h_i) (điểm chung)<br>"
           "⑦ b_i = logsumexp_w ⟨W_k h_i, W_q w_w⟩/√256<br>&nbsp;&nbsp;(token khớp từ nào nhất trong câu hỏi)<br>"
           "⑧ <b>s_i = a_i + b_i</b> → s [N]</div>",
           LEARN, 13, bold_title="S2 — Score head 3d (1,99M tham số, HỌC)")

qual = box(1770, 290, 200, 160, "q̂_i = softplus(s_i) + 0,001<br><br>chất lượng dương<br>cho từng token<br>[N]", ALGO, 13, bold_title="Chất lượng")

dpp = box(2020, 200, 330, 340,
          "<div style='text-align:left'>K = ⌈r·N⌉ (ví dụ 36 / 160)<br><br>"
          "Kernel <b>L = diag(q̂)·S·diag(q̂)</b><br>S_ij = cos(v_i, v_j)<br><br>"
          "MAP tham lam: mỗi bước chọn token làm<br>log det(L_chọn) tăng nhiều nhất<br>"
          "→ ưu tiên token <b>điểm cao</b> nhưng<br><b>khác nhau</b> (không chọn 36 token<br>giống hệt nhau cùng một vùng)<br><br>"
          "ra: tập giữ K chỉ số, tập bỏ N−K</div>",
          ALGO, 13, bold_title="S3 — DPP chọn K token")

fold = box(2400, 200, 360, 340,
           "<div style='text-align:left'>Mỗi token bị bỏ d gán cho token giữ k:<br>"
           "a(d) = argmax_k cos(v_d,v_k) − λ·dist(d,k)/diag<br><br>"
           "Mỗi mức DeepStack l = 1..3:<br>"
           "f'_k = norm-match( f_k + γ Σ_{d∈A(k)} w_dk f_d )<br>"
           "w_dk = softmax_d( cos(v_d,v_k) / τ )<br><br>"
           "λ = 0,5; γ = 0,5; τ = 0,1<br>nhánh chính v <b>không đổi</b><br>(ablation: fold trung tính trên GQA)</div>",
           ALGO, 13, bold_title="S4 — DeepStack fold")

llm = box(1760, 620, 740, 200,
          "Chuỗi vào = text + <b>K token ảnh giữ lại</b> (thay vì N) tại <b>vị trí M-RoPE gốc</b> (t, h, w) của chúng<br>"
          "DeepStack: f'⁽¹⁾,f'⁽²⁾,f'⁽³⁾ của K token cộng vào hidden state ở lớp 0, 1, 2<br>"
          "28 lớp, hidden 2048, 16 Q / 8 KV head, FFN 6144, 1,72B tham số<br>"
          "Attention nhanh (SDPA / FlashAttention) — DeSTiL không cần trọng số attention lúc suy luận",
          FROZEN, 13, bold_title="LLM Qwen3 (đóng băng)")
ans = box(2560, 700, 200, 120, "greedy decode<br>tối đa 32 token<br>→ câu trả lời", DATA, 13, bold_title="Đầu ra")
saving = box(980, 620, 470, 200,
             "<div style='text-align:left'>• LLM đọc ít token hơn: GQA 294 → 86 (−71%),<br>&nbsp;&nbsp;MME 1790 → 425 (−76%) ở L3<br>"
             "• KV cache ngắn hơn tương ứng<br>• Chi phí thêm: S1–S4 ≈ 10–90 ms / câu<br>"
             "• Chất lượng L3: GQA −0,50, MMB −1,71 điểm;<br>&nbsp;&nbsp;MME perception 1502 / 1512</div>",
             NOTE, 13, bold_title="Hiệu quả (kết quả test)")

edge(inp, proc)
edge(proc, vit, "pixel")
edge(proc, emb, "id câu hỏi", exit=(0.5, 1), entry=(0, 0.5))
edge(vit, feat, "v, f⁽¹⁻³⁾")
edge(emb, feat, "e", exit=(1, 0.5), entry=(0, 0.85))
edge(feat, head, "u [N×467], w")
edge(head, qual, "s")
edge(qual, dpp, "q̂")
edge(dpp, fold, "giữ / bỏ")
edge(fold, llm, "v_kept [K×2048]<br>f'_kept [3×K×2048]", exit=(0.5, 1), entry=(1, 0.3))
edge(emb, llm, "token văn bản", exit=(0.5, 1), entry=(0.1, 1))
edge(llm, ans, exit=(1, 0.75), entry=(0, 0.5))

# ---------------------------------------------------------------- panel B: training (teacher)
panel(20, 890, 1700, 560, "B. HUẤN LUYỆN SCORE HEAD — chưng cất attention của chính mô hình full (offline, một lần)", ("#F7F7F7", "#999999"))
data = box(50, 960, 230, 150, "GQA train balanced<br>50.000 câu hỏi<br>(ảnh khác tập đánh giá)", TRAIN, 13, bold_title="Dữ liệu")
full = box(330, 940, 300, 190, "Qwen3-VL full token, trả lời greedy<br>→ đáp án của chính mô hình<br><br>chạy lại <b>teacher-forced</b> trên đáp án đó,<br>attention <b>eager</b> (chỉ lượt này)", TRAIN, 13, bold_title="Lượt teacher")
hooks = box(680, 940, 330, 190,
            "<div style='text-align:left'>hook lớp 8–19 đọc A [16 head × L × L]<br>"
            "hàng: vị trí trả lời + vị trí câu hỏi<br>cột: N vị trí token ảnh → [16 × R × N]<br>"
            "chuẩn hóa lại trên N cột (bỏ phần nhìn chữ)<br>trung bình head → lớp → hàng</div>", TRAIN, 13, bold_title="Đọc attention")
label = box(1060, 940, 280, 190,
            "p_ans [N], p_q [N]<br><br><b>p = 0,5·p_ans + 0,5·p_q</b><br><br>N số ≥ 0, tổng = 1<br>(bản đồ nhiệt trên các token ảnh)", TRAIN, 13, bold_title="Nhãn teacher")
loss = box(1390, 940, 300, 190,
           "Loss = CE( p , softmax(s) ), T = 1<br>AdamW lr 3e-4 cosine, wd 0,01<br>batch 16, 20 epoch, patience 3<br><br>recall top-K so với teacher: 0,836", LEARN, 13, bold_title="Học score head")
text(50, 1160, 1640, 280,
     "<b>Vì sao làm vậy?</b> Attention cho biết LLM thực sự \"nhìn\" token ảnh nào khi trả lời câu hỏi. Lúc suy luận, đọc attention bên trong LLM "
     "(như FastV) buộc phải chạy LLM trên đủ N token và dùng attention eager (chậm, tốn bộ nhớ: ảnh MME 13.862 token cần ma trận 16 × 13.862² ≈ 6 GB). "
     "DeSTiL học trước một mạng nhỏ <b>đoán</b> bản đồ attention đó từ chính token ảnh + câu hỏi, nên chọn token <b>trước</b> LLM và LLM vẫn dùng FlashAttention.<br><br>"
     "<b>Nguồn nhãn thay thế (ablation):</b> nhãn gradient × input (head 3a). Với 50k nhãn hai nguồn ngang nhau; với 10k nhãn teacher hơn +1,70 điểm.<br>"
     "<b>Giới hạn:</b> teacher bắt chước mô hình full nên trần kỳ vọng là ngang full.", 14)
edge(data, full)
edge(full, hooks, "A")
edge(hooks, label)
edge(label, loss, "mục tiêu p")
edge(loss, head, "trọng số đã học", dashed=True, color="#9673A6", exit=(0.5, 0), entry=(0.5, 1))

# ---------------------------------------------------------------- panel C: PP3-fc
panel(1760, 890, 1040, 560, "C. BIẾN THỂ DỄ DIỄN GIẢI DeSTiL-fc (focus + context)", ("#FBF7FD", "#9673A6"))
sink = box(1790, 950, 470, 150,
           "Teacher dồn nhiều attention vào nền (trời, tường) cho <b>mọi</b> câu hỏi:<br>"
           "tập giữ của các câu hỏi khác nhau trên cùng ảnh trùng 0,65 (L3)<br>so với 0,13 nếu chọn ngẫu nhiên",
           NOTE, 13, bold_title="Attention sink")
contrast = box(2300, 950, 470, 150,
               "<b>t ∝ max(0, p − trung bình p của các câu hỏi khác cùng ảnh)</b><br>"
               "bỏ phần chung của ảnh, giữ phần riêng của câu hỏi<br>head 3d thứ hai học t",
               LEARN, 13, bold_title="Nhãn tương phản")
focus = box(1790, 1150, 470, 160,
            "25% của K: mọc vùng liên thông trên điểm tương phản<br>(chuẩn hóa z, thưởng 8 láng giềng β = 1)<br>"
            "→ vùng gọn: 1,4–1,8 vùng / 10 token<br>trùng giữa câu hỏi chỉ 0,15–0,18",
            ALGO, 13, bold_title="Focus (vùng theo câu hỏi)")
context = box(2300, 1150, 470, 160,
              "75% còn lại: DPP trên các token chưa chọn<br>với chất lượng q̂ của head chính<br><br>"
              "L5 (50%): GQA −0,15, MMB −0,21,<br>MME perception 1534 (> full 1512)",
              ALGO, 13, bold_title="Context")
edge(sink, contrast)
edge(contrast, focus, exit=(0.25, 1), entry=(0.75, 0))
edge(focus, context, "phần còn lại")

body = "".join(cells)
xml = ('<mxfile host="app.diagrams.net"><diagram id="pp3-3d" name="PP3_3d_attention_scorer">'
       '<mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="1" pageScale="1" pageWidth="2840" pageHeight="1480" math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>' + body + '</root></mxGraphModel></diagram></mxfile>')
OUT.write_text(xml, encoding="utf-8")
print(OUT, len(xml))
