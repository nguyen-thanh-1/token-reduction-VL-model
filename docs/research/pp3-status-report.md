# Báo cáo tình trạng PP3 — 2026-11-04

Tóm tắt toàn bộ những gì đã làm, kết quả hiện có, việc đang chạy và việc tiếp
theo cho PP3 (chọn token thị giác cho Qwen3-VL-2B-Instruct). Mọi số liệu dưới
đây là **tập phát triển tune2k** (2000 câu GQA val balanced, accuracy full-token
68,90%) trừ khi ghi khác; chưa phải kết quả test.

---

## 1. Mục tiêu

1. **Ít token hơn mà accuracy ≥ baseline full-token** (theo nghĩa thống kê:
   không kém quá 1 điểm, ADR 0002).
2. **Chọn đúng vùng**: token giữ lại nằm ở vùng câu hỏi nhắc tới (hoặc xung
   quanh), gọn, không lộn xộn, và phụ thuộc câu hỏi.
3. **Bài báo**: bản 1 (Q3–Q4 hoặc hội nghị Scopus) viết trước 2026-11-05;
   bản 2 mở rộng lên Q2 (thêm Qwen3-VL-4B, nâng cấp phương pháp).

---

## 2. Phương pháp PP3 (bản hiện tại)

```
ảnh + câu hỏi
  -> ViT + PatchMerger: N token v_i và 3 mức DeepStack f_i
  -> score head 3d: điểm s_i cho từng token (nhìn token + câu hỏi)
  -> q_i = softplus(s_i) + 0.001
  -> DPP tham lam: chọn K token vừa điểm cao vừa đa dạng
  -> fold: gộp đặc trưng DeepStack của token bị bỏ vào token giữ gần nhất
  -> LLM trả lời với K token (giữ nguyên vị trí M-RoPE gốc)
```

Các mức budget: L1 = 5,6%, L2 = 11%, L3 = 22%, L4 = 33%, L5 = 50% số token.
Ví dụ dùng bên dưới: một ảnh GQA thành lưới 10 × 16 = 160 token; ở L3 giữ
K = 36 token. Mô tả đầy đủ hơn: `docs/research/pp3-method-current.md`.

### 2.1 Score head 3d — bộ chấm điểm token

Là một mạng nhỏ (**khoảng 2 triệu tham số**, so với khoảng 2 tỉ của Qwen) chạy
**trước** LLM. Vào: 160 token ảnh và các từ của câu hỏi. Ra: **một điểm cho mỗi
token** — điểm cao nghĩa là "LLM có thể cần token này để trả lời".

Mỗi token được mô tả bằng 467 số: nội dung token (`v`, nén còn 256 số), 3 mức
DeepStack (3 × 64), độ giống với từ trong câu hỏi (`z_rel`), độ nổi bật (`z_ms`),
vị trí hàng/cột (16) và budget (1). Sau đó qua 5 khối:

| Khối | Làm gì | Ví dụ "What device is not on?" |
|---|---|---|
| 1. Encoder token | đưa 467 số về 256 số | — |
| 2. **Self-attention giữa token ảnh** | mỗi token nhìn các token khác để hiểu ngữ cảnh | token ở góc màn hình TV "biết" mình thuộc một vật lớn |
| 3. Mã hóa câu hỏi | các từ câu hỏi (đã bỏ "the", "is", "what"...) hiểu ngữ cảnh của nhau | "on" hiểu nó đi cùng "not" |
| 4. **Cross-attention token → câu hỏi** | mỗi token "đọc" câu hỏi | token TV, remote phản ứng mạnh với "device" |
| 5. FFN + điểm cuối | trộn thông tin, ra điểm | — |

Điểm cuối = điểm chung của token + **điểm liên quan từ–token**:
`logsumexp` trên các từ của ⟨token, từ⟩. Phần này lớn khi **ít nhất một từ** trong
câu hỏi trỏ mạnh tới token (gần như lấy max, nhưng mượt để dễ train). Điểm được
đổi thành số dương `q̂ = softplus(s) + 0,001` để DPP dùng.

### 2.2 Nhãn huấn luyện — bắt chước chỗ LLM nhìn vào (teacher)

Câu hỏi: score head học "token nào quan trọng" từ đâu? Trả lời: **từ chính LLM**.

1. Với mỗi câu hỏi train, cho Qwen xem **đủ 160 token** và trả lời (ví dụ "tv").
2. Chạy lại một lượt có ghi lại **attention** (mức độ chú ý) của LLM.
3. Ở **12 lớp giữa** của LLM (lớp 8–19 trên 28 — nơi LLM "đọc" ảnh nhiều nhất),
   xem các vị trí sau đang chú ý vào token ảnh nào:
   - **vị trí sinh câu trả lời** (đang "nghĩ ra" đáp án);
   - **vị trí các từ của câu hỏi**.
4. Chuẩn hóa và lấy trung bình → một phân phối trên 160 token (tổng = 1). Token
   được LLM nhìn nhiều có giá trị lớn. Nhãn = 50% phần trả lời + 50% phần câu hỏi.
5. Score head học sao cho `softmax(điểm)` của nó **giống phân phối này nhất**
   (cross-entropy).

Quy mô: **50.000 câu** GQA train (mỗi câu khoảng 1,1 giây để tạo nhãn), train
**20 epoch**, giữ 10% để kiểm tra. Kết quả: 83,6% token head chọn trùng với token
teacher chọn. Sau khi train xong, **không cần chạy LLM full-token nữa** — head tự
đoán được LLM sẽ nhìn đâu.

### 2.3 Ưu điểm thực tế: không cần attention bên trong LLM

- **FlashAttention** là cách tính attention nhanh và tiết kiệm bộ nhớ mà các
  model hiện nay dùng mặc định. Nó **không tạo ra ma trận attention** đầy đủ, nên
  không đọc được "token nào được chú ý bao nhiêu".
- **FastV** chọn token **dựa vào ma trận attention ở lớp 2 của LLM**. Muốn có ma
  trận đó phải tắt FlashAttention (ít nhất ở lớp đó) và chạy LLM với đủ token
  trước đã → chậm và tốn bộ nhớ, và chỉ cắt được từ lớp 3 trở đi.
- **PP3** chỉ cần ma trận attention **lúc tạo nhãn train** (một lần, offline). Khi
  chạy thật, score head tự dự đoán điểm từ token ảnh + câu hỏi, rồi LLM nhận ngay
  K token **từ lớp đầu tiên** và vẫn chạy FlashAttention bình thường.

### 2.4 Biến thể dễ diễn giải PP3-fc (ADR 0004)

**Vấn đề cần giải quyết — attention sink.** LLM hay dồn attention vào một số
token **nền** (trời, tường) **bất kể câu hỏi là gì**. Head bắt chước LLM nên cũng
chọn các token đó: hai câu hỏi khác nhau trên cùng ảnh có khoảng **65–72% token
giữ lại trùng nhau**. Accuracy vẫn tốt, nhưng nhìn vào vùng được chọn thì không
thấy nó "theo" câu hỏi.

**Bước 1 — nhãn tương phản (bỏ phần chung của mọi câu hỏi).** GQA có nhiều câu
hỏi trên cùng một ảnh. Với mỗi câu, lấy attention của nó **trừ đi trung bình
attention của các câu hỏi khác trên cùng ảnh**, giữ phần dương, chuẩn hóa lại:

| Token | Câu 1 "What device is not on?" | Câu 2 | Câu 3 | Trung bình câu 2, 3 | Nhãn tương phản câu 1 |
|---|---|---|---|---|---|
| trời (sink) | 0,30 | 0,30 | 0,28 | 0,29 | 0,01 → gần như bị xóa |
| remote | 0,20 | 0,02 | 0,03 | 0,025 | 0,175 → giữ lại |

(số minh họa). Phần còn lại là **phần riêng của câu hỏi**. Một head thứ hai
(cùng kiến trúc 3d, tên **3d_contrast**) được train trên nhãn này; 98,6% trong
50k câu có câu hỏi "anh em" trên cùng ảnh nên dùng được.

**Bước 2 — chia budget K thành hai phần.** Ví dụ K = 36:

- **Focus = 25% (9 token)**: "mọc vùng" theo điểm của head 3d_contrast. Bắt đầu từ
  token điểm cao nhất; mỗi bước chọn token có điểm cao nhất, nhưng token **nằm kề**
  (8 hướng) vùng đã chọn được **cộng thêm 1 độ lệch chuẩn**. Kết quả: các vùng
  **liền mạch** (khoảng 3 vùng), thay vì chấm rải rác.
- **Context = 75% (27 token)**: chọn bằng DPP như PP3 thường (head 3d_attn50k) trên
  các token còn lại → giữ các token tổng thể mà LLM vẫn cần (kể cả sink).

**Bước 3** — fold và đưa vào LLM như PP3.

**Kết quả**: phần focus gọn (1,8 cụm / 10 token, so với 3,8 của PP3 thường) và phụ
thuộc câu hỏi (trùng giữa các câu hỏi 0,15, gần mức random 0,20). Ở L5 (50% token)
không mất accuracy; ở L3–L4 mất thêm 0,15–0,6 điểm so với PP3 thường.

---

## 3.0 Kết quả cuối trên tập test (2026-11-04)

Bảng đầy đủ: `outputs/reports/pp3_final/tables.md`; bản tiếng Anh trong
`docs/paper/pp3_draft.md` mục 5.1. Chênh lệch so với full (điểm %), GQA testdev
(12 578 câu, full 59,18%) và MMBench val (4329 câu, full 83,90%); MME test là
điểm (full: perception 1511,6, cognition 533,6).

| Mức | GQA: PP3 / baseline tốt nhất | MMB: PP3 / baseline tốt nhất | MME perc. PP3 | MME cogn. PP3 |
|---|---|---|---|---|
| L1 (5,7%) | **−4,04** / −11,18 | **−9,84** / −11,67 | 1385 | 323 |
| L2 (11%) | **−1,88** / −5,97 | **−4,53** / −6,01 | 1470 | 359 |
| L3 (22%) | **−0,50** (LB −0,88 ✓) / −2,58 | **−1,71** / −3,03 | 1502 | 397 |
| L4 (33%) | **−0,10** (LB −0,42 ✓) / −1,39 | **−0,97** (LB −1,48 ✗) / −1,41 | **1518** | 454 |
| L5 PP3-fc | −0,15 (LB −0,41 ✓) | −0,21 (LB −0,62 ✓) | **1534** | 493 |

- PP3 tốt nhất ở mọi mức trên GQA, MMB và MME perception.
- Non-inferiority 1 điểm: GQA đạt ở L3 và L4; MMB chỉ PP3-fc ở L5 đạt.
- Điểm yếu: MME cognition (bài nhiều chữ) giảm với mọi phương pháp; selector
  chậm với ảnh lớn (303 ms ở L3 trên MME) nên chưa có lợi về tốc độ.

## 3. Kết quả chính (tune2k)

### 3.1 So với baseline đã công bố (chênh lệch so với full, điểm %)

**Cách đọc số:** mỗi ô = accuracy của phương pháp − accuracy full-token
(68,90%), tính bằng điểm phần trăm. Dấu **−** nghĩa là **kém hơn full** bấy
nhiêu điểm; dấu **+** là hơn full; càng gần 0 càng tốt. Ví dụ PP3 ở L3 là
−0,20: accuracy 68,70% so với 68,90% của full (sai thêm 4 câu trên 2000).
Random ở L3 là −6,95: accuracy 61,95%. Cột "giữ được" = accuracy / 68,90%.

| Tỉ lệ token | Random | DivPrune | Kiểu FastV | **PP3** | PP3 giữ được |
|---|---|---|---|---|---|
| 5,7% (L1) | −20,10 | −16,55 | −30,40 | **−4,35** | 93,7% |
| 11% (L2) | −13,05 | −7,90 | −17,85 | **−1,20** | 98,3% |
| 22% (L3) | −6,95 | −3,35 | −7,70 | **−0,20** | 99,7% |
| 33% (L4) | −4,00 | −1,45 | −4,35 | **−0,20** | 99,7% |
| 50% (L5) | | | | **−0,05** | 99,9% |

So trực tiếp trên cùng câu hỏi (McNemar, đều p < 0,001):

- PP3 hơn DivPrune **+6,70** điểm ở L2, **+3,15** ở L3;
- PP3 hơn kiểu FastV **+16,65** ở L2, **+7,50** ở L3.

Cách đọc: từ 22% token trở lên PP3 gần như bằng full (khoảng tin cậy 95% ở
L3 là [−1,33; +0,93], nên kết luận đúng là "không kém quá khoảng 1 điểm").
Càng ít token, PP3 càng bỏ xa baseline.

### 3.1b Xác nhận trên gqa_val_confirm (4890 câu, không trùng tune2k, full 69,55%)

Cấu hình đã chốt **trước** khi xem split này, nên đây là kết quả độc lập.

| Tỉ lệ token | Random | DivPrune | Kiểu FastV | **PP3** | LB một phía | Đạt non-inferiority (LB > −1)? |
|---|---|---|---|---|---|---|
| 5,6% (L1) | −18,69 | −14,46 | −29,08 | **−4,62** | −5,45 | không |
| 11% (L2) | | | | **−1,62** | −2,30 | không |
| 22% (L3) | −5,81 | −3,11 | −7,03 | **−0,18** | **−0,73** | **có** |
| 33% (L4) | | | | **+0,04** | **−0,43** | **có** |

So trực tiếp: PP3 hơn DivPrune **+9,84** (L1) và **+2,92** (L3), hơn kiểu FastV
**+6,85** (L3); đều p < 0,001. Câu verify ở L3/L4: 0,860 / 0,862 so với 0,865 của
full (khoảng cách nhỏ hơn trên tune2k).

**Kết luận:** với 22% và 33% token, PP3 **không kém full quá 1 điểm** một cách có
ý nghĩa thống kê (mục tiêu 1 đạt theo nghĩa non-inferiority, ADR 0002). Ở L2 trở
xuống chưa đạt.

### 3.2 Ablation — thành phần nào quan trọng

| Cấu hình | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Random (có fold) | −20,10 | −13,05 | −6,95 | −4,00 |
| DSFold-D (điểm tự thiết kế, không học) | −11,05 | −6,05 | −3,25 | −0,95 |
| 3a: MLP, nhãn gradient, 50k | −3,15 | −1,60 | −0,50 | −0,40 |
| 3c: cross-attention, nhãn gradient | | | −1,00 | −0,70 |
| 3d: teacher attention, 10k | | | −0,25 | −0,20 |
| **3d: teacher attention, 50k (chốt)** | −4,35 | −1,20 | −0,20 | −0,20 |
| 3d 50k, bỏ fold | | −1,25 | −0,25 | +0,15 |

Kết luận:

1. **Score head học được là bước lớn nhất**: so với DSFold-D không học,
   +7,9 (L1), +4,5 (L2), +2,8 (L3) điểm.
2. **DSFold-D không học đã thắng DivPrune** ở mọi budget.
3. **Nguồn nhãn**: ở 50k nhãn, teacher attention và nhãn gradient tương
   đương (chênh trong nhiễu). Teacher chỉ thắng rõ khi ít nhãn (10k: +1,70,
   p = 0,003) và không cần backward pass.
4. **Fold DeepStack không tạo khác biệt** trên GQA → chỉ là ablation, không
   phải đóng góp chính.
5. Cấu hình chốt cho bản 1: **3d_attn50k có fold**.

### 3.3 Chọn vùng (mục tiêu 2)

| Run | Số cụm / 10 token giữ | Chồng lấn giữa các câu hỏi cùng ảnh |
|---|---|---|
| Random L3 | 3,76 | 0,13 |
| 3d_attn50k L3 | khoảng 3,8 | **0,65** |
| PP3-fc, phần focus, L4 | **1,76** | **0,15** |

- Score head (và teacher) bị **attention sink** chi phối: chọn trời, nền,
  tường gần như giống nhau cho mọi câu hỏi trên cùng ảnh (chồng lấn 0,65–0,72).
- **PP3-fc** cho phần focus gọn (khoảng 3 vùng liền) và phụ thuộc câu hỏi.
- **Đánh đổi**: càng nhiều budget cho focus, accuracy càng giảm
  (L4: focus 0 / 25 / 40 / 55% → −0,20 / −0,35 / −0,60 / −1,05). Ở **L5 (50%)**
  focus không tốn gì (−0,05, hòa 3d_attn50k).
- Chưa đo được focus có nằm **đúng vật** hay không (cần scene graph GQA,
  khoảng 40 MB, chưa tải).

### 3.4 Câu yes/no — điểm yếu còn lại

- 566/2000 câu là yes/no. Ở mọi budget đến 50%, PP3 thấp hơn full
  0,7–1,8 điểm trên nhóm này.
- Gần như toàn bộ câu sai là **đáp án "no" nhưng model trả lời "yes"**
  (L4: mất 14, được 1). Random thì ngược lại: mất câu "yes" (bỏ sót vật).
- Đã thử và **bác bỏ**: tăng budget riêng cho câu yes/no (mô phỏng gate theo
  dạng câu), bỏ fold. Không cách nào sửa được.
- Giả thuyết: chọn theo câu hỏi giữ token "khớp" câu hỏi và bỏ bằng chứng phủ
  định. Để dành cho bản 2.

### 3.5 Hiệu năng (số sơ bộ, đo khi chạy song song nên còn nhiễu)

| Run | Token vào LLM | Selector | Prefill | Decode |
|---|---|---|---|---|
| Full | 227 | 0 ms | 51 ms | 55 ms |
| PP3 L1 | 37 | 18,5 ms | 51 ms | 53 ms |
| PP3 L3 | 71 | 36,5 ms | 59 ms | 68 ms |
| DivPrune L1 | 37 | 5,1 ms | 34 ms | 39 ms |

- Ảnh GQA nhỏ (khoảng 230 token) nên prefill của LLM 2B đã rất nhanh; cắt token
  **chưa giúp nhanh hơn thời gian thực**, và selector PP3 tốn thêm thời gian.
- Bản 1 báo cáo trung thực: giảm số token (tỉ lệ với FLOPs và KV cache), thời
  gian đo riêng ở bước 5 của runbook. Tối ưu selector để dành cho bản 2.

---

## 4. Điểm mạnh và điểm yếu cho bài báo

**Mạnh**

- Thắng DivPrune và FastV rất rõ, nhất là ở budget thấp.
- Score head chạy trước LLM, không cần attention bên trong LLM.
- Phân tích attention sink, chỉ số vùng, đánh đổi diễn giải–accuracy (ít paper có).
- Quy trình thống kê chặt: paired, McNemar, khoảng tin cậy, non-inferiority.

**Yếu / rủi ro**

- Kiểu FastV kém hơn cả random → reviewer có thể nói baseline cài yếu (mình áp
  dụng ở đầu vào, không cắt giữa LLM như bản gốc). Phải ghi rõ trong bài.
- Chỉ một model 2B; chưa tăng tốc thời gian thực trên ảnh nhỏ.
- Câu yes/no đáp án "no" vẫn kém full.
- Chưa có kết quả test và MMBench/MME.

---

## 5. Đang chạy và việc tiếp theo

Chi tiết lệnh: `docs/research/pp3-runbook.md`.

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1 | Baseline + ablation trên tune2k | **Xong** |
| 2 | Xác nhận trên gqa_val_confirm (4890 câu, không trùng tune2k) | **Xong**: L3 −0,18 (LB −0,73), L4 +0,04 (LB −0,43) |
| 3 | Đánh giá cuối: GQA testdev, MMBench val, MME test; full, random, DivPrune, FastV, PP3 ở L1–L4; PP3-fc ở L5 | **Xong** (3-III bỏ qua) |
| 4 | Bảng cuối (`final_report.py`), chỉ số vùng | **Xong**: xem mục 3.0 |
| 5 | Đo thời gian chạy riêng (300 câu GQA + 300 câu MME) | **Xong**: xem paper mục 5.6 |
| — | Điền số vào bản nháp `docs/paper/pp3_draft.md`, hình, viết | Claude, ngày 4–5/11 |

**Bản 2 (sau, hướng Q2)**: Qwen3-VL-4B; FastV bản gốc (cắt trong LLM); xử lý
câu yes/no; grounding bằng scene graph; 3 seed; tối ưu selector.

---

## 6. Tệp quan trọng

| Tệp | Nội dung |
|---|---|
| `docs/research/pp3-runbook.md` | Lệnh chạy từng bước, nhóm song song |
| `docs/research/pp3-roadmap-q2.md` | Lộ trình bản 1 / bản 2 |
| `docs/paper/pp3_draft.md` | Bản nháp bài báo (tiếng Anh) |
| `docs/adr/0002`–`0005` | Quyết định: giao thức, teacher, focus/context, baseline |
| `src/token_reduction_vl/pruning/pp3_scorer/` | Code PP3, baseline (`baselines.py`), focus/context |
| `scripts/pp3/run_diagnostics.py` | Chạy nhiều cấu hình trên một lần encode |
| `scripts/pp3/paired_report.py`, `final_report.py`, `region_report.py`, `simulate_gate.py` | Báo cáo |
| `outputs/pp3_scorer/runs/` | Kết quả từng run (tune2k_, confirm_, final_) |
| `outputs/pp3_scorer/checkpoints/` | Score head đã train |

Chưa commit: toàn bộ thay đổi từ ADR 0002 trở đi (code, config, test, kết quả).
