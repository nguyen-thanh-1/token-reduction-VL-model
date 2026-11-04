# Phương pháp PP3 hiện tại — giải thích dễ hiểu

Ngày: 2026-11-03. Cấu hình chốt cho bản 1: `configs/methods/pp3_scorer_attn50k.yaml`
(score head **3d_attn50k**, tham số DSFold **D-1.0**).

Tài liệu đi từ trực giác đến chi tiết. Một **ví dụ chạy xuyên suốt** (một câu
thật trong tập tune2k) được dùng ở mọi bước:

> Ảnh `2399127` của GQA, câu hỏi **"What device is not on?"**, đáp án đúng
> "remote control". Ảnh thành lưới **10 hàng × 16 cột = 160 token**. Ở mức L3
> (giữ 2/9), PP3 giữ **K = 36 token**.

---

## Đọc nhanh trong 30 giây

Qwen3-VL cắt ảnh thành nhiều "mảnh" (token) và đưa **tất cả** vào mô hình ngôn
ngữ (LLM). Phần lớn mảnh không cần cho câu hỏi (trời, tường, sàn...). PP3 là
một bộ lọc nhỏ đặt **trước LLM**:

1. nhìn từng mảnh và câu hỏi, **chấm điểm** "mảnh này LLM có cần không";
2. **chọn** K mảnh điểm cao nhưng **không trùng lặp** nhau;
3. **gói** thông tin phụ của các mảnh bị bỏ vào mảnh được giữ gần nhất;
4. đưa K mảnh vào LLM để trả lời.

Bộ chấm điểm học bằng cách **bắt chước chỗ mà chính LLM nhìn vào** khi được
xem đủ ảnh. Qwen không bị sửa, không train lại.

---

## 1. Qwen3-VL xử lý một ảnh như thế nào

1. Ảnh được chia thành các ô **16×16 pixel** (patch). ViT (bộ mã hóa ảnh, 24 lớp)
   biến mỗi patch thành một vector.
2. **PatchMerger** gộp mỗi khối 2×2 patch thành **1 token** → mỗi token đại diện
   cho một ô khoảng **32×32 pixel** của ảnh. Ảnh ví dụ thành lưới 10×16 = 160 token.
   Mỗi token là một vector `v_i` dài 2048 số — đúng thứ LLM nhận.
3. **DeepStack** (đặc điểm riêng của Qwen3-VL): ngoài `v_i`, mô hình còn lấy đặc
   trưng của từng token ở **3 lớp giữa của ViT** (lớp 5, 11, 17; `f_i^(1)`, `f_i^(2)`, `f_i^(3)`)
   và **cộng thêm** chúng vào LLM ở lớp 0, 1, 2. Giống như ngoài "bức ảnh đã xử
   lý xong", LLM còn được xem thêm 3 "bản phác thảo" ở các mức chi tiết khác nhau.
4. Chuỗi đưa vào LLM: `[mở đầu] [160 token ảnh] [câu hỏi] [chỉ dẫn trả lời]`.
   Mỗi token ảnh có vị trí 3 chiều (thời gian, hàng, cột) gọi là **M-RoPE**.

Vấn đề: 160 token ảnh chiếm phần lớn chuỗi (câu hỏi chỉ vài chục token). Với ảnh
lớn hoặc video, con số lên hàng nghìn → tốn tính toán và bộ nhớ.

---

## 2. Bức tranh tổng thể của PP3

```
ảnh ─► ViT + PatchMerger ─► 160 token v_i  + DeepStack f_i (3 mức)
                                  │
câu hỏi "What device is not on?" ─┤ (embedding từ bảng từ vựng của LLM)
                                  ▼
  Bước 1  Đặc trưng: mỗi token thành một vector 467 số
                                  ▼
  Bước 2  Score head 3d: mỗi token một điểm s_i (cao = LLM có thể cần)
                                  ▼
  Bước 3  DPP: chọn 36 token điểm cao + đa dạng
                                  ▼
  Bước 4  Fold: gói DeepStack của 124 token bị bỏ vào 36 token giữ
                                  ▼
  Bước 5  LLM Qwen3-VL nhận 36 token (giữ vị trí gốc) ─► "tv" / "remote control"
```

Số token giữ: **K = ⌈r · N⌉**, với r là tỉ lệ giữ (budget). Các mức chung:

| Mức | Tỉ lệ r | Ví dụ N = 160 |
|---|---|---|
| L1 | 1/18 ≈ 5,6% | 9 token |
| L2 | 1/9 ≈ 11% | 18 token |
| L3 | 2/9 ≈ 22% | 36 token |
| L4 | 1/3 ≈ 33% | 54 token |
| L5 | 1/2 = 50% | 80 token |

---

## 3. Bước 1 — Mô tả mỗi token bằng một vector đặc trưng

Score head không nhìn ảnh gốc; nó nhìn một "bản tóm tắt" của từng token gồm:

**a) Nội dung token** — `v_i` (2048 số). Để lưu được đặc trưng của 50k ảnh khi
train (nếu không sẽ cần khoảng 220 GB), dùng **PCA cố định** nén 2048 → 256 số. PCA
fit một lần trên 1000 ảnh train và không bao giờ đổi; lúc chạy dùng đúng PCA đó.

**b) Ba mức DeepStack** — mỗi mức chuẩn hóa rồi nén 2048 → 64 số.

**c) Hai điểm "tự thiết kế" của DSFold** (có sẵn từ phương pháp gốc PP1):

- `z_rel` — **token có giống từ nào trong câu hỏi không**. Lấy embedding của các
  từ câu hỏi (bỏ từ vô nghĩa như "the", "is", "what"; ví dụ còn lại "device",
  "not", "on"), đo độ giống cosine với `v_i`, lấy giá trị lớn nhất, rồi chuẩn hóa
  (z-score: trừ trung bình, chia độ lệch chuẩn, cắt trong [−3, 3]).
- `z_ms` — **token có "nổi bật" không**: đo xem đặc trưng của token thay đổi
  nhiều đến đâu giữa 3 mức DeepStack (`‖f̂³ − f̂²‖ + ‖f̂² − f̂¹‖`). Vùng có cạnh,
  vật thể, chữ thường thay đổi nhiều; vùng trời phẳng thay đổi ít.

**d) Vị trí** — hàng và cột của token mã hóa bằng sin/cos ở 4 tần số (16 số), để
head biết token ở giữa hay ở rìa ảnh.

**e) Budget** — `log r` (1 số), để head biết đang phải chọn ít hay nhiều.

Tổng: 256 + 3×64 + 2 + 16 + 1 = **467 số cho mỗi token**. Câu hỏi: mỗi từ
(sau khi bỏ stopword) cũng nén 2048 → 256 số.

---

## 4. Bước 2 — Score head 3d chấm điểm từng token

Score head là một mạng nhỏ (**khoảng 2,0 triệu tham số**, so với 2 tỉ của Qwen).
Đầu vào: 160 vector token (467 số) + các vector từ câu hỏi. Đầu ra: **160 điểm**.

Các lớp, theo thứ tự:

| # | Lớp | Làm gì (trực giác) |
|---|---|---|
| 1 | Linear 467 → 256 | đưa mọi token về cùng một không gian |
| 2 | **Self-attention giữa token ảnh** (4 head) | mỗi token "nhìn" các token khác: biết mình là một phần của cái TV, hay là nền |
| 3 | **Mã hóa câu hỏi**: thêm vị trí từ, 1 lớp Transformer | embedding từ ban đầu là tĩnh ("on" không biết đứng cạnh "not"); lớp này cho các từ hiểu ngữ cảnh của nhau |
| 4 | **Cross-attention token → câu hỏi** | mỗi token "đọc" câu hỏi, tự hỏi: câu hỏi có nói về mình không |
| 5 | FFN 256 → 1024 → 256 + LayerNorm | trộn thông tin |
| 6 | **Điểm cuối** | `s_i = (điểm chung của token) + (điểm liên quan mạnh nhất với một từ)` |

Phần "điểm liên quan" là `logsumexp` trên các từ của `⟨k_i, q_w⟩ / 16`: nó **cao
khi có ít nhất một từ trỏ mạnh tới token i** (ví dụ từ "device" trỏ tới TV,
remote). Gần giống lấy max nhưng mượt hơn để dễ train.

Cuối cùng đổi điểm thành **chất lượng dương**: `q̂_i = softplus(s_i) + 0,001`
(DPP ở bước 3 cần số dương).

Các phiên bản head đã thử:

| Head | Kiến trúc | Tham số |
|---|---|---|
| 3a | MLP từng token, câu hỏi tóm thành 1 vector trung bình | 0,78 triệu |
| 3c | 3a + self-attention + cross-attention | 1,32 triệu |
| **3d** | 3c + mã hóa câu hỏi + điểm liên quan từ–token | 1,99 triệu |

---

## 5. Score head học như thế nào (ADR 0003)

### 5.1 Ý tưởng: bắt chước "ánh mắt" của LLM

Ta muốn giữ những token mà **LLM thực sự dùng để trả lời**. Cách đo trực tiếp:
cho LLM xem **đủ** 160 token, rồi xem khi trả lời nó **đặt attention** (sự chú ý)
vào token ảnh nào. Phân phối attention đó là **nhãn** (gọi là *teacher*). Score
head (*student*) học dự đoán phân phối này **chỉ từ ảnh và câu hỏi**, không cần
chạy LLM. Lúc dùng thật, ta không cần LLM full-token nữa.

### 5.2 Tạo nhãn cho một câu hỏi (làm 50k lần, mỗi lần khoảng 1,1 giây)

1. Chạy Qwen full-token, sinh câu trả lời (greedy), ví dụ "tv".
2. Chạy lại một lượt, đưa sẵn câu trả lời vào (teacher forcing), và **bật chế độ
   trả về attention** (eager attention) chỉ cho lượt này.
3. Ở **12 lớp giữa của LLM (lớp 8–19 trên 28)**, đọc attention từ hai nhóm vị trí
   tới 160 cột token ảnh:
   - **nhóm trả lời**: vị trí cuối của prompt và các token của câu trả lời
     (những vị trí đang "nghĩ ra" đáp án);
   - **nhóm câu hỏi**: các vị trí của từ trong câu hỏi.
4. Với mỗi head attention, chuẩn hóa sao cho tổng trên 160 token ảnh = 1; rồi lấy
   trung bình theo head, theo vị trí và theo lớp.
5. Nhãn: `p = 0,5 × p_trả_lời + 0,5 × p_câu_hỏi` — một phân phối trên 160 token
   (tổng = 1), token càng được nhìn nhiều thì `p_i` càng lớn.

Vì sao lớp 8–19: các lớp đầu chủ yếu xử lý cục bộ, các lớp cuối tập trung vào
sinh chữ; các lớp giữa là nơi LLM "đọc" ảnh để lấy thông tin.

### 5.3 Hàm mất mát và cách train

- Head cho 160 điểm `s`; `softmax(s)` biến chúng thành phân phối.
- **Distillation**: cross-entropy giữa `softmax(s)` và `p` — head bị phạt khi dồn
  điểm vào token mà LLM không nhìn, hoặc bỏ qua token LLM nhìn nhiều.
- 50k câu GQA train balanced; 10% giữ lại để kiểm tra; AdamW, lr 3e-4 giảm dần
  (cosine), weight decay 0,01, batch 16, tối đa 20 epoch, dừng sớm nếu 3 epoch
  không cải thiện; seed 0.
- Kết quả: trong K token head chọn, **83,6% trùng** với K token teacher chọn
  (trên tập giữ lại).
- File: `outputs/pp3_scorer/checkpoints/score_head_3d_attn50k_seed0.pt`.

Nguồn nhãn khác đã thử: **gradient × input** (mức ảnh hưởng của từng token tới
câu trả lời, cần một lượt lan truyền ngược). Với 50k nhãn hai nguồn cho kết quả
tương đương; với 10k nhãn teacher attention tốt hơn +1,70 điểm.

### 5.4 Một hiện tượng quan trọng: attention sink

Teacher thường dồn nhiều attention vào **các token nền** (trời, tường) dù câu hỏi
là gì — gọi là *attention sink* (LLM dùng chúng như "chỗ đổ" attention thừa). Hệ
quả: head bắt chước teacher cũng chọn những token đó cho mọi câu hỏi trên cùng
ảnh (65–72% token trùng nhau giữa các câu hỏi khác nhau). Điều này **không làm
giảm accuracy** (LLM có vẻ cần chúng), nhưng làm vùng chọn **khó diễn giải**. Biến
thể PP3-fc ở mục 9 xử lý chuyện này.

---

## 6. Bước 3 — Chọn K token bằng DPP

### 6.1 Vì sao không lấy top-K điểm cao nhất

Nếu TV chiếm 20 token liền nhau, cả 20 đều có điểm cao và **gần như giống hệt
nhau**. Lấy top-36 sẽ tốn 20 chỗ cho một vật, bỏ sót cái remote hay cái bàn. Ta
muốn: **điểm cao** nhưng **mỗi token mang thông tin mới**.

### 6.2 DPP (Determinantal Point Process) làm điều đó thế nào

Mỗi cặp token có độ giống `S_ij = cos(v_i, v_j)`. DPP cho mỗi tập token một "điểm
tập hợp" bằng định thức của ma trận `L = diag(q̂) · S · diag(q̂)`. Định thức lớn
khi các token **có chất lượng cao** và **khác nhau** (hai token giống hệt nhau làm
định thức gần bằng 0).

Thuật toán tham lam (Chen và cộng sự, 2018), lặp K lần:

1. Lần đầu: chọn token có `q̂` cao nhất.
2. Mỗi lần sau: với mỗi token chưa chọn, tính **phần "mới" còn lại** của nó sau khi
   trừ đi phần đã được các token đã chọn bao phủ (nhân với chất lượng); chọn token
   có phần mới lớn nhất.
3. Một token gần giống token đã chọn thì phần mới gần 0 → bị bỏ qua dù điểm cao.

Ví dụ: 36 token giữ lại sẽ có vài token của TV, vài token của remote, bàn, tay
người... thay vì 36 token đều nằm trên TV.

(Ảnh rất lớn, trên 4096 token, dùng phiên bản nhanh hơn "max–min có trọng số";
với GQA luôn dùng DPP chính xác.)

---

## 7. Bước 4 — Fold: không vứt hẳn token bị bỏ

124 token bị bỏ vẫn có thông tin DeepStack. Fold "gói" phần này vào token giữ:

1. **Gán chủ**: mỗi token bị bỏ `d` tìm token giữ `k` vừa **giống nó**
   (cosine cao) vừa **gần nó trên ảnh**:
   `chủ(d) = argmax_k [ cos(v_d, v_k) − 0,5 × khoảng_cách(d, k) / đường_chéo_lưới ]`.
2. **Gộp**: với mỗi mức DeepStack, token giữ `k` cộng thêm trung bình có trọng số
   của các token mà nó làm chủ:
   `f'_k = f_k + 0,5 × Σ w_dk · f_d`, trọng số `w_dk` theo độ giống (softmax,
   nhiệt độ 0,1), rồi chỉnh lại độ lớn bằng độ lớn ban đầu của `f_k`.
3. Token chính `v_k` **không đổi** — chỉ phần DeepStack được gộp.

Kết quả thí nghiệm: bỏ fold **không khác biệt** trên GQA (chênh trong khoảng nhiễu).
Giữ fold vì không hại, nhưng trong bài báo nó chỉ là một ablation.

---

## 8. Bước 5 — Đưa K token vào LLM

1. Tính vị trí M-RoPE cho chuỗi **đầy đủ** 160 token như bình thường.
2. Xóa các vị trí token ảnh bị bỏ khỏi chuỗi; token giữ lại **mang theo tọa độ
   gốc** (hàng, cột) → LLM vẫn biết token đó nằm ở đâu trên ảnh.
3. Đặt 36 vector `v` được giữ vào chỗ của chúng; cộng DeepStack (đã fold) ở lớp
   0–2 như bình thường.
4. Chạy prefill một lần, rồi sinh câu trả lời từng token (greedy, tối đa 32 token).

Chuỗi đưa vào LLM ở ví dụ giảm từ khoảng 227 xuống khoảng 71 token.

---

## 9. Biến thể dễ diễn giải: PP3-fc (ADR 0004)

Mục đích: token giữ lại tạo thành **vùng gọn** đúng **chỗ câu hỏi nói tới**, thay
vì rải rác trên nền.

1. **Head focus (3d_contrast)**: cùng kiến trúc 3d nhưng học **nhãn tương phản**:
   lấy attention của câu hỏi này **trừ đi trung bình attention của các câu hỏi
   khác trên cùng ảnh**, giữ phần dương. Token nền mà câu hỏi nào cũng nhìn (sink)
   bị triệt tiêu; còn lại là phần **riêng** của câu hỏi.
2. **Chia budget**: 25% của K cho **focus**, 75% cho **context**.
3. **Focus**: "mọc vùng" — bắt đầu từ token có điểm focus cao nhất, mỗi bước chọn
   token có điểm cao nhất, **cộng thưởng** cho token nằm kề (8 hướng) vùng đã chọn
   → các vùng liền mạch.
4. **Context**: DPP như PP3 trên các token còn lại, với điểm của head 3d_attn50k.
5. Fold như PP3.

Kết quả: phần focus gọn (khoảng 1,8 cụm / 10 token, so với 3,8 của PP3 thường) và
phụ thuộc câu hỏi; ở L5 không mất accuracy, ở L3–L4 mất thêm 0,15–0,6 điểm.

Cấu hình: `configs/methods/pp3_scorer_fc.yaml` với
`--fc beta=1,share_min=0.25,share_max=0.25,context_dpp=1`.

---

## 10. Các phương pháp đem ra so sánh (ADR 0005)

| Phương pháp | Chọn token thế nào | Có học không | Fold |
|---|---|---|---|
| Random | ngẫu nhiên K token | không | có |
| **DivPrune** (2025) | chỉ chọn cho đa dạng nhất: lần lượt lấy token **xa nhất** với các token đã chọn; không xét câu hỏi | không | không |
| **Kiểu FastV** (2024) | chạy LLM full-token tới lớp 2, lấy K token **được chú ý nhiều nhất** | không | không |
| DSFold-D | DPP với chất lượng tự thiết kế `exp(z_rel + 0,5·z_ms)` | không | có |
| **PP3** | DPP với chất lượng do score head học được | **có** | có |

Khác biệt cốt lõi: PP3 có **cả** chất lượng phụ thuộc câu hỏi (học từ LLM) **và**
độ đa dạng (DPP). DivPrune chỉ có đa dạng; FastV chỉ có attention (và bị
attention sink kéo về nền).

---

## 11. Chi phí

- **Train (một lần)**: tạo nhãn 50k mẫu × khoảng 1,1 giây; train head trên đặc
  trưng đã lưu.
- **Mỗi câu hỏi khi chạy**: thêm head 2 triệu tham số và DPP. Thời gian selector
  khoảng 18 ms (L1) đến 36–48 ms (L3–L5) — số sơ bộ, sẽ đo lại (bước 5 runbook).
  Với ảnh GQA nhỏ (khoảng 230 token), phần prefill tiết kiệm được chưa bù được
  chi phí này; lợi ích chính là **giảm số token** (tính toán và bộ nhớ của LLM tỉ lệ
  với số token), rõ hơn khi ảnh lớn hoặc video.

---

## 12. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Token thị giác | một mảnh ảnh khoảng 32×32 pixel đã biến thành vector |
| N, K, K/N | số token ban đầu, số token giữ, tỉ lệ giữ |
| Budget, mức L1–L5 | tỉ lệ token được giữ |
| Full-token | Qwen gốc, giữ toàn bộ token (mốc so sánh) |
| Score head | mạng nhỏ chấm điểm từng token |
| Teacher / student | LLM full-token (cho nhãn) / score head (học theo) |
| Attention sink | token nền nhận attention lớn bất kể câu hỏi |
| DPP | cách chọn tập vừa chất lượng cao vừa đa dạng |
| DeepStack | 3 mức đặc trưng giữa của ViT được cộng vào LLM ở lớp 0–2 |
| Fold | gộp DeepStack của token bị bỏ vào token được giữ |
| Chênh lệch "−0,20" | kém full 0,20 điểm phần trăm accuracy |
| Non-inferiority | không kém hơn full quá một ngưỡng (ở đây 1 điểm) |

---

## 13. Tệp code chính

| Thành phần | Tệp |
|---|---|
| Điểm DSFold (`z_rel`, `z_ms`) | `src/token_reduction_vl/pruning/pp3_scorer/dsfold_base/scoring.py` |
| Đặc trưng + PCA | `src/token_reduction_vl/pruning/pp3_scorer/features.py` |
| Score head 3a/3b/3c/3d | `src/token_reduction_vl/pruning/pp3_scorer/models.py` |
| Teacher attention | `src/token_reduction_vl/pruning/pp3_scorer/attention_teacher.py` |
| Hàm mất mát, train | `pp3_scorer/losses.py`, `pp3_scorer/train.py` |
| DPP, fold | `pp3_scorer/dsfold_base/dpp.py`, `dsfold_base/fold.py` |
| Pruner PP3 | `pp3_scorer/pruner.py` |
| Focus + context | `pp3_scorer/focus_context.py` |
| Baseline | `pp3_scorer/baselines.py` |
| Đưa token vào LLM | `src/token_reduction_vl/models/qwen3_vl_pruned.py` |
