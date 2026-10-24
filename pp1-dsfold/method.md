# PP1 — DSFold không huấn luyện (toán thuần, r cố định)

- Trạng thái: Đề xuất
- Vai trò: phương pháp độc lập số 1. Đồng thời là **tài liệu nền mà mọi thành viên PP2–PP4 tự cài lại** trước khi làm phần mở rộng
- Bộ tham số mặc định **D** (dùng chung, không ai tự chỉnh): mục 8.0
- Bài kiểm tra "cài đúng PP1" (mọi thành viên phải qua): mục 13
- Bao nhiêu token: **r cố định**, 7 mức · Token nào: **công thức toán**
- Ký hiệu, cách lắp vào model, công thức chi phí, giao thức đánh giá: xem
  [`00-common/README.md`](../00-common/README.md) mục 2–9. File này chỉ mô tả phần riêng của PP1.
- Ánh xạ lên sơ đồ (M1–M8): [`docs/research/dsfold-method-proposal.md`](../docs/research/dsfold-method-proposal.md) mục 5

## 0. Tóm tắt nhanh

**Một câu:** sau khi ViT chạy xong, chọn K token vừa **đa dạng** vừa **liên
quan tới câu hỏi** bằng DPP; token bị bỏ không mất hẳn mà được **gộp vào kênh
DeepStack** của token giữ lại gần nó; sau đó đưa K token vào LLM với **toạ độ
M-RoPE gốc**.

**Không huấn luyện gì.** Chỉ có 5 siêu tham số (α, β, λ, γ, τ), chỉnh một lần
trên GQA val rồi khoá lại.

### 0.1 Các mức ngân sách dùng trong cách này

PP1 chạy **riêng từng mức**, cùng một r cho mọi ảnh:

| Mức | L7 | L6 | L5 | L4 | L3 | L2 | L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **r** | **100%** | **75%** | **50%** | **33,3%** | **22,2%** | **11,1%** | **5,6%** |
| K của ảnh GQA trung vị (N = 260) | 260 | 195 | 130 | 87 | 58 | 29 | 15 |
| K của ảnh MMBench trung vị (N = 176) | 176 | 132 | 88 | 59 | 40 | 20 | 10 |
| K của ảnh MME trung vị (N = 384) | 384 | 288 | 192 | 128 | 86 | 43 | 22 |

Số liệu đầy đủ (trung bình, P95, `N_total′`) nằm ở phần chung (`00-common`) mục 6.2.

### 0.2 Luồng dữ liệu

```mermaid
flowchart LR
  A["ViT (không đổi)<br/>v: N×2048<br/>f₁,f₂,f₃: 3×N×2048"] --> S1
  Q["Câu hỏi<br/>q: N_q×2048"] --> S1
  S1["S1 Score<br/>rel, ms → quality: N"] --> S2
  S2["S2 Greedy DPP<br/>I_keep: K chỉ số"] --> S3
  S3["S3 Assign<br/>a(d): N−K → K"] --> S4
  S4["S4 Fold<br/>f′: 3×K×2048"] --> S5
  S2 --> S5
  S5["S5 Lắp vào LLM<br/>input_ids: N_text+K<br/>position_ids: gather gốc"] --> LLM["28 decoder layer<br/>(không đổi)"]
```

### 0.3 Ví dụ số với ảnh GQA trung vị, ở L3 (r = 22,2%)

| Bước | Kết quả cụ thể |
| --- | --- |
| Ảnh sau resize | 640 × 416 pixel, lưới patch 40 × 26 = 1.040 patch, lưới token 20 × 13 |
| ViT | `v`: [260, 2048]; `f₁, f₂, f₃`: mỗi cái [260, 2048] |
| Câu hỏi | "Who is wearing the dress?", sau lọc stopword còn khoảng 3 token: "Who", " wearing", " dress" ⇒ `q`: [3, 2048] |
| S1 | 260 giá trị `rel`, 260 giá trị `ms`, rồi 260 giá trị `quality` |
| S2 | K = ⌈0,222 × 260⌉ = **58**. Greedy DPP chạy 58 bước. `I_keep` có 58 chỉ số, `I_drop` có 202 |
| S3 | 202 token bị bỏ chia cho 58 token giữ, trung bình khoảng 3,5 token mỗi nhóm |
| S4 | 58 × 3 vector DeepStack được fold. `v′ = v[I_keep]`: [58, 2048] |
| S5 | `N_total`: 286 → **84**. Tại 58 vị trí ảnh, M-RoPE vẫn mang toạ độ (h, w) gốc trên lưới 20 × 13 |
| Chi phí | LLM: 408 → 119 GMAC (−71%). ViT giữ nguyên 393 GMAC. Tổng prefill: 801 → 512 GMAC (−36%). KV cache: 32,0 → 9,4 MB |

## 1. Bài toán

**Cho:** ảnh đã qua ViT (`v`, `f_1`, `f_2`, `f_3`, mỗi tensor `[N, 2048]`), câu
hỏi (`q`), và tỉ lệ `r`.

**Tìm:**
1. `I_keep` gồm `K = clip(⌈r·N⌉, 1, N)` chỉ số;
2. `v′ = v[I_keep]` và `f′_l` có kích thước `[K, 2048]`.

**Mục tiêu:** câu trả lời của model khi chỉ thấy K token **giống nhất** với câu
trả lời khi thấy đủ N token.

**Ràng buộc:**
- không huấn luyện, không đổi trọng số;
- chi phí bộ chọn nhỏ hơn nhiều so với 1 decoder layer.

## 2. Tổng quan thuật toán

```
v, f₁, f₂, f₃, q, r
   │
   ├─ S1  Chấm điểm:      quality_i  (liên quan câu hỏi × multi-level saliency)
   ├─ S2  Chọn:           I_keep = greedy DPP(L = diag(quality)·S·diag(quality), K)
   ├─ S3  Gán:            mỗi d ∈ I_drop → a(d) ∈ I_keep
   ├─ S4  Fold:           f′_l[k] = norm-match(f_l[k] + γ·Σ w_dk·f_l[d])
   └─ S5  Lắp vào model:  phần chung (`00-common`) mục 4.2
```

## 3. S1 — Chấm điểm

### 3.1 Embedding câu hỏi `q`

Prompt do `build_question_prompt` (`src/token_reduction_vl/data/messages.py`) tạo ra có cấu trúc:

```
[Context: <hint>]  ·  <question>  ·  [Options: A. … B. …]  ·  <câu chỉ dẫn cố định>
```

Ta định nghĩa:

```
text_q = question + " " + (nối các choice.text, nếu có)        # KHÔNG gồm hint và câu chỉ dẫn
ids_q  = tokenizer(text_q, add_special_tokens=False).input_ids
ids_q  ← bỏ các token thuộc STOPWORDS  (danh sách cố định trong config, ví dụ:
          the, a, an, is, are, of, in, on, to, what, which, this, that, ?, ., ',')
q      = embed_tokens(ids_q)  ∈ R^{N_q×2048}                    # tra bảng embedding, fp32
```

- Phép tra embedding không phụ thuộc ngữ cảnh, nên tra riêng `text_q` cho
  **cùng vector** với các vị trí tương ứng trong prompt đầy đủ.
- Nếu sau khi lọc `N_q = 0` thì dùng lại `ids_q` chưa lọc.
- Ablation: không lọc stopword; có thêm hint.

### 3.2 Mức liên quan câu hỏi

```
rel_i = max_{j=1..N_q} cos(v_i, q_j)            i = 0..N−1
```

Lý do dùng `max` thay cho `mean`: một token ảnh chỉ cần khớp mạnh với **một**
từ quan trọng (ví dụ "ground") là đủ. Dùng `mean` sẽ bị các từ khác pha loãng.
Ablation: dùng `mean`.

### 3.3 Multi-level saliency `ms` (thành phần mới)

Các tầng DeepStack đi qua **merger khác nhau**, nên trước khi so sánh, mỗi tầng
được chuẩn hoá về cùng thang đo:

```
μ_l = mean_i f_l[i]  ∈ R^2048                    # trung bình theo token của ảnh này
σ_l = std của mọi phần tử (f_l − μ_l)            # một số vô hướng cho mỗi tầng
f̂_l = (f_l − μ_l) / (σ_l + 1e-6)
```

Ba biến thể; **mặc định là ms_A**:

| Biến thể | Công thức | Trực giác |
| --- | --- | --- |
| **ms_A** (mặc định) | `‖f̂_3[i] − f̂_2[i]‖₂ + ‖f̂_2[i] − f̂_1[i]‖₂` | đặc trưng còn biến đổi qua các tầng ⇒ vùng giàu thông tin |
| ms_B | `std_{l=1..3}( ‖f̂_l[i]‖₂ )` | độ nổi bật thay đổi theo tầng |
| ms_C | `(1/3)·Σ_l ‖f̂_l[i]‖₂` | token khác xa token trung bình của ảnh ở mọi tầng ⇒ nổi bật |

Đây là **giả thuyết H4**. Nếu cả ba biến thể đều không giúp thì đặt β = 0, và
báo cáo đó là một kết quả âm.

### 3.4 Gộp thành điểm chất lượng

```
z(x)_i    = clip( (x_i − mean(x)) / (std(x) + 1e-6), −3, 3 )      # z-score trong ảnh
quality_i = exp( α · z(rel)_i + β · z(ms)_i )                     # luôn > 0
```

- α = β = 0 cho `quality ≡ 1`, tức DPP chỉ còn tối đa độ đa dạng (gần với DivPrune).
- `exp` ở thang log-tuyến tính giúp α và β có ý nghĩa rõ: tăng α thêm 1 nghĩa là
  nhân `quality` thêm `e` lần cho mỗi độ lệch chuẩn của `rel`.

## 4. S2 — Chọn K token bằng DPP

### 4.1 Kernel

```
ṽ_i  = v_i / ‖v_i‖                         (fp32)
S    = ṽ ṽᵀ ∈ R^{N×N}                      # S_ii = 1, S là ma trận nửa xác định dương
L    = diag(quality) · S · diag(quality)    # L_ij = quality_i · quality_j · cos(v_i, v_j)
```

DPP chọn tập I với xác suất tỉ lệ với `det(L_I)`, trong đó:

```
det(L_I) = (Π_{i∈I} quality_i²) · det(S_I)
```

- Thừa số thứ nhất thưởng cho token có chất lượng cao.
- Thừa số thứ hai bằng bình phương "thể tích" do các vector `ṽ_i` tạo thành.
  Nó gần 0 khi các token gần trùng nhau, nên phạt sự trùng lặp.

### 4.2 Greedy MAP (Chen và cộng sự, NeurIPS 2018)

```
d2   ← quality²                           # d2_i = L_ii
C    ← zeros(N, K)
sel  ← []
for t = 0 .. K−1:
    j ← argmax_{i ∉ sel} d2_i
    if d2_j < 1e-6:                        # mọi token còn lại gần như trùng token đã chọn
        bổ sung K − t token còn lại theo quality giảm dần; break
    sel.append(j)
    e      ← ( L[j, :] − C[j, :t] · C[:, :t]ᵀ ) / √d2_j        # vector độ dài N
    C[:, t] ← e
    d2     ← d2 − e²
I_keep ← sort(sel)                        # sắp tăng dần để giữ thứ tự raster
```

- Độ phức tạp: `O(N·K²)` phép tính, bộ nhớ `O(N²)` cho S và `O(N·K)` cho C.
  - GQA trung vị (N = 260, K = 58 ở L3): khoảng 0,9 triệu phép tính, không đáng kể.
  - MME trung vị (N = 384, K = 288 ở L6): khoảng 32 triệu phép tính, không đáng kể.
  - MME lớn nhất (N = 16.320): riêng S đã cần khoảng **1 GB** fp32, và C cần
    thêm khoảng 0,76 GB ở L6. Như vậy **không khả thi** nếu tính DPP trực tiếp.
- `rank(S) ≤ 2048`. Nếu K vượt quá rank thực tế, `d2` về 0, và nhánh "bổ sung
  theo quality" xử lý trường hợp này.

### 4.2b Quy tắc cho ảnh lớn (bắt buộc, là một phần của phương pháp)

```
nếu N ≤ N_cap = 4.096:   DPP chính xác như trên
nếu N > N_cap:           max–min có trọng số (mục 4.3), không lập ma trận S:
                         mỗi bước chỉ cập nhật vector min_dist độ dài N
                         ⇒ bộ nhớ O(N), thời gian O(N·K)
```

- Tỉ lệ mẫu bị ảnh hưởng: ở GQA và MMBench là 0% (N tối đa là 400 và 256).
  Ở MME, khoảng trên 5% mẫu (P95 = 11.844). Phải báo cáo đúng con số này.
- Ablation: chạy DPP chính xác trên tập con các ảnh MME có N > 4.096 (nếu đủ
  VRAM) để đo mức chênh lệch so với max–min.

### 4.3 Bộ chọn thay thế (ablation A6)

| Bộ chọn | Quy tắc |
| --- | --- |
| **DPP** (mặc định) | như trên |
| Max–min có trọng số | `sel₀ = argmax quality`. Các bước sau: `j = argmax_{i∉sel} quality_i · min_{k∈sel}(1 − cos(v_i, v_k))` |
| Top-K | K token có `quality` cao nhất (không chống trùng lặp) |

## 5. S3 — Gán token bị bỏ cho token được giữ

```
dist(d, k) = √((row_d − row_k)² + (col_d − col_k)²) / √(H′² + W′²)        ∈ [0, 1]
a(d)       = argmax_{k ∈ I_keep} [ cos(v_d, v_k) − λ · dist(d, k) ]       d ∈ I_drop
A(k)       = { d ∈ I_drop : a(d) = k }                                     # "nhóm" của token k
```

- λ = 0 nghĩa là gán theo nội dung thuần.
- λ lớn nghĩa là gán vào token giữ lại gần nhất về vị trí.
- Chi phí: `O((N − K)·K)`.

## 6. S4 — Fold vào DeepStack

Với mỗi token được giữ k và mỗi tầng l ∈ {1, 2, 3}:

```
nếu A(k) = ∅:    f′_l[k] = f_l[k]
ngược lại:
    w_dk   = exp(cos(v_d, v_k)/τ) / Σ_{d'∈A(k)} exp(cos(v_d', v_k)/τ)      d ∈ A(k)    (Σ_d w_dk = 1)
    f̃_l[k] = f_l[k] + γ · Σ_{d∈A(k)} w_dk · f_l[d]
    f′_l[k] = f̃_l[k] · ‖f_l[k]‖₂ / ‖f̃_l[k]‖₂                               # norm-match
v′ = v[I_keep]                                                           # main stream giữ nguyên
```

Giải thích từng phần:
- **Trọng số `w`** tổng bằng 1, nên `Σ w·f_l[d]` là **trung bình có trọng số**
  của nhóm. γ điều khiển "thêm bao nhiêu" so với chính token k.
- **Norm-match** giữ độ lớn residual DeepStack đúng như model từng thấy lúc
  huấn luyện. Chỉ **hướng** của vector thay đổi.
- γ = 0 cho Select thuần.

Ablation A3 so bốn biến thể:

| Biến thể | `v′` | `f′_l` |
| --- | --- | --- |
| none | `v[I_keep]` | `f_l[I_keep]` |
| **fold-DS** (mặc định) | `v[I_keep]` | công thức trên |
| merge-v | `norm-match(v_k + γ_v·Σ w_dk v_d)` | `f_l[I_keep]` |
| both | merge-v | fold-DS |

## 7. S5 — Lắp vào model

Theo đúng phần chung (`00-common`) mục 4.2 (bước 3–6), với `v′` và `f′_l` vừa tính.

## 8. Siêu tham số và cách chọn

### 8.0 Bộ tham số mặc định D (chốt, dùng chung cho cả 4 thành viên)

Mọi thành viên cài PP1 với **đúng** bộ tham số này. Thành viên PP2–PP4 dùng
bộ D cho phần PP1 bên trong phương pháp của mình và **không tự chỉnh**. Nhờ vậy:
- 4 bản cài PP1 cho cùng kết quả (mục 13);
- khác biệt giữa 4 phương pháp chỉ đến từ phần mở rộng.

| Tham số | Giá trị D | Ý nghĩa |
| --- | --- | --- |
| α | 1 | trọng số mức liên quan câu hỏi |
| β | 0,5 | trọng số multi-level saliency |
| biến thể `ms` | A | `‖f̂₃ − f̂₂‖ + ‖f̂₂ − f̂₁‖` |
| λ | 0,5 | phạt khoảng cách khi gán |
| γ | 0,5 | mức fold |
| τ | 0,1 | nhiệt độ softmax khi fold |
| `N_cap` | 4.096 | trên ngưỡng này dùng max–min thay cho DPP |
| clip z-score | [−3, 3] | |
| ngưỡng `d2` | 1e-6 | dừng DPP và bổ sung theo quality |
| STOPWORDS | danh sách trong `configs/prune_dsfold.yaml` (mục 3.1) | phải giống hệt nhau giữa các thành viên |
| dtype tính điểm | fp32 (`v`, `f`, `q` chuyển sang fp32 trước S1) | |
| phá hoà khi argmax | chọn chỉ số **nhỏ nhất** | để mọi bản cài ra cùng `I_keep` |

Bộ D được ghi vào `configs/prune_dsfold.yaml` (phiên bản `D-1.0`). Mọi kết quả
của cả 4 phương pháp phải ghi lại phiên bản bộ D đã dùng.

### 8.1 Chỉnh tham số (chỉ thành viên PP1 làm)

Thành viên PP1 chỉnh tham số để báo cáo **kết quả tốt nhất của riêng PP1**.
Kết quả chỉnh được lưu thành bộ `D-tuned` riêng; bộ D **không đổi**.

| Tham số | Lưới tìm kiếm | Giá trị D |
| --- | --- | --- |
| α | {0; 0,5; 1; 2} | 1 |
| β | {0; 0,5; 1} | 0,5 |
| biến thể `ms` | {A, B, C} | A |
| λ | {0; 0,25; 0,5; 1} | 0,5 |
| γ | {0; 0,25; 0,5; 1} | 0,5 |
| τ | {0,05; 0,1; 0,2} | 0,1 |

**Giao thức chọn** (tránh chỉnh quá tay, và không chạm vào tập đánh giá):
1. Tập chỉnh: 2.000 câu GQA **val**, seed 42. Chỉ chỉnh ở **r = 22,2%**.
2. Bước 1, với γ = 0: lưới (α, β) gồm 4 × 3 = 12 lần chạy, cộng biến thể `ms` ở
   β tốt nhất: thêm 2 lần.
3. Bước 2: lưới (λ, γ) gồm 16 lần chạy với τ = 0,1, rồi τ: thêm 2 lần.
4. Tổng khoảng 32 lần × 2.000 câu × khoảng 0,12 s ≈ **2,1 giờ GPU**.
5. **Khoá** kết quả thành bộ `D-tuned`, dùng chung cho **mọi** mức r và mọi benchmark của riêng PP1. Bộ D không đổi.
6. Chỉ tiêu chọn: `agree` (phần chung (`00-common`) mục 8.2), hoà thì xét accuracy.

## 9. Thí nghiệm

### 9.1 Ma trận chạy chính

| Giai đoạn | Mức | Phương pháp | Dữ liệu |
| --- | --- | --- | --- |
| Dev | L7, L6, L5, L4, L3, L2, L1 (cả 7 mức) | Full, P0, Random-K ×3, Grid-K, Pool k×k, FastV k = 2/3, DivPrune, CDPruner, **DSFold** | dev subset 1.000 câu × 3 benchmark |
| Full | L6 (75%), L5 (50%), L4 (33,3%), L3 (22,2%), L2 (11,1%) | các phương pháp tốt nhất từ giai đoạn dev, cộng P0, Random-K và **DSFold** | toàn bộ GQA testdev, MMBench val, MME test |

Với MME, mọi bảng kết quả có thêm 3 cột chia theo kích thước ảnh: N ≤ 512,
512 < N ≤ 4.096, N > 4.096 (phần chung (`00-common`) mục 5.1).

### 9.2 Ablation (trên GQA testdev và MME, ở r ∈ {33,3; 11,1}%)

| Ký hiệu | So sánh | Giả thuyết |
| --- | --- | --- |
| A1 | DeepStack: `f_l[I_keep]` / đặt `f_l = 0` (bỏ DeepStack) | H1: cắt đúng chỉ số tốt hơn bỏ |
| A2 | M-RoPE: toạ độ gốc / đánh số lại thành lưới dày `⌈√K⌉ × ⌈√K⌉` | H2: giữ toạ độ gốc tốt hơn |
| A3 | none / fold-DS / merge-v / both | H3: fold-DS tốt nhất ở r ≤ 22,2% |
| A4 | β = 0 / ms_A / ms_B / ms_C | H4: `ms` có ích |
| A5 | α = 0 / `max` / `mean` / có hint / không lọc stopword | vai trò của câu hỏi |
| A6 | DPP / max–min / top-K | vai trò của chống trùng lặp |

### 9.3 Phân tích định tính

- Bản đồ token giữ lại trên ảnh (tô màu ô 32×32), thêm vào failure gallery.
- Kiểm tra position bias: histogram `row_j` của token được chọn, so với phân bố
  đều. FastV được biết là thiên về token phía dưới ảnh (FEATHER, 2412.13180).
- MME theo category: category nào giảm mạnh nhất khi cắt (OCR, count, position …).

## 10. Chi phí lúc suy luận

| Phần | Chi phí |
| --- | --- |
| `rel`: N × N_q tích vô hướng | không đáng kể |
| `ms`: 6N phép chuẩn | không đáng kể |
| `S`: N × N × 2048 | GQA trung vị (N = 260): khoảng 0,14 GMAC. MME trung vị (N = 384): khoảng 0,30 GMAC. So với LLM prefill 408 và 599 GMAC, dưới 0,1% |
| Greedy DPP | `O(N·K²)`, tuần tự K bước (K ≤ 288 ở ảnh trung vị). Phải đo thời gian thật |
| Max–min (N > 4.096) | `O(N·K)`, ví dụ N = 11.844, K = 2.632: khoảng 3·10⁷ phép tính mỗi chiều |
| S3 + S4 | `O((N − K)·K)` + `O(N·2048·3)` |

Nếu đo thấy greedy DPP vẫn chậm ở L6 hoặc L5 trên ảnh trung bình, hạ `N_cap`
xuống (ví dụ 2.048) và ghi rõ trong báo cáo.

## 11. Tiêu chí thành công

| Mốc | Tiêu chí đo được |
| --- | --- |
| M1 | `r = 100%` trùng baseline 100/100 mẫu mỗi benchmark |
| M2 | có đường Pareto của các đối chứng ở 7 mức trên dev subset |
| M3 | FastV / DivPrune / CDPruner cho xu hướng giống paper (giảm nhẹ ở 33,3%, giảm mạnh ở 5,6%) |
| M4 | có kết luận A1, A2 với CI 95% |
| M5 | DSFold thắng phương pháp tốt nhất ở r ∈ {22,2; 11,1}% trên ≥ 2/3 benchmark, và CI của chênh lệch không chứa 0 |

## 12. Hạn chế đã biết

- ViT vẫn chạy đủ, nên lợi ích chỉ nằm ở LLM prefill và KV cache (phần chung (`00-common`) mục 5).
- `rel` dùng embedding đầu vào, tức một không gian khá "nông". Nó có thể yếu với
  câu hỏi trừu tượng.
- `ms` và Fold là giả thuyết. Bộ tham số khoá ở r = 22,2% có thể không tối ưu ở
  mức khác; ta chấp nhận điều này để tránh chỉnh tham số theo từng mức.

## 13. Bài kiểm tra "cài đúng PP1" (bắt buộc cho cả 4 thành viên)

4 thành viên tự cài PP1 theo cùng tài liệu này, nên phải kiểm tra rằng 4 bản
cài **cho cùng kết quả**. Nếu không, so sánh giữa 4 phương pháp sẽ không còn
công bằng.

**Bộ kiểm tra chung** (một người tạo một lần, lưu trong `tests/fixtures/pp1_conformance/`):
- 100 mẫu cố định: 40 GQA testdev, 30 MMBench val, 30 MME test (seed 42). Trong
  30 mẫu MME có ít nhất 3 mẫu N > 4.096, để kiểm tra nhánh max–min;
- với mỗi mẫu, lưu `v`, `f₁..f₃` (fp16), `grid_hw`, `question_ids`;
- kết quả tham chiếu từ bản cài đầu tiên đã được review: `keep_indices` và
  `deepstack_kept` ở 3 mức L5, L3, L1.

**Tiêu chí đạt:**

| Kiểm tra | Tiêu chí |
| --- | --- |
| `keep_indices` | **trùng 100%** với tham chiếu ở cả 3 mức, cho cả 100 mẫu |
| `deepstack_kept` (sau Fold) | sai số tuyệt đối lớn nhất ≤ 1e-3 (fp32) |
| `v_kept` | trùng chính xác (`v[keep_indices]`) |
| Câu trả lời end-to-end ở L3 (cần wrapper) | trùng tham chiếu ≥ 99/100 mẫu (cho phép lệch do bf16 trên GPU khác) |

Bài kiểm tra này được viết thành `pytest` để chạy được trên CPU. Thành viên nào
chưa qua thì **chưa được** dùng PP1 của mình để làm phần mở rộng hay báo cáo.
