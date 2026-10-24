# Bốn cách thực hiện giảm visual token — tổng quan và phần dùng chung

- Trạng thái: **Đề xuất** (chưa triển khai; phương pháp cuối cùng sẽ được ghi vào ADR 0002)
- Cập nhật: 2026-10-24
- Model: `Qwen/Qwen3-VL-2B-Instruct`, snapshot `89644892e4d85e24eaac8bacfd4f463576704203`, Transformers 5.x
- Vị trí can thiệp chung: **P2**, sau Final 2×2 PatchMerger, trước `masked_scatter`
  (sơ đồ: [`architecture/dsfold-proposal/`](../../../architecture/dsfold-proposal/))

File này chứa mọi thứ **dùng chung** cho cả 4 cách: ký hiệu, thông số model,
cách lắp token đã cắt vào model, công thức chi phí, mức ngân sách, giao thức dữ
liệu và đánh giá. Mỗi file cách 1–4 chỉ mô tả phần riêng của nó và tham chiếu
về đây.

## 1. Hai câu hỏi và bốn cách

Mọi phương pháp cắt token đều trả lời hai câu hỏi độc lập:

1. **Giữ bao nhiêu token?** Đó là ngân sách `K`, hay tỉ lệ giữ `r = K / N`.
2. **Giữ những token nào?** Đó là tập chỉ số `I_keep ⊂ {0, …, N−1}`, với `|I_keep| = K`.

| | **Token nào: công thức toán** | **Token nào: mạng học** |
| --- | --- | --- |
| **Bao nhiêu: r cố định** | [Cách 1 — DSFold không huấn luyện](01-training-free-dsfold.md) | [Cách 3 — Score head](03-learned-token-scorer.md) |
| **Bao nhiêu: mạng học** | [Cách 2 — DSFold-A (budget head)](02-learned-budget-dsfold-a.md) | [Cách 4 — Học toàn bộ](04-fully-learned.md) |

| | Cách 1 | Cách 2 | Cách 3 | Cách 4 |
| --- | --- | --- | --- | --- |
| Tham số học thêm | 0 | ≈ 0,66M | ≈ 2,1M (3a) / 3,5M (3b) / 2,6M (3c) | ≈ 2,6M |
| Gradient qua model 2B | không | không | tuyến B: chỉ gradient theo đầu vào, 1 lần mỗi mẫu; tuyến A: có | có |
| Chi phí GPU thêm (ước tính) | ≈ 2 giờ chỉnh tham số | ≈ 9 giờ tạo nhãn | ≈ 6 giờ (B) / ≈ 12 giờ thêm (A) | ≈ 3–5 ngày |
| K thích ứng theo mẫu | không | **có** | không | **có** |
| Rủi ro | thấp | thấp–trung bình | trung bình–cao | cao |
| Vai trò trong khoá luận | **bắt buộc** | **phần "adaptive" chính** | mở rộng | hướng phát triển |

Mọi con số chi phí là **ước tính** từ thời gian của baseline (GQA ≈ 0,155 s/mẫu,
MME ≈ 1,61 s/mẫu, peak VRAM 4,1–7,6 GiB). Cần đo lại trên GPU thực tế trước khi
lập lịch.

## 2. Ký hiệu

| Ký hiệu | Kích thước / kiểu | Định nghĩa |
| --- | --- | --- |
| `H×W` | pixel | ảnh sau khi processor resize (bội số của 32) |
| `G_h = H/16`, `G_w = W/16` | số nguyên | lưới patch (`image_grid_thw = [1, G_h, G_w]`) |
| `N_patch = G_h·G_w` | số nguyên | số patch vào ViT |
| `H′ = G_h/2`, `W′ = G_w/2` | số nguyên | lưới visual token sau 2×2 merge |
| `N = N_visual = H′·W′` | số nguyên | số visual token (bằng `N_patch / 4`) |
| `j ∈ {0, …, N−1}` | chỉ số | thứ tự token, **row-major trên lưới H′×W′**: `j = row·W′ + col` |
| `(row_j, col_j)` | số nguyên | `row_j = ⌊j / W′⌋`, `col_j = j mod W′` |
| `v ∈ R^{N×2048}` | bf16 | main visual embeddings (đầu ra Final PatchMerger, `pooler_output`) |
| `f_l ∈ R^{N×2048}`, `l = 1, 2, 3` | bf16 | DeepStack features, tap sau ViT block có chỉ số 0-based 5, 11, 17 |
| `x_text` | danh sách token | các token chữ và token đặc biệt trong prompt |
| `Q ⊂ chỉ số` | tập | vị trí các token **câu hỏi** trong `input_ids` (không gồm template/token đặc biệt) |
| `e_t ∈ R^{2048}` | | embedding đầu vào (bảng embedding) của token t |
| `q = {e_t : t ∈ Q} ∈ R^{N_q×2048}` | | embedding các token câu hỏi |
| `N_text` | số nguyên | số token không phải ảnh trong prompt |
| `N_total = N_text + N` | số nguyên | độ dài `input_ids` (cột `input_tokens` của baseline) |
| `r ∈ (0, 1]` | số thực | tỉ lệ giữ |
| `K = clip(⌈r·N⌉, 1, N)` | số nguyên | số visual token giữ lại |
| `I_keep` | danh sách tăng dần | chỉ số giữ lại, **luôn sắp tăng dần** để giữ thứ tự raster |
| `I_drop = {0..N−1} \ I_keep` | | chỉ số bị bỏ |
| `N_total′ = N_text + K` | | độ dài chuỗi sau khi cắt |
| `cos(a, b) = aᵀb / (‖a‖·‖b‖)` | | tính bằng fp32 |

Giả định: **mỗi mẫu có đúng 1 ảnh** (đúng với GQA, MMBench, MME trong
`data/processed/v1`). Wrapper phải `assert` điều này.

## 3. Thông số model (đọc từ `config.json` của checkpoint)

| Thành phần | Giá trị |
| --- | --- |
| Processor | `min_pixels = 65.536`, `max_pixels = 16.777.216`, patch 16, merge 2, temporal patch 2, mean = std = 0,5 |
| ViT | 24 block, hidden 1024, FFN 4096, 16 head, GELU-tanh, position embedding học 2304 (48×48, nội suy) + 2D RoPE |
| DeepStack | tap tại block 0-based `[5, 11, 17]`; mỗi tap có merger riêng: gom 2×2 → 4096 → LayerNorm(4096) → Linear 4096→4096 → GELU → Linear 4096→2048 |
| Final merger | LayerNorm(1024) cho từng patch → gom 2×2 → 4096 → Linear 4096→4096 → GELU → Linear 4096→2048 |
| LLM | 28 layer, hidden 2048, FFN 6144 (SwiGLU), 16 Q head / 8 KV head, head dim 128, RMSNorm ε = 1e-6 |
| M-RoPE | `mrope_section = [24, 20, 20]` (interleaved), `rope_theta = 5.000.000` |
| Từ vựng | 151.936 (LM head dùng chung trọng số với embedding) |
| Token ID | `<|image_pad|>` = 151655, `<|vision_start|>` = 151652, `<|vision_end|>` = 151653 |

## 4. Phần cài đặt chung: lắp token đã cắt vào model

Phần này **giống nhau cho cả 4 cách**. Đầu vào là `I_keep` và các tensor
`v′ ∈ R^{K×2048}` cùng `f′_l ∈ R^{K×2048}`. Mỗi cách tạo ra chúng theo kiểu
riêng; chỉ cách nào có Fold thì `f′_l` mới khác `f_l[I_keep]`.

### 4.1 Vì sao không dùng thẳng `model.generate()`

`generate()` tự tính lại M-RoPE từ `input_ids` và `image_grid_thw` bằng
`get_rope_index`. Hàm này giả định số token ảnh bằng `N` (đủ lưới), nên sẽ
**sai** khi chỉ còn K token. Wrapper vì vậy tự làm prefill và decode, và tự
cung cấp `position_ids`.

### 4.2 Các bước

**Bước 0 — Baseline đầy đủ cho phần chữ.** Gọi processor như baseline để có
`input_ids` (độ dài `N_total`), `attention_mask`, `mm_token_type_ids` (0 = chữ,
1 = ảnh), `pixel_values`, `image_grid_thw`.

**Bước 1 — Chạy ViT.** Gọi `model.model.get_image_features(pixel_values, image_grid_thw)`.
Kết quả gồm `v = pooler_output[0]` có kích thước `[N, 2048]` và
`f_l = deepstack_features[l][0]` có kích thước `[N, 2048]`.

**Bước 2 — Tính M-RoPE đầy đủ.** Gọi
`P, _ = model.model.get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask=attention_mask)`
để có `P ∈ Z^{3×1×N_total}` cho chuỗi **chưa cắt**. Theo code Qwen3-VL:

```
gọi s là vị trí bắt đầu của khối ảnh (= số token chữ đứng trước ảnh)
token ảnh j:        P[:, j_seq] = ( s,  s + row_j,  s + col_j )      # (t, h, w)
token chữ sau ảnh:  vị trí tiếp tục từ  s + max(G_h, G_w) / 2  (tăng 1 mỗi token, giống nhau ở 3 trục)
```

**Bước 3 — Chọn cột cần giữ.** Gọi `img_pos` là các vị trí trong chuỗi có
`input_ids == 151655` (có N vị trí, theo đúng thứ tự j). Ta giữ:

```
keep_seq = [mọi vị trí chữ]  ∪  [img_pos[j] for j in I_keep]     (sắp tăng dần)
input_ids′ = input_ids[:, keep_seq]          # còn K token 151655
P′         = P[:, :, keep_seq]               # GIỮ NGUYÊN toạ độ gốc, không đánh số lại
attn′      = attention_mask[:, keep_seq]
```

**Bước 4 — Embedding và fusion.**

```
E = embed_tokens(input_ids′)                          # [1, N_total′, 2048]
mask = (input_ids′ == 151655)                         # K vị trí
E[mask] = v′                                          # tương đương masked_scatter
```

**Bước 5 — Prefill.** Gọi `model.model.language_model(inputs_embeds=E, position_ids=P′, attention_mask=attn′, visual_pos_masks=mask, deepstack_visual_embeds=[f′_1, f′_2, f′_3], use_cache=True)`,
rồi tính logit của vị trí cuối bằng `lm_head`.

**Bước 6 — Decode.** Với bước sinh thứ `s = 0, 1, …` (tối đa 32 token, dừng ở EOS `151645`):

```
rope_delta′ = max(P′) + 1 − N_total′
vị trí của token sinh thứ s:  (N_total′ + s) + rope_delta′  =  max(P′) + 1 + s   (cùng giá trị cho 3 trục)
```

Vì `max(P′) = max(P)` (token chữ cuối cùng luôn được giữ), token sinh ra nhận
**đúng vị trí như baseline**. Chỉ phần ảnh ở giữa thưa đi. Đây là lý do việc
"giữ toạ độ gốc" nhất quán với phần decode.

**Bước 7 — Kiểm chứng bắt buộc (unit + smoke test).**
- `r = 100%`, không fold ⇒ `I_keep = {0..N−1}`, `P′ = P`, và câu trả lời phải
  **trùng từng ký tự** với `outputs/baseline/*.jsonl` trên 100 mẫu mỗi benchmark
  (greedy, bf16, cùng GPU).
- Kiểm tra `mask.sum() == K` và `f′_l.shape[0] == K`.
- Kiểm tra `P′` của các token ảnh giữ lại bằng đúng `(s, s + row_j, s + col_j)`.

## 5. Công thức chi phí (đếm MAC, nhân–cộng)

Chỉ đếm các phép nhân ma trận; bỏ qua norm, activation và softmax.

**ViT (không đổi ở P2):**
```
linear_ViT     = 24 · N_patch · (4·1024² + 2·1024·4096)        ≈ N_patch · 302M
attention_ViT  = 24 · 2 · N_patch² · 1024                        (attention đầy đủ trong mỗi ảnh)
mergers        = 4 · N · (4096·4096 + 4096·2048)                ≈ N · 101M
```

**LLM prefill với độ dài `n` (`n = N_total` hoặc `N_total′`):**
```
linear_LLM     = 28 · n · (2·2048·2048 + 2·2048·1024 + 3·2048·6144)   ≈ n · 1,409G
attention_LLM  ≈ 28 · 2 · n² · 2048 / 2                              (causal ⇒ khoảng một nửa)
lm_head        = 2048 · 151.936 ≈ 311M   (chỉ vị trí cuối)
```

**Decode bước s** (KV cache): khoảng `1,409G + 28 · 2 · (n + s) · 2048 + 311M`.

**KV cache:** `28 layer × 2 (K, V) × 8 head × 128 × n × 2 byte` ≈ `n × 114,7 KB`.

Báo cáo **riêng từng phần** (ViT, LLM prefill, decode) và tổng. Không được gọi
"giảm x% token" là "giảm x% thời gian".

## 6. Mức ngân sách cố định (7 mức)

| Mức | `r` | Tương ứng LLaVA (576 token) | Mục đích |
| --- | ---: | ---: | --- |
| L7 | 100% | 576 | kiểm chứng: trùng baseline |
| L6 | 75% | 432 | cắt nhẹ: tìm điểm chất lượng bắt đầu giảm |
| L5 | 50% | 288 | mức các paper Qwen2.5-VL hay báo cáo |
| L4 | 33,3% | 192 | chuẩn paper LLaVA |
| L3 | 22,2% | 128 | chuẩn paper LLaVA |
| L2 | 11,1% | 64 | chuẩn paper LLaVA |
| L1 | 5,6% | 32 | cắt rất mạnh |

- `K = clip(⌈r·N⌉, 1, N)`, tính riêng cho từng ảnh.
- **Quét dày trên tập dev** trước khi chạy full: r ∈ {100, 90, 75, 60, 50, 40,
  33,3, 25, 22,2, 15, 11,1, 5,6}% để tìm điểm gãy.
- Mức trần tuyệt đối {64, 128, 256} token chỉ dùng làm **phân tích phụ**.
- Cách 2 và cách 4 có K thay đổi theo mẫu, nên được so ở cùng **tỉ lệ giữ trung
  bình thực tế** `r̄ = (Σ K) / (Σ N)` trên từng benchmark.

## 7. Giao thức dữ liệu (theo `configs/data_v1.yaml`)

| Vai trò | Split | Dùng cho |
| --- | --- | --- |
| Huấn luyện | GQA train balanced | tạo nhãn và huấn luyện mọi mạng nhỏ (cách 2–4) |
| Chỉnh tham số | GQA val balanced | chọn siêu tham số, ngưỡng, λ; early stopping |
| Dev subset | 1.000 mẫu ngẫu nhiên cố định (seed 42) từ mỗi split đánh giá | **chỉ** để quét nhanh và so sánh sơ bộ; không chỉnh tham số trên đây |
| Đánh giá | GQA testdev, MMBench validation, MME test | số liệu báo cáo cuối cùng |

- Tách train/early-stop trong GQA train **theo `image_id`**, để không có ảnh
  nào xuất hiện ở cả hai phía.
- MMBench và MME **không bao giờ** được dùng để huấn luyện hay chỉnh tham số.
  Kết quả trên chúng là đánh giá ngoài miền.

## 8. Giao thức đánh giá

### 8.1 Chỉ số chất lượng

Dùng đúng code chấm điểm hiện có (`src/token_reduction_vl/evaluation/metrics.py`):
- GQA: normalized exact match;
- MMBench: nhãn A–E sau khi parse;
- MME: question accuracy, Accuracy+ theo cặp, và MME score.

Không lấy trung bình giữa các benchmark.

### 8.2 Chỉ số khớp với baseline

```
agree = (1/M) · Σ_i 1[ norm(pred_pruned_i) == norm(pred_full_i) ]
```

`norm` dùng cùng hàm chuẩn hoá hoặc parse của từng benchmark. Chỉ số này đo
"giữ được hành vi của model gốc", độc lập với đáp án đúng.

### 8.3 Chỉ số hiệu năng

Báo cáo:
- `r̄` thực tế, `N_total′` trung bình;
- MAC theo mục 5;
- thời gian ViT / bộ chọn token / prefill / decode, đo bằng `torch.cuda.Event`,
  warm-up 10 mẫu, báo median và P95;
- peak VRAM.

### 8.4 Thống kê

**Paired cluster bootstrap theo ảnh:**
1. Lấy mẫu lại có hoàn lại các `image_id`, B = 10.000 lần.
2. Mỗi lần tính `Δ = metric(pruned) − metric(full)` trên cùng tập ảnh.
3. Báo CI 95% bằng phân vị 2,5% và 97,5%.
4. Một khác biệt chỉ được gọi là "có ý nghĩa" khi CI không chứa 0.

Với MME, đơn vị lấy mẫu lại là ảnh, vì Accuracy+ được tính theo cặp câu hỏi của
cùng một ảnh.

### 8.5 Ghi nhận thí nghiệm

Mỗi lần chạy ghi lại:
- config YAML;
- git SHA;
- model và dataset revision;
- phiên bản Transformers và PyTorch;
- GPU và driver;
- seed;
- `report.id` riêng.

Không ghi đè báo cáo baseline.

## 9. Đối chứng dùng chung (ở mọi mức K)

| Tên | Định nghĩa chính xác |
| --- | --- |
| Full | baseline, `r = 100%` |
| **P0 resolution-matched** | chọn `max_pixels′` sao cho N mới ≈ K. Vì `N = H·W / 32²`, nên `max_pixels′ = K · 1024` pixel. Đồng thời phải hạ `min_pixels` (mặc định 65.536 ⇒ không bao giờ dưới 64 token) xuống `min(65.536, K · 1024)`. Báo cáo N thực tế, vì lưới bị làm tròn |
| Random-K | chọn K chỉ số đều ngẫu nhiên, 3 seed {0, 1, 2}; báo cáo trung bình ± độ lệch chuẩn |
| Grid-K | `a_h = max(1, round(√(K·H′/W′)))`, `a_w = ⌈K / a_h⌉`. Lấy tâm ô `((i + 0,5)·H′/a_h, (k + 0,5)·W′/a_w)`, chọn token gần nhất với mỗi tâm, bỏ trùng. Nếu chưa đủ K thì bổ sung ngẫu nhiên (seed 0); nếu thừa thì bỏ ngẫu nhiên |
| Pool k×k | average pooling `v` và `f_l` trên lưới H′×W′ với kernel = stride = k. k = 2, 3, 4 cho `r` = 25%, 11,1%, 6,25%. Toạ độ M-RoPE lấy theo token ở góc trên-trái của mỗi ô. So với mức gần nhất và ghi rõ `r̄` thực tế |
| FastV | cắt sau LLM layer k ∈ {2, 3} theo attention trung bình mà token cuối dành cho mỗi visual token. Nếu cắt **trước** khi DeepStack #l được cộng (sau layer l−1), phải cắt `f_l` theo cùng chỉ số |
| DivPrune | max–min diversity trên `v` (cosine) |
| CDPruner | DPP có điều kiện theo câu hỏi (như paper) |

## 10. Thứ tự thực hiện

```
M1 wrapper + kiểm chứng (mục 4.2 bước 7)
  → Cách 1 (M2–M5)
  → Cách 2 (M6)
  → Cách 3, tuyến B (M7, nếu còn thời gian)
  → Cách 4: chỉ thiết kế, trình bày ở phần hướng phát triển
```
