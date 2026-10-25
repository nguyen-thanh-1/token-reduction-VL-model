# Phần dùng chung cho cả 4 phương pháp

- Trạng thái: **Đề xuất** (chưa triển khai; phương pháp được chọn sẽ ghi vào ADR 0002)
- Cập nhật: 2026-10-24
- Model: `Qwen/Qwen3-VL-2B-Instruct`, snapshot `89644892e4d85e24eaac8bacfd4f463576704203`, Transformers 5.x
- Vị trí can thiệp chung: **P2**, sau Final 2×2 PatchMerger, trước `masked_scatter`
  (sơ đồ: [`architecture/dsfold-proposal/`](../architecture/dsfold-proposal/))
- Chỉ mục 4 phương pháp và cách tổ chức nhóm: [`../METHODS.md`](../METHODS.md)

> **7 mức ngân sách:** L7 = 100% · L6 = 75% · L5 = 50% · L4 = 33,3% · L3 = 22,2% · L2 = 11,1% · L1 = 5,6%.
> Bảng đầy đủ kèm số token thật của từng benchmark nằm ở [mục 6](#6-các-mức-ngân-sách-7-mức-cố-định).

**Mọi thành viên phải đọc file này trước.** File chứa những thứ **giống nhau
cho cả 4 phương pháp**, để kết quả so sánh được với nhau:
- ký hiệu và thông số model;
- wrapper lắp token đã cắt vào model;
- công thức chi phí và 7 mức ngân sách;
- giao thức dữ liệu và đánh giá;
- các đối chứng.

Không thành viên nào được tự đổi những phần này. Muốn đổi thì phải thống nhất cả
nhóm và cập nhật file này.

## 1. Hai câu hỏi mà mọi phương pháp phải trả lời

1. **Giữ bao nhiêu token?** Đó là ngân sách `K`, hay tỉ lệ giữ `r = K / N`.
2. **Giữ những token nào?** Đó là tập chỉ số `I_keep ⊂ {0, …, N−1}`, với `|I_keep| = K`.

Mỗi phương pháp trả lời hai câu hỏi này theo cách riêng (xem bảng trong
[`../METHODS.md`](../METHODS.md)). Nhưng **mọi phương pháp đều xuất ra cùng một
dạng kết quả** (mục 4.0), để dùng chung wrapper và giao thức đánh giá.

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

Phần này **giống nhau cho cả 4 phương pháp**. Đầu vào là `I_keep` và các tensor
`v′ ∈ R^{K×2048}` cùng `f′_l ∈ R^{K×2048}`. Mỗi phương pháp tạo ra chúng theo
kiểu riêng.

### 4.0 Interface bắt buộc cho mọi phương pháp

Mọi phương pháp cài một lớp theo protocol `VisualTokenPruner` (mở rộng file
`src/token_reduction_vl/pruning/base.py`). Lớp đó có đúng một hàm `select`:

```python
@dataclass(frozen=True)
class PrunerInput:
    v: Tensor                  # [N, 2048]  main visual embeddings
    deepstack: list[Tensor]    # 3 × [N, 2048]  f₁, f₂, f₃
    grid_hw: tuple[int, int]   # (H′, W′); token j nằm ở (j // W′, j % W′)
    question_ids: Tensor       # [N_q] token id của câu hỏi + lựa chọn (định nghĩa ở PP1 mục 3.1), CHƯA lọc stopword
    question_tokens: tuple[str, ...]  # chuỗi giải mã của từng id, để PP1 lọc stopword
    embed_tokens: nn.Embedding # bảng embedding của LLM (chỉ đọc)
    ratio: float | None        # r cố định (PP1, PP3); None nếu phương pháp tự chọn K (PP2, PP4)

@dataclass(frozen=True)
class PrunerOutput:
    keep_indices: Tensor       # [K], int64, SẮP TĂNG DẦN, không trùng, 1 ≤ K ≤ N
    v_kept: Tensor             # [K, 2048]  đưa vào masked_scatter
    deepstack_kept: list[Tensor]  # 3 × [K, 2048]  đưa vào decoder layer 0/1/2
    metadata: dict             # bắt buộc: "K", "N", "method", "ratio_target"; tuỳ chọn: thời gian, điểm số …

class VisualTokenPruner(Protocol):
    name: str
    def select(self, x: PrunerInput) -> PrunerOutput: ...
```

Wrapper (mục 4.2) **chỉ** gọi `select`, rồi làm phần còn lại. Nhờ vậy:
- 4 phương pháp chạy cùng một code suy luận và đánh giá;
- một phương pháp đúng interface có thể cắm vào ngay khi wrapper sẵn sàng;
- trước đó, mỗi thành viên kiểm thử hàm `select` của mình trên tensor giả
  (unit test, không cần model).

Ràng buộc mà wrapper phải `assert`:
- `keep_indices` tăng dần;
- `v_kept.shape[0] == deepstack_kept[l].shape[0] == K` với mọi l;
- dtype khớp với `v`.

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

**Lưu ý:** ViT của Qwen3-VL dùng attention **đầy đủ trong mỗi ảnh**
(`cu_seqlens` theo ảnh, không chia cửa sổ như Qwen2.5-VL). Vì vậy
`attention_ViT` tăng theo **bình phương** số patch.

### 5.1 Chi phí prefill cụ thể cho ảnh điển hình (GMAC, tính theo công thức trên)

| Ảnh | Lưới H′×W′ | N | N_text | ViT (không đổi) | Full: LLM / tổng | L5 50%: tổng (giảm) | L3 22,2%: tổng (giảm) | L1 5,6%: tổng (giảm) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| GQA trung vị | 20×13 | 260 | 26 | 393 | 408 / 801 | 615 (−23%) | 512 (−36%) | 451 (−44%) |
| MMB trung vị | 16×11 | 176 | 52 | 255 | 324 / 579 | 453 (−22%) | 385 (−34%) | 342 (−41%) |
| MME trung vị | 24×16 | 384 | 34 | 618 | 599 / 1.218 | 940 (−23%) | 788 (−35%) | 698 (−43%) |
| MME P95 | 94×126 | 11.844 | 34 | **125.820** (attention 110.321) | 24.830 / 150.650 | 136.248 (−10%) | 129.985 (−14%) | 126.823 (−16%) |

Ba kết luận từ bảng:
1. Với ảnh nhỏ và vừa, cắt về 22,2% giảm khoảng **71%** chi phí LLM nhưng chỉ
   khoảng **35%** tổng chi phí prefill, vì ViT chiếm khoảng một nửa.
2. Với ảnh rất lớn, **attention của ViT chiếm hơn 70%** tổng chi phí. Cắt token
   sau ViT (P2) gần như không giúp; chỉ giảm độ phân giải (P0) hoặc cắt trong
   ViT (P1) mới giảm được phần này. Đây là lý do P0 là đối chứng **bắt buộc**,
   và kết quả MME phải được **chia theo nhóm kích thước** (N ≤ 512, 512 < N ≤ 4.096, N > 4.096).
3. KV cache giảm đúng theo tỉ lệ `N_total′ / N_total`. Ví dụ GQA trung vị: từ
   32,0 MB xuống 9,4 MB ở L3; MME P95: từ 1,33 GB xuống 0,30 GB.

## 6. Các mức ngân sách (7 mức cố định)

### 6.1 Danh sách mức

| Mức | `r` (tỉ lệ giữ) | Tỉ lệ cắt | Tương ứng LLaVA (576 token) | Vì sao có mức này |
| --- | ---: | ---: | ---: | --- |
| **L7** | **100%** | 0% | 576 | kiểm chứng: phải trùng baseline từng ký tự |
| **L6** | **75%** | 25% | 432 | lấp khoảng 100 → 50%; tìm điểm chất lượng bắt đầu giảm; cho budget head một lựa chọn "an toàn" |
| **L5** | **50%** | 50% | 288 | mức nhẹ, các paper trên Qwen2.5-VL hay báo cáo |
| **L4** | **33,3%** | 66,7% | 192 | mức chuẩn của các paper (FastV, SparseVLM, VisionZip …) |
| **L3** | **22,2%** | 77,8% | 128 | mức chuẩn của các paper; **mức dùng để chỉnh tham số** |
| **L2** | **11,1%** | 88,9% | 64 | mức chuẩn của các paper |
| **L1** | **5,6%** | 94,4% | 32 | cắt rất mạnh |

Quy tắc tính cho từng ảnh: `K = clip(⌈r · N⌉, 1, N)`.
Ví dụ N = 260 và r = 22,2% thì K = ⌈57,7⌉ = 58.

### 6.2 Số token thật ở từng mức

Số liệu dưới đây tính từ `outputs/baseline/*.jsonl`: `N = input_tokens − N_text`,
trong đó `N_text` đếm bằng tokenizer của checkpoint trên đúng prompt đã chạy.

**Phân bố N_visual của từng benchmark:**

| Benchmark | Số mẫu | N_text TB | N TB | N nhỏ nhất | N P5 | **N trung vị** | N P95 | N lớn nhất |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GQA testdev | 12.578 | 25,7 | 268,4 | 96 | 165 | **260** | 340 | 400 |
| MMBench val | 4.329 | 73,1 | 155,3 | 64 | 65 | **176** | 256 | 256 |
| MME test | 2.374 | 34,3 | 1.756,2 | 65 | 90 | **384** | 11.844 | 16.320 |

Ba nhận xét ảnh hưởng tới thiết kế:
1. **MME lệch rất mạnh.** Trung vị chỉ là 384, nhưng có những ảnh tới 16.320
   token (gần trần `max_pixels`). Trung bình 1.756 là do phần đuôi kéo lên. Khi
   báo cáo MME phải **chia theo nhóm kích thước ảnh** (mục 5.1).
2. **MMBench có phần chữ dài** (trung bình 73 token: hint + 4 lựa chọn), chiếm
   khoảng 32% chuỗi. Cắt token ảnh ở đây tiết kiệm **ít hơn** so với GQA.
3. **N nhỏ nhất = 64–65** là do `min_pixels = 65.536` (= 64 × 1024).

**K trung bình (và trung vị) ở từng mức:**

| Mức | r | GQA: K TB (trung vị) | GQA: N_total′ TB | MMB: K TB (trung vị) | MMB: N_total′ TB | MME: K TB (trung vị) | MME: N_total′ TB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L7 | 100% | 268,4 (260) | 294,1 | 155,3 (176) | 228,3 | 1.756,2 (384) | 1.790,4 |
| L6 | 75% | 201,3 (195) | 227,0 | 116,5 (132) | 189,6 | 1.317,3 (288) | 1.351,5 |
| L5 | 50% | 134,2 (130) | 159,9 | 77,6 (88) | 150,7 | 878,2 (192) | 912,4 |
| L4 | 33,3% | 89,7 (87) | 115,4 | 52,0 (59) | 125,1 | 585,6 (128) | 619,9 |
| L3 | 22,2% | 60,0 (58) | 85,7 | 35,0 (40) | 108,0 | 390,6 (86) | 424,9 |
| L2 | 11,1% | 30,3 (29) | 56,0 | 17,7 (20) | 90,8 | 195,5 (43) | 229,8 |
| L1 | 5,6% | 15,4 (15) | 41,0 | 9,0 (10) | 82,0 | 98,0 (22) | 132,2 |

Ở L1, một ảnh MMBench trung bình chỉ còn khoảng 9 visual token. Mức này rất
mạnh, nên dự kiến chất lượng sẽ giảm rõ.

### 6.3 Quy tắc dùng mức

- **Chạy chính:** PP1 và PP3 chạy ở cả 7 mức. Trên dev subset 1.000 câu
  thì chạy cả 7 mức; trên full benchmark thì chạy L6, L5, L4, L3, L2.
- **Quét dày trên dev subset** (chỉ để tìm điểm gãy, không báo cáo làm kết quả
  chính): r ∈ {100, 90, 75, 60, 50, 40, 33,3, 25, 22,2, 15, 11,1, 5,6}%.
- **PP2 và PP4** có K thay đổi theo mẫu. Chúng được so ở cùng **tỉ lệ giữ
  trung bình thực tế** trên từng benchmark: `r̄ = (Σ K) / (Σ N)`.
- **Mức trần tuyệt đối** {64, 128, 256} token chỉ là phân tích phụ.

## 7. Giao thức dữ liệu (theo `configs/data_v1.yaml`)

| Vai trò | Split | Dùng cho |
| --- | --- | --- |
| Huấn luyện | GQA train balanced | tạo nhãn và huấn luyện mọi mạng nhỏ (PP2–4) |
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

## 10. Ai phụ trách phần dùng chung

| Thành phần | Mục | Người phụ trách (khuyến nghị) | Kiểm tra trước khi dùng |
| --- | --- | --- | --- |
| Wrapper suy luận (lắp token, M-RoPE, decode): `src/token_reduction_vl/models/qwen3_vl_pruned.py` | 4 | **một người cài cho cả nhóm** trong tuần đầu | `r = 100%` trùng baseline 100/100 mẫu mỗi benchmark (mục 4.2, bước 7) |
| Bộ đánh giá: `src/token_reduction_vl/evaluation/{agreement,bootstrap,timing,run_meta}.py` | 8 | cùng người hoặc một người khác | chạy lại ra đúng số liệu baseline trong `outputs/reports/baseline/…` |
| Danh sách dữ liệu chung: 50k GQA train, 5k GQA val, dev subset 1.000 câu | 7 | một người, làm một lần | lưu trong `configs/splits/` (quy ước tên: `METHODS.md` mục 9.3), ghi seed |
| Đối chứng: `src/token_reduction_vl/pruning/baselines/`, kết quả ở `outputs/controls/` | 9 | chia đều cho các thành viên | mỗi đối chứng chạy một lần, mọi người dùng chung kết quả |

Nếu nhóm chọn để **mỗi người tự cài wrapper**, mọi bản cài đều phải qua cùng
bài kiểm tra ở cột cuối trước khi được dùng để báo cáo.
