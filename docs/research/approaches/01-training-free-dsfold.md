# Cách 1 — DSFold không huấn luyện (toán thuần, r cố định)

- Trạng thái: Đề xuất
- Vai trò: **bắt buộc**, là nền móng cho cả 3 cách còn lại
- Bao nhiêu token: **r cố định** (hard-code) · Token nào: **công thức toán**
- Tài liệu liên quan: [`../dsfold-method-proposal.md`](../dsfold-method-proposal.md) (ánh xạ M1–M8 lên sơ đồ)

## 1. Ý tưởng

Không huấn luyện gì, không thay đổi trọng số model. Sau khi ViT chạy xong,
dùng các công thức toán trên những đặc trưng đã có sẵn để:

1. chọn K token phủ được nội dung ảnh và liên quan tới câu hỏi (**Select**);
2. gộp thông tin của các token bị bỏ vào kênh DeepStack của token được giữ (**Fold**).

## 2. Vị trí trên sơ đồ

Khối mới **"2½. DSFold"** nằm giữa Vision Encoder (khối 2) và Multimodal
Token Fusion (khối 3).

- Các ô baseline bị ảnh hưởng: M1–M8.
- Phần không đổi: processor, 24 ViT block, 28 decoder layer, LM head.

## 3. Đầu vào (đều có sẵn, không tốn thêm tính toán)

| Ký hiệu | Kích thước | Nguồn |
| --- | --- | --- |
| `v` | `[N, 2048]` | Main visual embeddings (Final PatchMerger) |
| `f₁, f₂, f₃` | `3 × [N, 2048]` | DeepStack merger sau ViT block 6 / 12 / 18 |
| `grid` | `[N, 3]` | toạ độ (t, h, w) gốc trên lưới đã merge |
| `q` | `[N_q, 2048]` | embedding các token câu hỏi (bảng embedding của LLM) |
| `r` | số | tỉ lệ giữ, một trong {100, 50, 33,3, 22,2, 11,1, 5,6}% |

`N = N_visual`, và `K = ⌈r · N⌉`.

## 4. Thuật toán từng bước

### S1. Chấm điểm

```
rel(i) = max_j cos(v_i, q_j)                      # liên quan tới câu hỏi
ms(i)  = ‖f₃ᵢ − f₂ᵢ‖ + ‖f₂ᵢ − f₁ᵢ‖                # multi-level saliency (mới)
rel, ms ← z-score rồi đưa về dương (ví dụ sigmoid hoặc min-max)
quality(i) = rel(i)^α · ms(i)^β
S(i, j)    = cos(v_i, v_j)                        # độ giống nhau giữa token
L          = diag(quality) · S · diag(quality)    # kernel DPP
```

Biến thể an toàn hơn cho `ms`, cần thử nếu công thức trên không có ích:
mỗi fₗ đi qua một MLP riêng, nên phép trừ trộn cả "khác do nội dung" lẫn "khác
do MLP". Hai phương án thay thế:
- `ms(i) = std_l(‖fₗᵢ‖)`: độ biến thiên độ lớn qua các tầng;
- `ms(i) = 1 − mean_l cos(fₗᵢ, mean_token(fₗ))`: token càng khác token trung
  bình của tầng thì càng có thông tin.

### S2. Chọn K token

Greedy MAP cho DPP (thuật toán Chen và cộng sự 2018, độ phức tạp `O(N·K²)`):
mỗi vòng chọn token làm tăng `log det(L_I)` nhiều nhất. Đối chứng đơn giản là
max–min diversity có trọng số (giống DivPrune).

Đầu ra: `I_keep` (K chỉ số) và `I_drop`.

### S3. Gán token bị bỏ cho token được giữ

```
a(d) = argmax_{k ∈ I_keep} [ cos(v_d, v_k) − λ · dist_grid(d, k) ]
```

`dist_grid` là khoảng cách Euclid trên lưới (h, w), chuẩn hoá theo kích thước
lưới. Thành phần này đảm bảo chi tiết được gửi tới token **gần về vị trí**.

### S4. Fold vào DeepStack

```
w_dk   = softmax_{d: a(d)=k}( cos(v_d, v_k) / τ )
f̃ₗ[k]  = fₗ[k] + γ · Σ_{d: a(d)=k} w_dk · fₗ[d]
f′ₗ[k] = f̃ₗ[k] · ‖fₗ[k]‖ / ‖f̃ₗ[k]‖               # norm-match
```

- `v[I_keep]` giữ nguyên, **không** trộn vào main stream (trừ ở ablation).
- Nếu token k không được gán token nào thì `f′ₗ[k] = fₗ[k]`.

### S5. Lắp vào model

1. `input_ids`: thay khối N token `<|image_pad|>` bằng K token.
2. `inputs_embeds`: tra embedding, rồi `masked_scatter(v[I_keep])`.
3. `position_ids`: tính M-RoPE trên chuỗi đầy đủ (N token ảnh), rồi giữ các
   cột text cùng các cột ảnh thuộc `I_keep`, đúng thứ tự.
4. `rope_deltas = max(position_ids) + 1 − N_total′`.
5. `visual_pos_masks` có K vị trí, và DeepStack′ là `[f′₁, f′₂, f′₃]`.
6. Prefill trên `language_model`, sau đó greedy decode tối đa 32 token có KV cache.

## 5. Siêu tham số

| Tham số | Ý nghĩa | Giá trị ban đầu | Chọn trên |
| --- | --- | --- | --- |
| α | trọng số mức liên quan câu hỏi | 1 | GQA val |
| β | trọng số multi-level saliency | 0,5 | GQA val |
| λ | phạt khoảng cách không gian khi gán | 0,5 | GQA val |
| γ | mức fold | 0,5 | GQA val |
| τ | nhiệt độ softmax khi fold | 0,1 | GQA val |

Chỉ chỉnh tham số trên một tập con nhỏ của GQA val (ví dụ 2.000 câu), sau đó
**khoá lại** trước khi đánh giá.

## 6. Thí nghiệm

### Đối chứng (cùng K)

| Đối chứng | Mô tả |
| --- | --- |
| Full token | baseline |
| P0 resolution-matched | giảm `max_pixels` để `N_visual ≈ K` |
| Random-K | 3 seed |
| Grid-K | lấy mẫu đều trên lưới |
| Pool 2×2 | average pooling trên lưới token |

### Phương pháp từ paper (cài lại trên Qwen3-VL-2B)

FastV (k = 2 và k = 3), DivPrune, CDPruner, DART, và VisionZip đã chỉnh để
không cần [CLS] (ghi rõ cách chỉnh).

### Ablation

| Ký hiệu | Thay đổi | Trả lời giả thuyết |
| --- | --- | --- |
| A1 | DeepStack: cắt theo `I_keep` / bỏ hẳn / giữ nguyên N (không hợp lệ, chỉ để minh hoạ) | H1 |
| A2 | M-RoPE: toạ độ gốc / đánh số lại thành lưới dày | H2 |
| A3 | γ = 0 (không fold) / chỉ fold DeepStack / chỉ merge v / cả hai | H3 |
| A4 | β = 0 / ba biến thể `ms` | H4 |
| A5 | α = 0 (không dùng câu hỏi) | |
| A6 | DPP / max–min diversity / top-K theo quality | |

## 7. Chỉ số đo

- **Chất lượng:** GQA accuracy, MMBench accuracy, MME score + Accuracy+.
  Kèm paired bootstrap 95% CI so với full token, lấy mẫu lại theo ảnh.
- **Hiệu năng:**
  - số token giữ thực tế;
  - FLOPs prefill (ViT và LLM tách riêng);
  - thời gian ViT / DSFold / prefill / decode (CUDA event);
  - peak VRAM.

## 8. Chi phí

- Chi phí của DSFold: ma trận cos `N × N` cộng greedy DPP. Với N = 300 thì
  không đáng kể. Với ảnh MME lớn (N ≈ 1.750), ma trận khoảng 3 triệu phần tử,
  vẫn nhỏ so với một decoder layer. Cần đo để báo cáo.
- Thời gian chạy (ước tính): khoảng 1,5–2 giờ cho mỗi phương pháp × mỗi mức
  trên cả 3 benchmark. Nên chạy dev subset 1.000 câu mỗi benchmark trước.

## 9. Ưu, nhược điểm

| Ưu | Nhược |
| --- | --- |
| Không huấn luyện, áp dụng ngay cho bất kỳ checkpoint Qwen3-VL nào | r cố định, chưa thích ứng theo mẫu |
| Dễ giải thích, dễ ablation từng thành phần | Công thức `ms` và Fold là giả thuyết, có thể không giúp |
| Rẻ | Các phương pháp trong nhóm này rất đông người làm |

## 10. Đóng góp khoa học

1. Nghiên cứu có hệ thống về cắt token khi có DeepStack (A1, A2, so với P0).
2. Tín hiệu `ms` lấy từ DeepStack có sẵn.
3. Cơ chế Fold vào DeepStack.

## 11. Mốc hoàn thành

| Mốc | Điều kiện |
| --- | --- |
| M1 | wrapper trùng khớp baseline trên 100 mẫu khi `r = 100%` |
| M2 | các đối chứng ở 6 mức, có biểu đồ Pareto |
| M3 | tái hiện FastV / DivPrune / CDPruner |
| M4 | kết quả A1, A2 |
| M5 | DSFold đầy đủ thắng phương pháp tốt nhất ở mức ≤ 22,2%, CI không chứa 0 |
