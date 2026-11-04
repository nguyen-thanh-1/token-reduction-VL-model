# Kiến trúc PP3 — chi tiết từng khối

Ngày: 2026-11-03. Cấu hình: `configs/methods/pp3_scorer_attn50k.yaml`. Số tham số
dưới đây đếm trực tiếp từ code (`build_score_head("3d", (256, 64, 256))`) và từ
config của `Qwen/Qwen3-VL-2B-Instruct`.

**Trả lời ngắn:** đúng — hệ thống gồm **Qwen3-VL-2B giữ nguyên (đóng băng, không
train)** và **một model nhỏ mình tự train: score head 3d, 1,99 triệu tham số**
(khoảng 0,09% kích thước Qwen). Phần còn lại của PP3 (điểm DSFold, PCA, DPP, fold)
là **phép tính cố định, không có tham số học**.

---

## 1. Toàn cảnh

```
                    ┌──────────────────────── Qwen3-VL-2B (ĐÓNG BĂNG) ───────────────────────┐
 ảnh ──► ViT 24 lớp ──► PatchMerger ──► v: N×2048 ─────────────┐                              │
            │ lớp 5, 11, 17                                    │                              │
            └──────────► DeepStack f¹,f²,f³: 3×N×2048 ───────┐ │                              │
 câu hỏi ──► bảng embedding của LLM ──► từ: N_q×2048 ──────┐ │ │                              │
                    └──────────────────────────────────────┼─┼─┼──────────────────────────────┘
                                                           ▼ ▼ ▼
               ┌──────────────────────── PP3 (giữa ViT và LLM) ──────────────────────────┐
               │ [cố định] S1: điểm DSFold z_rel, z_ms + PCA nén + vị trí  → 467 số/token │
               │ [HỌC]     S2: score head 3d (1,99 triệu tham số) → điểm s_i              │
               │ [cố định] S3: DPP tham lam → chọn K token                                │
               │ [cố định] S4: fold DeepStack của token bị bỏ                             │
               └──────────────────────────────────────────────────────────────────────────┘
                                                           ▼
                    ┌──────────── LLM Qwen3-VL 28 lớp (ĐÓNG BĂNG) ────────────┐
                    │ K token ảnh (vị trí M-RoPE gốc) + câu hỏi → câu trả lời  │
                    │ DeepStack (đã fold) cộng vào lớp 0, 1, 2                 │
                    └──────────────────────────────────────────────────────────┘
```

| Phần | Tham số | Train? |
|---|---|---|
| ViT (vision encoder + PatchMerger) | 407 triệu | không |
| LLM (28 lớp, gồm bảng embedding 311 triệu, dùng chung với lm_head) | 1,72 tỉ | không |
| **Score head 3d** | **1,99 triệu** | **có** |
| PCA nén đặc trưng | — (ma trận cố định, fit một lần) | không |
| Điểm DSFold, DPP, fold | 0 | không |

---

## 2. Qwen3-VL-2B-Instruct (phần đóng băng)

| Thành phần | Thông số |
|---|---|
| Patch | 16 × 16 pixel (2 khung hình theo thời gian cho video; ảnh tĩnh nhân đôi) |
| ViT | 24 lớp, hidden 1024, 16 head, FFN 4096 |
| PatchMerger | gộp 2 × 2 patch → 1 token, chiếu ra 2048 chiều (mỗi token ≈ 32 × 32 pixel) |
| DeepStack | đặc trưng từ ViT lớp 5, 11, 17, mỗi mức qua merger riêng → 3 × N × 2048; cộng vào hidden state của LLM tại vị trí token ảnh ở lớp 0, 1, 2 |
| LLM | 28 lớp, hidden 2048, 16 head query / 8 head key-value (GQA), FFN 6144, từ vựng 151.936 |
| Vị trí | M-RoPE 3 chiều (thời gian, hàng, cột) cho token ảnh |

PP3 lấy từ Qwen: `v` (N × 2048), DeepStack (3 × N × 2048), lưới H' × W', và
**bảng embedding đầu vào** của LLM để biến từ câu hỏi thành vector. Không sửa trọng
số nào của Qwen.

---

## 3. S1 — chuẩn bị đầu vào cho score head (cố định, không học)

Ví dụ: ảnh 10 × 16 = **N = 160 token**; câu hỏi "What device is not on?" sau khi
bỏ stopword còn khoảng **N_q = 3–4 token từ** ("device", "not", "on", ...).

| Bước | Phép tính | Kích thước ra |
|---|---|---|
| Chuẩn hóa `v` | chia độ dài: `v̂_i = v_i / ‖v_i‖` | 160 × 2048 |
| Chuẩn hóa DeepStack | mỗi mức trừ trung bình theo token, chia độ lệch chuẩn | 3 × 160 × 2048 |
| `z_rel` | `max_j cos(v̂_i, từ_j)` → z-score, cắt [−3, 3] | 160 |
| `z_ms` | `‖f̂³−f̂²‖ + ‖f̂²−f̂¹‖` → z-score, cắt [−3, 3] | 160 |
| PCA `v` | 2048 → 256 (ma trận cố định) | 160 × 256 |
| PCA DeepStack | mỗi mức 2048 → 64 | 3 × 160 × 64 |
| PCA từ câu hỏi | 2048 → 256 (sau khi chuẩn hóa độ dài) | N_q × 256 |
| Vị trí | sin/cos(2^k·π·hàng/H'), sin/cos(2^k·π·cột/W'), k = 0..3 | 160 × 16 |
| Budget | `log r` (L3: log(2/9) ≈ −1,50) | 1 số |

PCA được fit một lần trên 1000 ảnh train (`scripts/pp3/fit_compressor.py`, lưu ở
`data/features/pp3_scorer/compressor.npz`). Lý do: lưu đặc trưng gốc của 50k mẫu
để train cần khoảng 220 GB; sau PCA chỉ còn một phần nhỏ.

---

## 4. S2 — Score head 3d (`ScoreTextGuided`, phần duy nhất được train)

Kích thước nội bộ: token 256 chiều, câu hỏi 256 chiều, 4 head attention, dropout 0,1.

### 4.1 Từng khối, theo thứ tự chạy

| # | Khối | Phép tính | Ra | Tham số |
|---|---|---|---|---|
| 1 | Encoder token: chiếu `v` | Linear 256 → 256 | 160 × 256 | 65.792 |
| 2 | Encoder token: chiếu DeepStack | 3 × Linear 64 → 64 | 3 × 160 × 64 | 12.480 |
| 3 | Ghép vector token `u_i` | [v 256 ; 3 mức 192 ; z_rel, z_ms 2 ; vị trí 16 ; log r 1] | 160 × 467 | 0 |
| 4 | Encoder: chiếu từ câu hỏi | Linear 256 → 256 | N_q × 256 | 65.792 |
| 5 | Vào không gian token | Linear 467 → 256 → `h` | 160 × 256 | 119.808 |
| 6 | **Self-attention giữa token ảnh** | `h = h + MHA(LN(h), LN(h), LN(h))` | 160 × 256 | 263.680 |
| 7 | Vị trí từ | cộng Embedding(64 vị trí, 256) vào mỗi từ (tối đa 64 từ) | N_q × 256 | 16.384 |
| 8 | **Mã hóa câu hỏi** | 1 lớp Transformer encoder (pre-norm, 4 head, FFN 512) | N_q × 256 | 527.104 |
| 9 | **Cross-attention token → từ** | `h = h + MHA(LN(h), từ, từ)` | 160 × 256 | 263.680 |
| 10 | FFN | `h = h + Linear(1024→256)(GELU(Linear(256→1024)(LN(h))))` | 160 × 256 | 526.080 |
| 11 | LayerNorm cuối | `h = LN(h)` | 160 × 256 | 512 |
| 12 | Điểm chung | `a_i = Linear(256 → 1)(h_i)` | 160 | 257 |
| 13 | **Điểm liên quan từ–token** | `R_iw = ⟨W_k h_i, W_q từ_w⟩ / √256` | 160 × N_q | 131.584 |
| 14 | Gộp theo từ | `b_i = logsumexp_w R_iw` (gần max theo từ, nhưng mượt) | 160 | 0 |
| 15 | **Điểm cuối** | `s_i = a_i + b_i` | 160 | — |
|   | **Tổng** | | | **1.993.153** |

(Số tham số của khối 6, 9 gồm cả LayerNorm đi kèm; khối 10 gồm LayerNorm của FFN.)

Sau head: `q̂_i = softplus(s_i) + 0,001` — chất lượng luôn dương, đưa vào DPP.

Trường hợp câu hỏi rỗng (không còn từ sau khi bỏ stopword): bỏ khối 7–9 và 13–14,
điểm chỉ là `a_i`.

### 4.2 Ý nghĩa từng phần

- **Khối 1–5**: biến 467 số của mỗi token (nội dung, 3 mức chi tiết, hai điểm
  DSFold, vị trí, budget) thành một vector 256 chiều.
- **Khối 6 — self-attention**: mỗi token xem các token khác trong ảnh. Một token
  góc màn hình TV "biết" nó thuộc một vật lớn; một token trời "biết" xung quanh
  toàn trời.
- **Khối 7–8 — mã hóa câu hỏi**: embedding từ của LLM là tĩnh ("on" giống nhau dù
  đứng trong "turn on" hay "on the table"). Một lớp Transformer cho các từ nhìn
  nhau, có thứ tự, để "not on" mang nghĩa "đang tắt".
- **Khối 9 — cross-attention**: mỗi token ảnh "đọc" câu hỏi, lấy thông tin từ các
  từ liên quan tới nó.
- **Khối 13–14 — điểm liên quan**: một đường tắt rõ ràng: token được cộng điểm
  khi **có một từ cụ thể** trỏ mạnh tới nó (ví dụ "device" → TV, remote).
- **Khối 12 + 15**: điểm cuối = mức quan trọng chung + mức khớp câu hỏi.

### 4.3 So sánh với các head đã thử

| Head | Khác 3d ở đâu | Tham số |
|---|---|---|
| 3a `ScoreMLP` | không có attention; câu hỏi tóm thành 1 vector trung bình; MLP 979 → 512 → 256 → 1 trên `[u_i ; câu hỏi ; v_i ⊙ câu hỏi]` | 777.409 |
| 3b `ScoreCNN` | 3 lớp tích chập 3 × 3 trên lưới, câu hỏi điều chỉnh qua FiLM | — (ít dùng) |
| 3c `ScoreTransformer` | có khối 6 và 9, không có khối 7–8 và 13–14 | 1.318.081 |
| **3d `ScoreTextGuided`** | đầy đủ như bảng 4.1 | **1.993.153** |

---

## 5. S3 và S4 — chọn và fold (cố định, 0 tham số)

**S3 — DPP tham lam**: kernel `L = diag(q̂) · S · diag(q̂)`, `S_ij = cos(v̂_i, v̂_j)`
(160 × 160). Chọn lần lượt K token, mỗi bước lấy token làm tăng định thức nhiều
nhất (điểm cao và khác các token đã chọn). Độ phức tạp O(N·K) mỗi bước nhờ phân
rã Cholesky tăng dần. Trên 4096 token thì dùng max–min có trọng số.

**S4 — fold**: mỗi token bị bỏ gán cho token giữ có `cos − 0,5 × khoảng_cách/đường_chéo`
lớn nhất; DeepStack của token giữ cộng thêm `0,5 ×` trung bình có trọng số
(softmax của cos / 0,1) của các token nó nhận, rồi đưa về độ dài cũ. `v` không đổi.

Tham số cố định (bộ D-1.0, `configs/dsfold_D-1.0.yaml`): λ = 0,5, γ = 0,5, τ = 0,1.

---

## 6. Lúc train và lúc chạy

| | Train (một lần) | Chạy (mỗi câu hỏi) |
|---|---|---|
| Qwen | chạy full-token, bật eager attention để lấy nhãn (khoảng 1,1 s/mẫu) | ViT chạy bình thường; LLM chỉ nhận K token |
| Đặc trưng S1 | tính một lần, lưu vào đĩa (`attn_gqa_train_*.npz`) | tính trực tiếp |
| Score head | học: cross-entropy giữa `softmax(s)` và phân phối attention của teacher; AdamW lr 3e-4 (cosine), 20 epoch, batch 16 | chỉ forward |
| DPP, fold | không cần | chạy |
| Gradient vào Qwen | không | không |

Checkpoint: `outputs/pp3_scorer/checkpoints/score_head_3d_attn50k_seed0.pt` (khoảng
8 MB ở float32).

---

## 7. Biến thể PP3-fc: hai model nhỏ

PP3-fc dùng **hai** score head cùng kiến trúc 3d (tổng khoảng 4 triệu tham số):

| Head | Nhãn train | Dùng cho |
|---|---|---|
| `3d_attn50k` | attention của LLM (như PP3) | 75% budget "context" (DPP) |
| `3d_contrast` | attention trừ trung bình các câu hỏi khác trên cùng ảnh | 25% budget "focus" (mọc vùng liền mạch) |

Phần mọc vùng và chia budget là phép tính cố định (`focus_context.py`), không có
tham số học.

---

## 8. Tệp code

| Khối | Tệp |
|---|---|
| Wrapper Qwen + đưa K token vào LLM | `src/token_reduction_vl/models/qwen3_vl_pruned.py` |
| S1 điểm DSFold | `src/token_reduction_vl/pruning/pp3_scorer/dsfold_base/scoring.py` |
| S1 PCA, vị trí, ghép đặc trưng | `src/token_reduction_vl/pruning/pp3_scorer/features.py` |
| S2 score head 3a/3b/3c/3d | `src/token_reduction_vl/pruning/pp3_scorer/models.py` |
| S3 DPP | `src/token_reduction_vl/pruning/pp3_scorer/dsfold_base/dpp.py` |
| S4 fold | `src/token_reduction_vl/pruning/pp3_scorer/dsfold_base/fold.py` |
| Ghép S1–S4 | `src/token_reduction_vl/pruning/pp3_scorer/pruner.py` |
| Teacher, train | `pp3_scorer/attention_teacher.py`, `pp3_scorer/train.py`, `pp3_scorer/losses.py` |
| PP3-fc | `pp3_scorer/focus_context.py` |
