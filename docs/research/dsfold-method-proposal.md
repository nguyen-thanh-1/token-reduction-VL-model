# Đề xuất phương pháp: DSFold — DeepStack-aware Select-and-Fold

- Trạng thái: **Đề xuất** (chưa triển khai, chưa chốt; sẽ chuyển thành ADR 0002 sau khi có kết quả thử nghiệm sơ bộ)
- Ngày: 2026-10-21
- Sơ đồ đi kèm: [`architecture/dsfold-proposal/Qwen3-VL-2B-Instruct-DSFold-proposal.drawio`](../../architecture/dsfold-proposal/Qwen3-VL-2B-Instruct-DSFold-proposal.drawio)
- Khảo sát tài liệu chi tiết (tiếng Anh): [`visual-token-reduction-proposal.md`](visual-token-reduction-proposal.md)
- Bốn cách thực hiện (không học → học toàn bộ): [`approaches/README.md`](approaches/README.md)

## 1. Tóm tắt

DSFold là phương pháp **không cần huấn luyện**. Nó giảm số visual token mà
Qwen3-VL-2B-Instruct đưa vào language model và được đặt **sau Final 2×2
PatchMerger, trước bước fusion `masked_scatter`** (điểm P2 trong sơ đồ).
Phương pháp gồm hai phần:

1. **Select:** giữ K token dựa trên ba tín hiệu: độ đa dạng (coverage), mức
   liên quan tới câu hỏi, và một tín hiệu mới lấy từ **DeepStack** (đặc trưng
   của token thay đổi bao nhiêu qua ba tầng ViT 6/12/18).
2. **Fold:** token bị bỏ không bị vứt đi. Đặc trưng DeepStack của nó được
   **gộp vào token được giữ gần nhất**. DeepStack vốn là kênh cộng thêm
   (residual) vào visual token ở decoder layer 0–2, nên ta dùng nó như một
   "kênh phụ miễn phí" để giữ lại chi tiết mà không làm chuỗi dài thêm.

Kèm theo đó là hai ràng buộc kỹ thuật riêng của Qwen3-VL: **giữ toạ độ M-RoPE
gốc** của token được giữ, và **cắt DeepStack đúng theo chỉ số được giữ**.

## 2. Mục tiêu

- Giảm số visual token `N_visual → K`, tức giảm chi phí LLM prefill và KV cache.
- Giữ chất lượng bằng hoặc tốt hơn baseline full-token trên GQA testdev,
  MMBench validation và MME test.
- Báo cáo chất lượng và hiệu năng **tách riêng**: accuracy so với số token giữ
  lại, FLOPs, latency từng giai đoạn, và VRAM.

## 3. Vì sao can thiệp ở P2 (lý do)

### 3.1 Bốn vị trí có thể giảm token

Trên sơ đồ, mỗi vị trí được đánh dấu bằng một nhãn tròn màu đỏ.

| Nhãn | Vị trí trên sơ đồ | Giảm được | Vai trò trong đề tài |
| --- | --- | --- | --- |
| **P0** | Khối 1, ô "Resize động" (`max_pixels`) | ViT + LLM | **Baseline đối chứng** (resolution-matched), không phải phương pháp |
| **P1** | Khối 2, giữa các ViT block | ViT + LLM | Mở rộng sau này; khó vì có 2D RoPE và gom nhóm 2×2 |
| **P2** | Giữa khối 2 và 3: khối mới "2½. DSFold" | LLM prefill + KV cache | **Vị trí của phương pháp đề xuất** |
| **P3** | Khối 4, sau decoder layer k ≥ 3 (kiểu FastV) | các layer ≥ k | **Baseline so sánh** |

### 3.2 Lý do chọn P2

1. **Tiết kiệm nhiều nhất phía LLM.** Cắt trước khi vào LLM thì cả 28 layer
   đều xử lý ít token hơn, và KV cache cũng nhỏ hơn.
2. **Đây là chỗ Qwen3-VL khác LLaVA, nên có chỗ cho điểm mới:**
   - Không có token [CLS], nên các phương pháp kiểu VisionZip/PruMerge không
     dùng nguyên được.
   - Có **DeepStack**: mỗi visual token còn nhận 3 đặc trưng cộng thêm ở
     decoder layer 0/1/2. Khi cắt token phải quyết định xử lý DeepStack của
     token bị bỏ ra sao. Trong 9 paper về Qwen3-VL năm 2026 mà tôi đọc được
     toàn văn, **không paper nào nhắc tới DeepStack**.
   - Có **M-RoPE 3 chiều** (t, h, w): cắt token mà đánh số lại vị trí là làm
     sai bố cục không gian.
3. **Tín hiệu DeepStack có sẵn, gần như miễn phí.** Ba tầng đặc trưng đã được
   tính và chiếu lên 2048 chiều, chỉ việc dùng lại để chấm điểm.
4. **Có thể làm hoàn toàn không cần huấn luyện,** hợp với GPU hiện có và thời
   gian làm khoá luận.

### 3.3 Hai giới hạn cần nói rõ từ đầu

- **ViT vẫn chạy đủ.** Với bản 2B, ViT tốn khoảng **46%** lượng tính toán của
  mỗi visual token: khoảng 1,21 GMACs ở ViT so với 1,41 GMACs ở LLM prefill
  (chỉ tính các lớp linear). Vì vậy P2 giảm tối đa khoảng 54% phần tính toán
  liên quan tới visual token. Phải báo cáo thời gian ViT / prefill / decode
  riêng, không được gọi "giảm x% token" là "nhanh hơn x%".
- **Số token phụ thuộc vào độ phân giải ảnh.** MME trung bình 1.790 token đầu
  vào, GQA chỉ 294. Vì vậy **mọi kết quả phải so với P0 ở cùng số token K**.
  Nếu chỉ giảm `max_pixels` mà đã tốt hơn thì phương pháp không có giá trị.

## 4. Phương pháp DSFold

**Đầu vào** (đều có sẵn sau ViT, chưa chạy LLM):

- `v ∈ R^{N×2048}`: main visual embeddings (đầu ra của Final PatchMerger)
- `f₁, f₂, f₃ ∈ R^{N×2048}`: DeepStack features (tap sau ViT block 6/12/18)
- `grid (t, h, w)`: toạ độ gốc của từng token trên lưới đã merge
- `q ∈ R^{N_q×2048}`: embedding của các token câu hỏi, lấy từ bảng embedding
- tỉ lệ giữ lại `r`, nên `K = ⌈r·N⌉`

### S1. Chấm điểm (score)

- **Coverage/diversity:** dùng kernel DPP hoặc max–min diversity trên `v`.
  Đây là phần "lõi bền" theo paper 2502.11501: chống trùng lặp quan trọng hơn
  chọn token "quan trọng".
- **Liên quan tới câu hỏi:** `rel(i) = max_j cos(v_i, q_j)`.
- **Multi-level saliency (mới):** `ms(i) = ‖f₃ᵢ − f₂ᵢ‖ + ‖f₂ᵢ − f₁ᵢ‖`. Giả
  thuyết: vùng có vật thể hoặc chữ tiếp tục biến đổi qua các tầng, còn nền
  phẳng thì ổn định sớm.
- Kết hợp lại: `quality(i) = rel(i)^α · ms(i)^β` (đã chuẩn hoá z-score),
  `similarity = cos(v_i, v_j)`.

### S2. Chọn (select)

Greedy MAP cho conditional DPP với `L = diag(quality)·S·diag(quality)`, hoặc
max–min diversity có trọng số. Kết quả là `I_keep` (K token) và `I_drop`.

### S3. Gán (assign)

Mỗi token bị bỏ `d` được gán vào token giữ lại gần nhất:
`a(d) = argmax_k [cos(v_d, v_k) − λ·dist_grid(d, k)]`.

### S4. Gộp vào DeepStack (fold — mới)

```
f′ₗ[k] = norm-match( fₗ[k] + γ · Σ_{d: a(d)=k} w_dk · fₗ[d] ),   l = 1, 2, 3
w_dk   = softmax_d( cos(v_d, v_k) / τ )
```

`norm-match` đưa độ lớn của `f′ₗ[k]` về gần `‖fₗ[k]‖`, để DeepStack không bị
lệch thang đo so với lúc model được huấn luyện. Ablation sẽ thử: chỉ fold
DeepStack, merge cả main stream (kiểu ToMe/VisionZip), cả hai, hoặc không gộp.

### Đầu ra đưa vào phần còn lại của model

- `v[I_keep]` có kích thước `[K, 2048]`, dùng thay cho Main visual embeddings
  khi `masked_scatter`.
- `DeepStack′` gồm 3 tensor `[K, 2048]` đã fold, cộng vào ở decoder layer 0/1/2.
- `input_ids` có K token `<|image_pad|>`. `position_ids` được tính trên **lưới
  gốc** rồi lấy (gather) K cột tương ứng. `rope_deltas` được tính lại cho bước
  decode.

**Ngân sách K:** thí nghiệm chính dùng tỉ lệ cố định
`r ∈ {100%, 75%, 50%, 33,3%, 22,2%, 11,1%, 5,6%}` (7 mức). Các mức 33,3/22,2/11,1/5,6%
tương ứng với 192/128/64/32 token của LLaVA (576 token), nên so được trực tiếp
với các paper. Mức trần tuyệt đối `{64, 128, 256}` chỉ dùng làm phân tích phụ.
K thích ứng theo từng mẫu được mô tả ở cách 2 và cách 4 trong
[`approaches/`](approaches/README.md).

## 5. Can thiệp ở đâu trong sơ đồ

Sơ đồ đề xuất là bản sao sơ đồ baseline của bạn. Quy ước:

- **Khối/viền đỏ** là phần mới hoặc bị thay đổi; phần còn lại giữ nguyên.
- Khối 3–5 và panel B được dời sang phải để chừa chỗ cho khối mới
  **"2½. PROPOSED — DSFold"**.
- Panel mới **"D. DSFold — thuật toán chi tiết"** chứa mã giả.
- Ô **⑦** trong panel C là bảng kê số token (token ledger) sau khi cắt.
- Bảng **"Intervention map"** ở góc trên trái giải thích P0–P3.
- Bảng **"Ô baseline bị thay đổi"** ở góc trên phải liệt kê M1–M8.

| Nhãn | Ô trong sơ đồ | Thay đổi | Vị trí trong code (Transformers 5.x, `modeling_qwen3_vl.py`) |
| --- | --- | --- | --- |
| M1 | Khối 2 – Main visual embeddings | Không nối thẳng vào fusion nữa mà đi qua DSFold | `get_image_features` L1037 → `pooler_output` |
| M2 | Khối 2 – DeepStack feature #1–#3 | Đi qua DSFold để chấm điểm và fold, không nối thẳng vào decoder | `deepstack_features`, L1050–1051 |
| M3 | Khối 3 – Tạo image_mask | Mask tính trên `input_ids` đã rút gọn, chỉ còn K vị trí | `get_placeholder_mask` L1055 |
| M4 | Khối 3 – Kiểm tra số lượng vị trí ảnh | Số vị trí phải bằng K | trong `get_placeholder_mask` |
| M5 | Khối 3 – Thay embedding (masked_scatter) | Scatter `v[I_keep]` có kích thước `[K, 2048]` | L1189–1194 |
| M6 | Khối 3 – Tính position_ids T/H/W | Tính trên lưới gốc rồi gather K cột; tính lại `rope_deltas` | `get_rope_index` L929, `compute_3d_position_ids` L1096 |
| M7 | Khối 3 – Dữ liệu gửi vào Language Decoder | `N_total′ = N_text_and_special + K`; `visual_pos_masks` có K vị trí; truyền `DeepStack′` | L1205–1251 |
| M8 | Khối 4 – DeepStack #1–#3 ở layer 0/1/2 | Nhận `DeepStack′` `[K, 2048]` đã fold | `_deepstack_process` L851–858 |

Những phần **không** thay đổi: toàn bộ khối 1 (processor, chat template), 24
ViT block và các merger, 28 decoder layer, LM head, greedy decoding và cách
chấm điểm benchmark.

## 6. Cách làm (kế hoạch, chưa viết code)

1. **Wrapper suy luận có cắt token** (`models/qwen3_vl_pruned.py`). Gọi
   `get_image_features`, chạy pruner, dựng lại `input_ids`, `inputs_embeds`,
   `position_ids` và DeepStack′, rồi chạy prefill trên `language_model`. Phần
   decode dùng vòng lặp greedy tự viết (tối đa 32 token, có KV cache), không
   dùng `generate()`, để kiểm soát hoàn toàn vị trí M-RoPE.
2. **Kiểm chứng bắt buộc:** với `r = 100%` và không fold, kết quả phải **trùng
   khớp** với baseline. Kiểm trên 100 mẫu trước khi làm bất kỳ thí nghiệm nào.
3. **Pruner** (`pruning/`) theo interface `VisualTokenPruner` có sẵn:
   - đối chứng: `RandomK`, `GridK`, `Pool2x2`;
   - baseline từ paper: `DivPrune`, `CDPruner`, FastV ở P3;
   - phương pháp: `DSFold`.
   Mở rộng `PruningOutput` để chứa DeepStack′ và toạ độ lưới của token giữ lại.
4. **Đo đạc:** dùng CUDA event để đo thời gian ViT / pruner / prefill / decode,
   kèm bộ đếm FLOPs tính theo công thức.
5. **Config và báo cáo:** mỗi phương pháp/ngân sách có `configs/prune_*.yaml`
   và `report.id` riêng. **Không ghi đè** báo cáo baseline.

## 7. Giả thuyết và thí nghiệm

| Giả thuyết | Nội dung | Cách kiểm |
| --- | --- | --- |
| H1 | Cắt DeepStack đúng theo `I_keep` tốt hơn bỏ hẳn DeepStack | ablation slice / drop |
| H2 | Giữ toạ độ M-RoPE gốc tốt hơn đánh số lại thành lưới dày | ablation, xem kỹ câu hỏi không gian của GQA |
| H3 | Fold vào DeepStack cải thiện chất lượng ở mức giữ ≤ 22,2% | so có fold / không fold |
| H4 | Tín hiệu multi-level saliency có ích thêm ngoài diversity | bỏ `ms` |
| H5 | Ở cùng K, cắt token tốt hơn giảm độ phân giải | so với P0. Nếu thua, đó vẫn là phát hiện đáng báo cáo cho model nhỏ |

**Đối chứng bắt buộc:** full token; resolution-matched (P0); random-K (3 seed);
grid-K; pooling 2×2.

**So sánh với paper:** FastV (k = 2 và k = 3), DivPrune, CDPruner, DART, và
VisionZip đã chỉnh để không cần [CLS] (ghi rõ cách chỉnh).

**Thống kê:** paired bootstrap 95% CI so với full token, lấy mẫu lại theo từng ảnh.

**Muốn ra paper** thì nên thêm một benchmark nhiều chữ (TextVQA hoặc OCRBench)
và POPE. Việc này cần adapter mới và ADR cập nhật protocol dữ liệu.

## 8. Điểm mới so với các nghiên cứu đã có

| Đã có | DSFold khác ở đâu |
| --- | --- |
| DivPrune / CDPruner / DART chọn token theo diversity hoặc duplication | Dùng các thành phần này làm lõi, nhưng thêm tín hiệu DeepStack đặc thù Qwen3-VL |
| VisionZip / ToMe gộp token vào main stream | Gộp vào **kênh DeepStack**, nên chuỗi không dài thêm và main stream không bị trộn |
| SPIDER / STD dùng đặc trưng ViT tầng giữa để chọn token | Dùng các tầng **đã có sẵn** trong DeepStack nên không tốn thêm tính toán |
| OccamToken, RoRA, ET-Prune, RUTA (2026) thử trên Qwen3-VL | Không paper nào trong số này nói tới cách xử lý DeepStack hay M-RoPE khi cắt token |
| TCRR coi độ phân giải là phương pháp | Ta dùng độ phân giải làm **đối chứng bắt buộc** ở cùng K |

**Đóng góp dự kiến:**

1. Nghiên cứu có hệ thống đầu tiên về việc cắt visual token khi model có
   DeepStack: cách cắt, cách giữ vị trí, và đối chứng độ phân giải.
2. Cơ chế Fold.

Đóng góp (1) vẫn có giá trị kể cả khi (2) chỉ cải thiện ít.

## 9. Rủi ro và phương án dự phòng

- **Hướng nghiên cứu rất đông người làm:** không trình bày như "thêm một cách
  chấm điểm", mà nhấn vào câu hỏi DeepStack và đối chứng chặt chẽ. Chạy lại
  tìm kiếm arXiv với từ khoá "DeepStack" trước mỗi mốc, vì khoảng trống này có
  thể bị người khác lấp.
- **Tăng tốc ít trên bản 2B** vì ViT chiếm khoảng 46%: báo cáo trung thực theo
  từng giai đoạn; có thể kết hợp giảm nhẹ độ phân giải (P0) để có lợi end-to-end.
- **Fold làm hỏng phân phối DeepStack:** đã có `norm-match` và hệ số γ; nếu
  vẫn hỏng thì chỉ giữ phần Select (H1, H2, H4 vẫn đủ làm khoá luận).
- **Dự phòng nếu Fold thất bại:** dùng cascade không cần huấn luyện. Trả lời ở
  độ phân giải thấp trước; chỉ khi xác suất token trả lời thấp mới chạy lại ở
  độ phân giải cao kèm cắt token.

## 10. Lộ trình

| Mốc | Sản phẩm | Điều kiện hoàn thành |
| --- | --- | --- |
| M1 | Wrapper + kiểm chứng NoOp + đo thời gian từng giai đoạn | trùng khớp baseline trên 100 mẫu |
| M2 | Các đối chứng P0 / random / grid / pool ở 7 mức ngân sách | biểu đồ Pareto đầu tiên |
| M3 | FastV, DivPrune, CDPruner trên Qwen3-VL-2B | tái hiện được xu hướng của paper |
| M4 | Nghiên cứu DeepStack + vị trí (H1, H2) | đóng góp (1) |
| M5 | DSFold đầy đủ (H3, H4) | thắng baseline tốt nhất ở mức giữ ≤ 22,2%, CI không chứa 0 |
| M6 | Benchmark nhiều chữ + POPE, ngân sách thích ứng | chứng minh tính tổng quát |
| M7 | Viết khoá luận / bài báo | |
