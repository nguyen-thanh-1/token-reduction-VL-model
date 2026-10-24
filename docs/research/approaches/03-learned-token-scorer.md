# Cách 3 — Score head: mạng nhỏ chấm điểm từng token, r cố định

- Trạng thái: Đề xuất (mở rộng, làm khi cách 1 và 2 đã xong)
- Bao nhiêu token: **r cố định** · Token nào: **mạng nhỏ có huấn luyện** (kết hợp DPP)
- Điều kiện: wrapper của cách 1; GPU đủ VRAM để backprop qua model 2B (cần đo)

## 1. Ý tưởng

Ở cách 1, điểm `quality(i) = rel(i)^α · ms(i)^β` là **công thức do người
đặt**. Cách 3 thay công thức đó bằng một **mạng nhỏ học được**, nhận đặc trưng
của từng token rồi dự đoán token đó quan trọng tới đâu cho câu hỏi hiện tại.

Phần **chống trùng lặp vẫn giữ bằng DPP**: kernel `L = diag(q̂) · S · diag(q̂)`
dùng điểm học được `q̂` thay cho `quality`. Như vậy mạng chỉ phải học "token
nào quan trọng", còn "đừng chọn token trùng nhau" do toán đảm nhiệm. Mạng sẽ
dễ học hơn và ít overfit hơn.

## 2. Vị trí trên sơ đồ

Score head thay ô **S1 (Score)** trong khối "2½. DSFold". S2–S4 (Select,
Assign, Fold) giữ nguyên.

## 3. Đầu vào cho mỗi token i

| Đặc trưng | Kích thước | Ghi chú |
| --- | --- | --- |
| `v_i` | 2048 | chiếu xuống 256 |
| `f₁ᵢ, f₂ᵢ, f₃ᵢ` | 3 × 2048 | dùng chung một lớp chiếu xuống 128 cho mỗi tầng |
| `rel(i)`, `ms(i)` | 2 | đặc trưng thủ công của cách 1, giúp mạng khởi đầu tốt |
| toạ độ (h, w) chuẩn hoá | 2 | qua Fourier features 16 chiều |
| `q` (cả câu hỏi) | `[N_q, 2048]` | cho cross-attention |

## 4. Ba lựa chọn kiến trúc (thử theo thứ tự từ đơn giản đến phức tạp)

| | Kiến trúc | Tham số | Nắm bối cảnh |
| --- | --- | --- | --- |
| 3a | **MLP cho từng token**, gồm `[v, f, rel, ms, pos]` và `mean(q)` | ~1,5M | không có bối cảnh không gian |
| 3b | **CNN nhỏ trên lưới token** (H′ × W′ × 256), 3 lớp conv 3×3 | ~2M | có bối cảnh lân cận, phù hợp lưới 2D |
| 3c | **1 lớp transformer nhẹ**: self-attention giữa các token + cross-attention tới q | ~4M | bối cảnh toàn ảnh và câu hỏi |

Đầu ra: điểm `s_i ∈ R`, và `q̂_i = softplus(s_i)` dùng trong kernel DPP.

Cách 3b là chỗ **CNN có lý do để dùng**: visual token của Qwen3-VL xếp được
thành lưới 2D (ví dụ 15 × 20), nên conv 3×3 nắm được việc token có nằm trên
cạnh hay ranh giới vật thể không.

## 5. Hai cách huấn luyện

### Cách huấn luyện A — chưng cất qua model đóng băng (end-to-end)

1. **Teacher:** chạy full token trên mẫu GQA train, lưu phân phối xác suất của
   các token trả lời (tối đa vài token đầu). Chỉ chạy một lần, rồi lưu lại.
2. **Student:** model 2B đóng băng, với tập token được chọn bởi score head.
3. **Vấn đề:** chọn top-K không lấy đạo hàm được. Giải pháp:
   - **Lúc huấn luyện, dùng mặt nạ mềm:** không xoá token mà cộng
     `log(g_i + ε)` vào attention logits của các key ở vị trí ảnh, với
     `g = sigmoid(s / t)`. Kỹ thuật này giống DynamicViT, và dùng được nhờ
     custom 4D attention mask của Transformers.
   - **Straight-through:** forward dùng mặt nạ cứng top-K (để khớp với lúc
     suy luận), backward lan gradient qua `g` mềm.
   - **Ràng buộc ngân sách:** `mean(g) ≈ r`, thêm penalty `(mean(g) − r)²`.
4. **Loss:**
   ```
   L = KL( p_teacher(ans) ‖ p_student(ans) ) + μ · CE(ans_gt) + ρ · (mean(g) − r)²
   ```
5. **Chi phí (ước tính, cần đo):**
   - Backprop qua 28 layer với bf16 và gradient checkpointing, chuỗi khoảng
     300 token, batch 8: khoảng **10–16 GB VRAM**.
   - 50k mẫu × 2 epoch: khoảng **14 giờ**, cộng khoảng 2 giờ tạo teacher logits.
   - Huấn luyện riêng cho từng r, hoặc một mạng cho mọi r bằng cách đưa r vào
     làm đặc trưng đầu vào.

### Cách huấn luyện B — học theo nhãn saliency (rẻ hơn, không cần mặt nạ mềm)

1. Với mỗi mẫu GQA train, chạy full token **một lần forward và một lần
   backward** để lấy saliency của từng visual token:
   `sal_i = |∂ log p(ans) / ∂ v_i · v_i|` (gradient × input).
2. Huấn luyện score head để **xếp hạng** token giống saliency, dùng ranking
   loss (ListNet hoặc pairwise hinge).
3. Lúc suy luận: điểm học được, kết hợp DPP, rồi top-K.
4. **Chi phí:** 50k forward + backward, khoảng **8–10 giờ**. Score head huấn
   luyện vài phút.
5. **Nhược điểm:** saliency đo mức quan trọng **khi có đủ token**, không đo
   **sau khi đã cắt** (hiệu ứng tương tác giữa token). Kết quả có thể kém cách A.

**Khuyến nghị:** làm cách B trước (rẻ, đơn giản). Chỉ làm cách A nếu B cho tín
hiệu tốt và còn thời gian.

## 6. Thí nghiệm

| Câu hỏi | So sánh |
| --- | --- |
| Điểm học được có hơn công thức thủ công không? | cách 3 so với cách 1 ở cùng r, cùng DPP, cùng Fold |
| Có cần DPP không khi đã học điểm? | `q̂` + DPP so với top-K theo `q̂` |
| Kiến trúc nào đủ? | 3a / 3b / 3c |
| Cách huấn luyện nào? | A (chưng cất) / B (saliency) |
| Tổng quát hoá | học trên GQA, đánh giá trên MMBench và MME |
| Chi phí thêm lúc suy luận | thời gian chạy score head so với một decoder layer |

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
| --- | --- |
| Overfit GQA, kém ở MMBench/MME | giữ DPP; đầu vào có cả đặc trưng thủ công; kiến trúc nhỏ |
| Hết VRAM ở cách huấn luyện A | gradient checkpointing, batch 1 kèm tích luỹ gradient, hoặc chuyển sang B |
| Lợi ích nhỏ so với cách 1 | vẫn là kết quả có giá trị ("công thức thủ công đã đủ tốt") |
| Mất ưu điểm "không huấn luyện" | trình bày như một mở rộng; cách 1 vẫn là phương pháp chính |

## 8. Liên hệ paper

- VisPruner, TReVS: điểm thủ công, không học. Đây là các đối chứng.
- RUTA (2608.04132), ETC (2606.00543): có học, nhưng huấn luyện theo hướng khác.
  Cách 3 khác ở chỗ chỉ học `quality` và vẫn giữ DPP + Fold DeepStack.
- DynamicViT: nguồn gốc kỹ thuật mặt nạ mềm cùng straight-through.

## 9. Mốc hoàn thành

| Mốc | Điều kiện |
| --- | --- |
| M7a | đã tạo nhãn saliency cho 50k mẫu; score head 3a học xong |
| M7b | cách 3 thắng cách 1 ở ít nhất 2 mức r trên GQA val |
| M7c | (tuỳ chọn) cách huấn luyện A chạy được trong VRAM hiện có |
