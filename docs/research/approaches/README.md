# Bốn cách thực hiện giảm visual token — tổng quan

- Trạng thái: **Đề xuất** (chưa triển khai; phương pháp cuối cùng sẽ được ghi vào ADR 0002)
- Ngày: 2026-10-24
- Model: `Qwen/Qwen3-VL-2B-Instruct` (full-token baseline đã đo)
- Vị trí can thiệp chung: **P2**, sau Final 2×2 PatchMerger, trước `masked_scatter`.
  Xem sơ đồ [`architecture/dsfold-proposal/`](../../../architecture/dsfold-proposal/)

## 1. Hai câu hỏi độc lập

Mọi phương pháp cắt token đều phải trả lời hai câu hỏi:

1. **Giữ bao nhiêu token?** Đó là ngân sách K, hay tỉ lệ giữ `r = K / N_visual`.
2. **Giữ những token nào?** Đó là tập chỉ số `I_keep`.

Mỗi câu có thể trả lời bằng **công thức toán (không huấn luyện)** hoặc bằng
**mạng neural nhỏ (có huấn luyện)**. Ghép lại ta có 4 cách:

| | **Token nào: công thức toán** | **Token nào: mạng học** |
| --- | --- | --- |
| **Bao nhiêu: r cố định** | [Cách 1 — DSFold không huấn luyện](01-training-free-dsfold.md) | [Cách 3 — Score head](03-learned-token-scorer.md) |
| **Bao nhiêu: mạng học** | [Cách 2 — DSFold-A (budget head)](02-learned-budget-dsfold-a.md) | [Cách 4 — Học toàn bộ](04-fully-learned.md) |

## 2. So sánh nhanh

| | Cách 1 | Cách 2 | Cách 3 | Cách 4 |
| --- | --- | --- | --- | --- |
| Huấn luyện | Không | MLP nhỏ (~1M tham số) | Scorer nhỏ (~2–5M) | Scorer + gate (~3–6M) |
| Gradient qua model 2B | Không | Không | Có (hoặc nhãn oracle) | Có |
| Chi phí GPU thêm (ước tính) | 0 | ~12 giờ tạo nhãn | ~1–3 ngày | ~2–5 ngày |
| Thích ứng theo mẫu | Không (r cố định) | **Có** (K thay đổi) | Không (K cố định) | **Có** |
| Rủi ro | Thấp | Thấp–trung bình | Trung bình–cao | Cao |
| Đối thủ trực tiếp | DivPrune, CDPruner, VisionZip | ET-Prune, OccamToken | VisPruner, các scorer có học | RUTA, ETC |
| Vai trò trong khoá luận | **Bắt buộc** (nền móng) | **Phần "adaptive" chính** | Mở rộng nếu còn thời gian | Hướng phát triển |

Các con số chi phí là ước tính dựa trên thời gian baseline (GQA khoảng 0,15
s/mẫu). Cần đo lại trên GPU thực tế.

## 3. Thành phần dùng chung cho cả 4 cách

1. **Cắt DeepStack đúng chỉ số.** f₁, f₂, f₃ phải được lấy theo đúng
   `I_keep`, cùng thứ tự với v.
2. **Giữ toạ độ M-RoPE gốc.** Tính `position_ids` trên lưới đầy đủ rồi gather
   các cột được giữ, và tính lại `rope_deltas` cho bước decode.
3. **Dựng lại `input_ids`** với K token `<|image_pad|>` (ID 151655).
4. **Wrapper suy luận riêng** với vòng lặp greedy decode tự viết. Phép kiểm
   chứng bắt buộc: `r = 100%` phải trùng khớp baseline.
5. **Fold vào DeepStack** (tuỳ chọn ở cách 3 và 4; là thành phần chính ở cách 1 và 2).

## 4. Các mức ngân sách cố định (hard-code)

| r | 100% | 50% | 33,3% | 22,2% | 11,1% | 5,6% |
| --- | --- | --- | --- | --- | --- | --- |
| Tương ứng LLaVA (576 token) | 576 | 288 | 192 | 128 | 64 | 32 |
| Mục đích | kiểm chứng | mức nhẹ | chuẩn paper | chuẩn paper | chuẩn paper | rất mạnh |

`K = ⌈r · N_visual⌉` tính riêng cho từng ảnh. Mức trần tuyệt đối {64, 128,
256} chỉ dùng làm phân tích phụ. Các cách có K thích ứng (2 và 4) được so ở
cùng **tỉ lệ giữ trung bình thực tế**.

## 5. Quy tắc dữ liệu (theo `configs/data_v1.yaml`)

- Huấn luyện bất kỳ mạng nào: chỉ dùng **GQA train** (balanced).
- Chọn siêu tham số: **GQA val**.
- Đánh giá: **GQA testdev, MMBench validation, MME test**. MMBench và MME là
  dữ liệu ngoài miền huấn luyện, nên kết quả trên đó trả lời luôn câu hỏi về
  khả năng tổng quát hoá.
- Không bao giờ tạo nhãn hay chỉnh tham số trên tập đánh giá.

## 6. Thứ tự thực hiện đề xuất

```
M1 wrapper + kiểm chứng  →  Cách 1 (M2–M5)  →  Cách 2 (M6)  →  Cách 3 (tuỳ thời gian)
                                                               Cách 4: chỉ nêu ở phần hướng phát triển
```
