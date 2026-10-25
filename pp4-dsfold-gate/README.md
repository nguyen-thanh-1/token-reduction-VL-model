# PP4 — DSFold + Cổng học được (học toàn bộ)

**Tóm tắt:** một mạng nhỏ (khoảng 2,6M tham số) cho **mỗi token** một cổng
đóng/mở. Số cổng mở chính là K, nên K tự thay đổi theo từng mẫu. Mạng được
huấn luyện end-to-end qua model 2B đóng băng (khung rate–utility). Token bị bỏ
vẫn được Fold vào DeepStack theo PP1.

**Công thức:** PP4 = **PP1 (tự cài, bộ D: dùng Fold và `I_DPP` để neo)** + **mạng cổng hard-concrete**.

⚠️ Đây là phương pháp **tốn GPU nhất (khoảng 3–5 ngày)** và **rủi ro cao
nhất**. Phải lên lịch GPU với cả nhóm ngay từ đầu.

## Đọc theo thứ tự

1. [`../METHODS.md`](../METHODS.md): tổ chức nhóm, tiêu chí chọn phương pháp, lịch GPU
2. [`../00-common/README.md`](../00-common/README.md): ký hiệu, interface, wrapper, 7 mức, đánh giá
3. [`../pp1-dsfold/method.md`](../pp1-dsfold/method.md): **phần nền, phải tự cài**. Chú ý mục 8.0 (bộ D) và mục 13 (bài kiểm tra)
4. [`method.md`](method.md): phần mở rộng cổng học được

## Phần nào làm theo PP1, phần nào là của PP4

| Thành phần | Nguồn |
| --- | --- |
| `rel`, `ms` (đặc trưng đầu vào) | PP1 mục 3, bộ D |
| `I_DPP` ở L3 (để neo lúc huấn luyện) | PP1 S1–S2, bộ D |
| S3 Assign, S4 Fold | PP1, bộ D, **không tự chỉnh** |
| **Chọn bao nhiêu + chọn token nào** | **PP4**: cổng hard-concrete (`method.md` mục 2–3) |
| Spike custom 4D mask, huấn luyện | **PP4**: `method.md` mục 4–6 |

## Các bước thực hiện

Cấu trúc code và nơi lưu kết quả của cả nhóm: [`../METHODS.md`](../METHODS.md) mục 8–9.

| # | Việc | Sản phẩm | Xong khi |
| --- | --- | --- | --- |
| 1 | Tự cài PP1 theo `../pp1-dsfold/method.md` với bộ D | `src/token_reduction_vl/pruning/pp4_gate/dsfold_base/` | **qua bài kiểm tra PP1 mục 13** |
| 2 | **Spike** custom 4D float attention mask (làm sớm, quyết định tính khả thi) | `scripts/pp4/spike_attention_bias.py` → `outputs/pp4_gate/spike/` | qua 4 kiểm tra ở `method.md` mục 4.1 |
| 3 | Cài mạng cổng + hard-concrete + 5 thành phần loss | `src/token_reduction_vl/pruning/pp4_gate/` (`hard_concrete.py`, `model.py`, `losses.py`, `attention_bias.py`) | unit test trên tensor giả |
| 4 | Tính trước teacher logit (top-64) cho 50k GQA train | `scripts/pp4/make_teacher_logits.py` → `data/features/pp4_gate/teacher_gqa_train_50k.npz` | khoảng 2 giờ GPU |
| 5 | Chạy thử 1 giá trị λ trên 5k mẫu | `outputs/pp4_gate/training/` (log `E[K]/N`, KL, histogram π) | không rơi vào chế độ hỏng (`method.md` mục 7) |
| 6 | Huấn luyện đủ 5 giá trị λ | `outputs/pp4_gate/checkpoints/` (không lên Git) | khoảng 10 giờ mỗi λ |
| 7 | Đánh giá trên dev subset, rồi full benchmark | `outputs/pp4_gate/runs/<run_id>/`, báo cáo ở `outputs/pp4_gate/reports/<report_id>/` | đủ bảng ở `method.md` mục 8 |

**Điểm dừng sớm:** nếu bước 2 thất bại (không có cách nào đưa bias vào
attention), hoặc bước 5 không ổn định sau 2–3 lần chỉnh, thì báo nhóm và
chuyển PP4 thành "thiết kế + kết quả sơ bộ" cho chương hướng phát triển.

## Chi phí GPU ước tính

| Việc | Thời gian |
| --- | --- |
| Spike (bước 2) | khoảng 1 ngày, chủ yếu thử code |
| Teacher (bước 4) | khoảng 2 giờ |
| Chạy thử (bước 5) | khoảng 1–2 giờ mỗi lần |
| Huấn luyện 5 λ (bước 6) | khoảng 50 giờ |
| Đánh giá | khoảng 5–6 giờ |

## Tiêu chí hoàn thành

PP4 nằm trên đường nội suy của PP1 r cố định, ở cùng `r̄`, trên ≥ 2/3
benchmark (CI không chứa 0), ở ít nhất 2 giá trị λ.
