# PP1 — DSFold (không huấn luyện, r cố định)

**Tóm tắt:** sau ViT, chọn K token vừa đa dạng vừa liên quan tới câu hỏi bằng
DPP; gộp (fold) token bị bỏ vào kênh DeepStack của token giữ lại; đưa K token
vào LLM với toạ độ M-RoPE gốc. Không huấn luyện gì.

**Vai trò đặc biệt:** tài liệu [`method.md`](method.md) của PP1 cũng là
**tài liệu nền** mà thành viên PP2, PP3, PP4 tự cài lại trước khi làm phần mở
rộng của họ. Vì vậy, mọi thay đổi trong `method.md` phải được **cả nhóm đồng
ý**, và phải tăng phiên bản bộ D.

## Đọc theo thứ tự

1. [`../METHODS.md`](../METHODS.md): tổ chức nhóm, tiêu chí chọn phương pháp
2. [`../00-common/README.md`](../00-common/README.md): ký hiệu, interface, wrapper, 7 mức, đánh giá
3. [`method.md`](method.md): toàn bộ thuật toán PP1

## Các bước thực hiện

| # | Việc | Sản phẩm | Xong khi |
| --- | --- | --- | --- |
| 1 | Cài `select()` theo interface (`00-common` mục 4.0): S1–S4 | `src/token_reduction_vl/pruning/dsfold.py` | unit test trên tensor giả chạy được trên CPU |
| 2 | Ghi bộ tham số D | `configs/prune_dsfold.yaml` (phiên bản `D-1.0`) | file có đủ mọi tham số ở `method.md` mục 8.0 |
| 3 | Tạo bộ kiểm tra "cài đúng PP1" (100 mẫu) và kết quả tham chiếu | `tests/fixtures/pp1_conformance/`, `tests/test_pp1_conformance.py` | được một thành viên khác review |
| 4 | Ghép với wrapper chung, kiểm chứng `r = 100%` | — | trùng baseline 100/100 mẫu mỗi benchmark |
| 5 | Chỉnh tham số trên GQA val ở L3 | bộ `D-tuned` | khoảng 32 lần chạy (`method.md` mục 8.1) |
| 6 | Chạy dev subset ở 7 mức, rồi full ở L6–L2 | `outputs/reports/pp1-dsfold-*` | đủ bảng ở `method.md` mục 9.1 |
| 7 | Ablation A1–A6 | bảng ablation | `method.md` mục 9.2 |
| 8 | Phân tích định tính (bản đồ token, position bias, MME theo category và kích thước) | hình + bảng | `method.md` mục 9.3 |

## Chi phí GPU ước tính

| Việc | Thời gian |
| --- | --- |
| Chỉnh tham số (bước 5) | khoảng 2 giờ |
| Dev subset, 7 mức × 3 benchmark | vài giờ |
| Full benchmark, 5 mức | khoảng 1,5–2 giờ mỗi mức |
| Ablation | khoảng 1 ngày |

## Tiêu chí hoàn thành

Theo `method.md` mục 11 (M1–M5). Tiêu chí chính: DSFold thắng phương pháp tốt
nhất trong các đối chứng ở L3 và L2 trên ≥ 2/3 benchmark, với CI không chứa 0.
