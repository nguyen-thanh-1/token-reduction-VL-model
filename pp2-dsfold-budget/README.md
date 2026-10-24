# PP2 — DSFold + Budget head (ngân sách thích ứng)

**Tóm tắt:** PP1 với **K thay đổi theo từng mẫu**. Một MLP nhỏ (khoảng 0,66M
tham số) nhìn vào thống kê của ảnh và câu hỏi, rồi chọn 1 trong 7 mức ngân
sách. Sau đó DSFold (PP1, bộ tham số D) chọn và fold đúng số token đó.

**Công thức:** PP2 = **PP1 (tự cài, bộ D)** + **budget head**.

## Đọc theo thứ tự

1. [`../METHODS.md`](../METHODS.md): tổ chức nhóm, tiêu chí chọn phương pháp
2. [`../00-common/README.md`](../00-common/README.md): ký hiệu, interface, wrapper, 7 mức, đánh giá
3. [`../pp1-dsfold/method.md`](../pp1-dsfold/method.md): **phần nền, phải tự cài**. Chú ý mục 8.0 (bộ D) và mục 13 (bài kiểm tra)
4. [`method.md`](method.md): phần mở rộng budget head

## Phần nào làm theo PP1, phần nào là của PP2

| Thành phần | Nguồn |
| --- | --- |
| S1 Score, S2 DPP, S3 Assign, S4 Fold | PP1 `method.md`, bộ D, **không tự chỉnh** |
| Chọn bao nhiêu token | **PP2**: budget head (`method.md` mục 4–8) |
| Lắp vào model, đánh giá | `00-common` |

## Các bước thực hiện

| # | Việc | Sản phẩm | Xong khi |
| --- | --- | --- | --- |
| 1 | Tự cài PP1 theo `../pp1-dsfold/method.md` với bộ D | module PP1 của riêng bạn | **qua bài kiểm tra PP1 mục 13** |
| 2 | Cài phần trích đặc trưng x (403 chiều), có trả về `sel`, `d2` của greedy DPP | `src/token_reduction_vl/pruning/budget_head.py` | unit test trên tensor giả |
| 3 | Script tạo nhãn: 50k GQA train + 5k GQA val, 7 mức, dùng lại kết quả ViT | `scripts/make_budget_labels.py`, file `.npz` | khoảng 9 giờ GPU; báo cáo tỉ lệ nhãn không đơn điệu |
| 4 | Huấn luyện budget head (CPU), 3 seed | checkpoint + log | AUROC ≥ 0,75 cho `p_1..p_3` trên GQA val |
| 5 | Chọn θ₁, θ₂, θ₃ trên GQA val cho `r̄` ≈ 33,3 / 22,2 / 11,1% | `configs/prune_dsfold_budget.yaml` | khoá θ |
| 6 | Cài các so sánh: ngưỡng DPP, ngưỡng entropy, oracle | — | `method.md` mục 10 |
| 7 | Đánh giá trên dev subset, rồi full benchmark | `outputs/reports/pp2-dsfold-budget-*` | trả lời Q1–Q5 (`method.md` mục 11) |

## Chi phí GPU ước tính

| Việc | Thời gian |
| --- | --- |
| Tạo nhãn (bước 3) | khoảng 9 giờ (chạy qua đêm) |
| Huấn luyện | vài phút trên CPU |
| Đánh giá, 3 điểm vận hành × 3 benchmark | khoảng 5–6 giờ |

## Tiêu chí hoàn thành

Theo `method.md` mục 15 (M6a–M6d). Tiêu chí chính: PP2 nằm **trên** đường nội
suy của PP1 r cố định, ở cùng `r̄` thực tế, trên ≥ 2/3 benchmark, với CI không
chứa 0.
