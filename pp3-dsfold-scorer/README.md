# PP3 — DSFold + Score head (điểm token học được)

**Tóm tắt:** PP1 nhưng thay **công thức chấm điểm token** (S1) bằng một mạng
nhỏ học được. Mạng học từ **saliency**: mỗi token ảnh hưởng tới câu trả lời
của model full-token bao nhiêu. Vẫn giữ DPP, Assign, Fold của PP1, và r cố
định ở 7 mức.

**Công thức:** PP3 = **PP1 (tự cài, bộ D)**, trong đó S1 được thay bằng **score head**.

## Đọc theo thứ tự

1. [`../METHODS.md`](../METHODS.md): tổ chức nhóm, tiêu chí chọn phương pháp
2. [`../00-common/README.md`](../00-common/README.md): ký hiệu, interface, wrapper, 7 mức, đánh giá
3. [`../pp1-dsfold/method.md`](../pp1-dsfold/method.md): **phần nền, phải tự cài**. Chú ý mục 8.0 (bộ D) và mục 13 (bài kiểm tra)
4. [`method.md`](method.md): phần mở rộng score head

## Phần nào làm theo PP1, phần nào là của PP3

| Thành phần | Nguồn |
| --- | --- |
| `rel`, `ms` (làm đặc trưng đầu vào) | PP1 mục 3, bộ D |
| **S1: điểm `quality`** | **PP3**: `q̂ = softplus(s_θ) + 1e-3` (`method.md` mục 2–3) |
| S2 DPP, S3 Assign, S4 Fold | PP1, bộ D, **không tự chỉnh** (DPP dùng `q̂` thay cho `quality`) |
| r cố định, 7 mức | `00-common` mục 6 |

## Các bước thực hiện

Cấu trúc code và nơi lưu kết quả của cả nhóm: [`../METHODS.md`](../METHODS.md) mục 8–9.

| # | Việc | Sản phẩm | Xong khi |
| --- | --- | --- | --- |
| 1 | Tự cài PP1 theo `../pp1-dsfold/method.md` với bộ D | `src/token_reduction_vl/pruning/pp3_scorer/dsfold_base/` | **qua bài kiểm tra PP1 mục 13** |
| 2a | Fit PCA cố định trên 1.000 mẫu train (`method.md` mục 2) | `scripts/pp3/fit_compressor.py` → `data/features/pp3_scorer/compressor.npz` | khoảng 5 phút GPU |
| 2b | Script tạo nhãn saliency (tuyến B): 50k GQA train, full token, 1 forward + 1 backward | `scripts/pp3/make_saliency_labels.py` → shard `data/features/pp3_scorer/saliency_gqa_train_50k_*.npz` | khoảng 6 giờ GPU; chia việc bằng `--start/--end` |
| 3 | Cài 3 kiến trúc 3a, 3b, 3c và loss ListNet | `src/token_reduction_vl/pruning/pp3_scorer/models.py`, `train.py` | unit test trên tensor giả |
| 4 | Huấn luyện 3a, rồi 3b và 3c (không cần VLM) | `outputs/pp3_scorer/checkpoints/` (không lên Git) + `outputs/pp3_scorer/training/` | recall@K ≥ `quality` thủ công trên GQA val |
| 5 | Ghép score head vào PP1 (thay S1), chạy dev subset 7 mức | `configs/methods/pp3_scorer.yaml` | so sánh được với PP1 bộ D ở cùng r |
| 6 | Đánh giá full benchmark | `outputs/pp3_scorer/runs/<run_id>/`, báo cáo ở `outputs/pp3_scorer/reports/<report_id>/` | đủ bảng ở `method.md` mục 6 |
| 7 | (Tuỳ chọn) Tuyến A: spike 4D mask, chưng cất end-to-end | — | `method.md` mục 5 |

## Chi phí GPU ước tính

| Việc | Thời gian |
| --- | --- |
| Nhãn saliency (bước 2) | khoảng 6 giờ (chạy qua đêm) |
| Huấn luyện tuyến B | dưới 1 giờ cho mỗi kiến trúc |
| Đánh giá | tương tự PP1 |
| Tuyến A (tuỳ chọn) | khoảng 12 giờ, cộng spike khoảng 1 ngày |

## Tiêu chí hoàn thành

Theo `method.md` mục 9 (M7a–M7c). Tiêu chí chính: PP3 thắng PP1 (bộ D) ở ≥ 2
mức r trên GQA testdev, với CI không chứa 0, và không thua trên MMBench và MME.
