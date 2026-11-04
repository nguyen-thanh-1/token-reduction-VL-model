# Lộ trình PP3 hướng tới bài báo Q2

Ngày lập: 2026-11-03. Mục tiêu: bài báo tạp chí Q2 về PP3 (DSFold + score head
distill từ attention, focus/context), trên Qwen3-VL-2B và Qwen3-VL-4B.

## Hai bản bài báo (cập nhật 2026-11-03)

- **Bản 1 (an toàn, Q3–Q4 hoặc hội nghị Scopus), bản thảo trước 2026-11-05:**
  giai đoạn A, chốt cấu hình 2B, xác nhận val_5k, đánh giá cuối 2B (D2 chỉ 2B),
  bảng hiệu năng, viết. Bản thảo: `docs/paper/pp3_draft.md`.
- **Bản 2 (mở rộng lên Q2, sau):** giai đoạn B, C, D3, E1, đánh giá cuối trên
  2B + 4B; ≥ 30% nội dung mới, trích dẫn bản 1, không nộp trùng.

Phát hiện hiệu năng (tune2k, cùng lượt chạy, ViT không tính): selector PP3
khoảng 36–48 ms/mẫu, trong khi prefill của LLM 2B với 227 token chỉ khoảng
50 ms và gần như không giảm khi còn 71 token. Trên ảnh GQA, PP3 chưa tăng tốc
thời gian thực. Bản 1 báo cáo FLOPs/số token và thời gian đo riêng
(`run_pruned.py`, có warm-up); tối ưu selector (DPP trên GPU, bỏ vòng Python)
là việc của bản 2.

Quy ước: **[C]** = Claude viết code/phân tích, **[U]** = người dùng chạy GPU.
Mỗi giai đoạn có tiêu chí xong; không sang giai đoạn đánh giá cuối khi chưa
chốt cấu hình (ADR 0002: chọn trên tune2k, xác nhận trên val_5k, rồi mới chạy
testdev / MMBench val / MME test).

## Giai đoạn A — Hạ tầng và baseline ngoài (mốc 2026-11-05)

| Bước | Việc | Ai |
|---|---|---|
| A1 | Hoàn thiện bảng ablation trên tune2k: `dsfoldD_L3/L4`, `3c_L3/L4` | U |
| A2 | Baseline **DivPrune** (max–min đa dạng trên `v`, không fold) | C |
| A3 | Baseline **kiểu FastV** (attention mà token ảnh nhận ở lớp 2 của LLM, top-K, không fold) | C |
| A4 | `run_diagnostics` chạy được MMBench/MME và các baseline ở L1–L5 | C |
| A5 | Chạy baseline trên tune2k ở L1–L4, paired report | U |

Xong khi: một bảng tune2k có random, DivPrune, FastV, DSFold-D, 3a, 3c,
3d_attn, 3d_attn50k, fcd25 ở cùng id.

## Giai đoạn B — Nâng cấp phương pháp trên 2B (tune2k)

Điểm yếu đo được: budget thấp (L2 −1.20, L1 chưa đo) và câu yes/no (thấp hơn
full ở mọi budget đến L5; ngang full chỉ khi giữ 75% không fold).

| Bước | Giả thuyết | Thí nghiệm | Ai |
|---|---|---|---|
| B1 | Fold DeepStack làm hỏng bằng chứng phủ định ở câu yes/no | 3d_attn50k L2–L5 với `gamma=0` (không fold) và `gamma` nhỏ hơn | C+U |
| B2 | Ở budget thấp cần cân bằng khác giữa chất lượng và đa dạng trong DPP | lũy thừa `q̂^a` (a ∈ {0.5, 1, 2, 4}) ở L1–L3 | C+U |
| B3 | Head học tốt top nhỏ nếu nhãn sắc hơn | train 3d với nhiệt độ T < 1 (0.5) | C+U |
| B4 | Kết hợp tốt nhất → ADR 0005b, chốt cấu hình 2B | paired report | C |

Xong khi: cấu hình 2B tốt hơn hoặc ngang các baseline ngoài ở L2–L4, và có
lời giải thích cho câu yes/no.

## Giai đoạn C — Qwen3-VL-4B

| Bước | Việc | Ai |
|---|---|---|
| C1 | Lập kế hoạch bộ nhớ (skill `model-memory-planning`), bỏ các thông số ghi cứng của 2B (lớp teacher theo tỉ lệ độ sâu, số chiều) | C |
| C2 | Tải trọng số 4B (khi người dùng cho phép), baseline full trên tune2k | U |
| C3 | Nhãn attention 4B (10k–20k mẫu), fit compressor, train 3d (+ 3d_contrast) | U |
| C4 | Diagnostics tune2k: full, random, DivPrune, FastV, PP3 cấu hình đã chốt | U |

Xong khi: cùng công thức (không chỉnh tay) cho kết quả tương tự trên 4B.

## Giai đoạn D — Xác nhận và đánh giá cuối

| Bước | Việc | Ai |
|---|---|---|
| D1 | gqa_val_5k (bỏ id trùng tune2k) cho cấu hình đã chốt, 2B và 4B | U |
| D2 | GQA testdev, MMBench val, MME test × {full, random, DivPrune, FastV, PP3, PP3-fc} × L2–L4 × {2B, 4B} | U |
| D3 | 3 seed cho score head chính (2B) | U |
| D4 | Bảng hiệu năng: token, thời gian selector / prefill / decode, VRAM đỉnh | C |

## Giai đoạn E — Phân tích và viết

| Bước | Việc | Ai |
|---|---|---|
| E1 | Chỉ số grounding với scene graph GQA (cần cho phép tải) | C |
| E2 | Chỉ số vùng, attention sink, ví dụ minh họa | C |
| E3 | Hình, bảng, sơ đồ PP3; bản thảo | C+U |

## Rủi ro

- Baseline ngoài ngang PP3 ở L3–L4: khi đó đóng góp dựa vào budget thấp (L1–L2),
  việc không cần attention trong LLM, và phân tích vùng.
- 4B tốn VRAM và thời gian tạo nhãn (eager attention): bắt đầu với 10k nhãn.
- Chọn quá nhiều biến thể trên tune2k gây overfit: mọi cấu hình chốt phải qua val_5k.
