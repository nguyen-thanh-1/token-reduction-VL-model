# Kế hoạch hoàn thiện paper PP3 cho đợt nộp thứ hai (20/11/2026)

Mục tiêu: nâng paper từ mức "Scopus conference / Q4" lên mức chắc chắn hơn,
bằng cách lấp các điểm reviewer sẽ hỏi đầu tiên. Thứ tự dưới đây là thứ tự ưu
tiên (lợi ích trên mỗi giờ GPU). Mỗi gói ghi rõ phần Claude làm (code, test CPU,
viết paper) và phần người dùng chạy (GPU).

## Hiện trạng (04/11)

- Chất lượng: PP3 tốt nhất ở mọi mức L1–L4 trên GQA, MMBench, MME perception;
  hơn DivPrune, FastV-style, CDPruner, ngẫu nhiên. Non-inferiority trên GQA ở L3/L4.
- Điểm yếu lớn nhất: **không có tăng tốc thực tế**. Bộ chọn tốn gần bằng phần
  prefill tiết kiệm được (MME L3: bộ chọn 303 ms, prefill 97 ms so với 396 ms).
- Mới có 3 benchmark, 1 mô hình (2B), 1 seed.

## WP1 — Tăng tốc thực tế (xong, ưu tiên 1)

Phân tích (`scripts/pp3/profile_selector.py`, GPU, đầu vào giả có kích thước
thật): gần như toàn bộ thời gian bộ chọn là vòng lặp DPP tham lam, bị giới hạn
bởi số lần gọi kernel (khoảng 12 kernel nhỏ cho mỗi token chọn), không phải
bởi số phép tính. FLOPs của bộ chọn chỉ khoảng 0,2–3% FLOPs prefill
(`scripts/pp3/flops_report.py`).

Đã làm:
- DPP và max–min chạy bằng CUDA graph (16 bước mỗi graph, chỉ số giữ trên GPU).
  Cùng kết quả chọn với bản cũ (kiểm tra trên 160 đầu vào ngẫu nhiên và toàn bộ
  fixture conformance; chỉ một trường hợp CPU N=3072, K=1536 lệch do thứ tự cộng
  số thực, nên CPU vẫn dùng bản cũ).
- Đo trên đầu vào giả: N=1813, K=403 (MME L3): DPP 121 → 23 ms; K=605: 183 → 30 ms.
- `run_diagnostics.py --shuffle-order SEED`: xáo thứ tự cấu hình theo từng câu,
  để kiểm tra bất thường "L3 chậm hơn L4".
- `scripts/pp3/compare_keep.py`: so tập token giữ lại giữa hai lượt chạy.

Người dùng chạy (runbook, Bước 7): kiểm tra trùng kết quả (~20 phút), đo lại
thời gian có xáo thứ tự (~1,5 giờ, chạy một mình).

Kết quả (04/11, đã vào paper): token trùng 2099/2100; MME chọn + prefill nhanh hơn
full 1,75× (L1), 1,53× (L3), 1,31× (L4); FLOPs prefill giảm 5,2× ở L3 trên toàn MME.
GQA không nhanh hơn ở batch 1 (prefill bị giới hạn bởi số lần gọi kernel).
Còn có thể làm: lưu sẵn graph theo kích thước (bỏ chi phí capture), để GQA và K < 48
cũng dùng graph.

## WP2 — Thêm benchmark (ưu tiên 2) — ĐÃ XONG 05/11 (bảng tab:pope, mục sec:pope)

Kết quả: POPE — PP3 non-inferiority ở L2–L4 (bootstrap theo ảnh), L1 mất 1,07 so với 9,04 (DivPrune) / 10,62 (CDPruner); baseline bỏ sót vật thể (yes% 29–37 so với 45,8). ScienceQA-IMG — PP3 tốt nhất L1–L3, ngang DivPrune ở L4; không phương pháp nào đạt non-inferiority. Lưu ý: random_L* trên POPE chạy lại từ câu ~4918 nên không tái lập bit-exact.

- **POPE** (ảo giác; adversarial/popular/random, 3×3000 câu, 500 ảnh COCO val2014).
  Báo cáo F1 trung bình 3 tập (yes là lớp dương), accuracy và tỉ lệ trả lời yes.
- **ScienceQA-IMG** (test, 2017 câu có ảnh, gợi ý dài).
- Tùy chọn: **TextVQA val** (OCR), để ở phụ lục nếu còn thời gian.

Đã làm: ADR 0006, adapter `pope.py` / `sqa.py`, downloader có revision cố định và
lưu mỗi ảnh COCO một lần, metric POPE (F1, yes-ratio), cấu hình `pope_test` /
`sqa_img_test`, cột POPE F1 trong `final_report.py`, 7 test.
Người dùng: đồng ý tải (~0,5 GB Parquet), tiền xử lý, chạy GPU
(POPE ~1 giờ/phương pháp cho 4 mức, SQA ~15 phút/phương pháp).

## WP5 — Mô hình thứ hai: Qwen3-VL-4B (ưu tiên 3) — ĐÃ XONG 06/11 (bảng tab:4b, mục sec:4b)

Kết quả (head `3d_attn4b50k_rec`, 50k nhãn, chọn checkpoint theo recall held-out; ADR 0007): DeSTiL tốt nhất ở L1 và L3 trên GQA, MMBench, POPE, MME perception; hơn baseline mạnh nhất có ý nghĩa ở mọi ô trừ POPE L3 so với CDPruner. L3 so với full: POPE non-inferiority, MME perception −8,7 (không khác 0), GQA −1,23, MMB −2,26. MME cognition: DivPrune cao nhất. MME chọn + prefill nhanh hơn full 5,7× (L1), 2,5× (L3). Chưa chạy L2/L4 (đợt B).


Một mô hình duy nhất là điểm reviewer dễ bắt nhất. Cần: tải trọng số 4B (~9 GB,
cần đồng ý), sinh nhãn teacher (bắt đầu với 20k thay vì 50k nếu thời gian không đủ),
huấn luyện head, đánh giá L1–L4 trên GQA testdev, MMBench, MME, POPE với full,
DivPrune, CDPruner, PP3. Claude kiểm tra trước: kích thước đặc trưng, lớp teacher
(theo tỉ lệ độ sâu), bộ nhớ (`model-memory-planning`).

## WP3 — Nhiều seed (ưu tiên 4)

Huấn luyện lại head 3d_attn50k với seed 1 và 2 (nhãn đã có), đánh giá GQA
testdev + MMBench ở L1–L4. Paper ghi trung bình ± độ lệch chuẩn.

## WP4 — Ablation còn thiếu (ưu tiên 5, chỉ cần huấn luyện lại)

- Trọng số nhãn answer/question: 0 / 0,5 / 1 (nhãn đã lưu riêng hai phần).
- Bỏ đặc trưng log r.
- Lớp teacher (8–19 so với 4–11 / 16–27): cần sinh lại nhãn → chỉ làm nếu còn thời gian.

Đánh giá trên tune2k (15 phút mỗi cấu hình).

## WP6 — Bản tiếng Anh (bắt buộc: hội nghị nhận bài tiếng Anh)

`docs/paper/latex/pp3_paper_en.tex`, cùng cấu trúc, hình có nhãn tiếng Anh
(`paper_figures.py --lang en`). Quy tắc của thầy giữ nguyên: không có mục
Limitations / Future work, không ngôi thứ nhất ("this paper", câu bị động).
Dịch các mục ổn định (phương pháp, liên quan, thiết lập) sớm; bảng kết quả dịch sau cùng.

## Lịch đề xuất

| Ngày | Người dùng (GPU) | Claude |
|---|---|---|
| 04–05/11 | tải + chạy POPE, SQA (qua đêm) | ADR/adapter (xong); kiểm tra 4B |
| 06–09/11 | 4B: sinh nhãn teacher, huấn luyện head | bản EN: phương pháp, liên quan; hình EN |
| 09–11/11 | 4B: đánh giá; seed 1, 2 | bảng POPE/SQA vào paper |
| 11–13/11 | WP4 trên tune2k | bảng 4B, seed, ablation |
| 13–17/11 | chạy bù | bản EN hoàn chỉnh, đọc lại |
| 17–20/11 | — | sửa lần cuối, nộp |

## Ý tưởng để sau (05/11): "cắt rồi phóng to" ở cùng ngân sách token

Mục tiêu: vượt mô hình full chứ không chỉ ngang. Lý do: head chưng cất attention
của mô hình full nên trần là ngang full; bỏ token nhiễu không giúp (lowdrop10/25 trên
tune2k: −0,20…+0,00), dù oracle dùng đáp án đúng cho +3,4 điểm. Hướng đề xuất: dùng
q̂ (có trước khi LLM chạy) để chọn vùng, crop vùng đó ở độ phân giải cao hơn, mã hoá
thêm bằng ViT, đưa vào LLM ~22% token toàn ảnh + token vùng crop, tổng ≤ full.
Gần với ViCrop ("MLLMs Know Where to Look", ICLR 2025) nhưng không cần lượt LLM
thêm. Cổng: pilot 2B trên tune2k + tập con MME; chỉ tiếp tục nếu ≥ +1 điểm so với
full với số token ≤ full. Cần ADR 0008. Hoãn đến khi xong WP5 (4B).
