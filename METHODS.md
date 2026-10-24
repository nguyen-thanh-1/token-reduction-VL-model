# Bốn phương pháp giảm visual token — chỉ mục và tổ chức nhóm

- Trạng thái: **Đề xuất** (chưa triển khai). Phương pháp được chọn sẽ ghi vào ADR 0002
- Cập nhật: 2026-10-24
- Mục tiêu: **thử 4 phương pháp độc lập, mỗi thành viên một phương pháp, rồi chọn 1 phương pháp tốt nhất** cho khoá luận

> **7 mức ngân sách** (dùng chung cho cả 4 phương pháp):
> L7 = 100% · L6 = 75% · L5 = 50% · L4 = 33,3% · L3 = 22,2% · L2 = 11,1% · L1 = 5,6%.
> Chi tiết và số token thật: [`00-common/README.md` mục 6](00-common/README.md#6-các-mức-ngân-sách-7-mức-cố-định).

## 1. Cấu trúc thư mục

```
token-reduction-VL-model/
├── METHODS.md                ← file này: tổ chức nhóm, so sánh, tiêu chí chọn
├── 00-common/README.md       ← PHẦN CHUNG: ký hiệu, interface, wrapper, chi phí, 7 mức, dữ liệu, đánh giá, đối chứng
├── pp1-dsfold/
│   ├── README.md             ← hướng dẫn thành viên PP1
│   └── method.md             ← tài liệu PP1, đồng thời là TÀI LIỆU NỀN cho PP2–PP4
├── pp2-dsfold-budget/
│   ├── README.md             ← hướng dẫn: tự cài PP1 → thêm budget head
│   └── method.md             ← phần mở rộng: budget head
├── pp3-dsfold-scorer/
│   ├── README.md             ← hướng dẫn: tự cài PP1 → thay S1 bằng score head
│   └── method.md             ← phần mở rộng: score head
└── pp4-dsfold-gate/
    ├── README.md             ← hướng dẫn: tự cài PP1 → thêm cổng học được
    └── method.md             ← phần mở rộng: cổng hard-concrete
```

Các folder nằm ngay ở gốc repo, cùng cấp với `src/`, `configs/`, `docs/`.

## 2. Mỗi thành viên đọc gì

| Phương pháp | Thành viên | Đọc | Tự cài |
| --- | --- | --- | --- |
| [**PP1 — DSFold**](pp1-dsfold/README.md) | _(điền tên)_ | `00-common` → `pp1-dsfold/method.md` | PP1 |
| [**PP2 — DSFold + Budget head**](pp2-dsfold-budget/README.md) | _(điền tên)_ | `00-common` → `pp1-dsfold/method.md` → `pp2-dsfold-budget/method.md` | PP1 (bộ D) + budget head |
| [**PP3 — DSFold + Score head**](pp3-dsfold-scorer/README.md) | _(điền tên)_ | `00-common` → `pp1-dsfold/method.md` → `pp3-dsfold-scorer/method.md` | PP1 (bộ D) + score head |
| [**PP4 — DSFold + Cổng học được**](pp4-dsfold-gate/README.md) | _(điền tên)_ | `00-common` → `pp1-dsfold/method.md` → `pp4-dsfold-gate/method.md` | PP1 (bộ D) + cổng |

**Ý nghĩa:**
- **Tài liệu PP1 chỉ có một bản** (`pp1-dsfold/method.md`). PP2–PP4 trỏ link tới
  đó, không sao chép, để không có phiên bản lệch nhau.
- **Mỗi thành viên tự cài PP1** theo cùng tài liệu và cùng **bộ tham số D**
  (PP1 mục 8.0). Không ai phải chờ ai.
- Để chắc rằng 4 bản cài PP1 giống nhau, mọi thành viên phải qua **bài kiểm tra
  "cài đúng PP1"** (PP1 mục 13) trước khi làm phần mở rộng.

## 3. So sánh nhanh 4 phương pháp

| | PP1 | PP2 | PP3 | PP4 |
| --- | --- | --- | --- | --- |
| Bao nhiêu token | r cố định (7 mức) | **học**: chọn 1 trong 7 mức cho từng mẫu | r cố định (7 mức) | **học**: số cổng mở |
| Token nào | DPP + `rel` + `ms` | như PP1 (bộ D) | DPP với **điểm học được** | **cổng học được** |
| Token bị bỏ | Fold vào DeepStack | như PP1 | như PP1 | như PP1 (Fold) |
| Tham số học thêm | 0 | ≈ 0,66M | ≈ 2,1–3,5M | ≈ 2,6M |
| Gradient qua model 2B | không | không | tuyến B: chỉ theo đầu vào; tuyến A: có | có |
| GPU thêm (ước tính) | ≈ 2 giờ chỉnh tham số | ≈ 9 giờ tạo nhãn | ≈ 6 giờ (B); ≈ 12 giờ thêm (A) | ≈ 3–5 ngày |
| Rủi ro | thấp | thấp–trung bình | trung bình | cao |
| Câu hỏi nó trả lời | công thức toán + Fold có đủ tốt không? | chọn K theo từng mẫu có giúp không? | học điểm token có hơn công thức không? | học toàn bộ có đáng chi phí không? |

Mọi con số chi phí là ước tính từ thời gian chạy baseline. Cần đo lại trên GPU thực tế.

## 4. Lịch làm việc (song song từ ngày đầu)

```
Tuần 1   Mọi thành viên: đọc 00-common + PP1 method.md; tự cài PP1 (select() trên tensor giả)
         Song song, chia việc chung (00-common mục 10):
           · 1 người: wrapper suy luận + kiểm chứng r = 100%
           · 1 người: bộ đánh giá (agree, bootstrap, đo thời gian) + danh sách dữ liệu chung
           · thành viên PP1: bộ D (configs/prune_dsfold.yaml) + bộ kiểm tra "cài đúng PP1"
           · thành viên PP4: bắt đầu spike 4D mask

Tuần 2   Mọi thành viên qua bài kiểm tra "cài đúng PP1" (PP1 mục 13)
         Chạy chung các đối chứng (P0, Random-K, DivPrune, FastV …) một lần

Tuần 3+  Mỗi thành viên làm phần mở rộng của mình (theo README trong folder)
         GPU dùng theo lịch chung (mục 6)

Cuối     So sánh 4 phương pháp theo mục 5 → chọn 1 → ghi ADR 0002
```

## 5. Cách chọn phương pháp thắng

### 5.1 So sánh công bằng

| Loại | Phương pháp | Cách so |
| --- | --- | --- |
| r cố định | PP1, PP3 | so trực tiếp ở từng mức L1–L7 |
| K thích ứng | PP2, PP4 | so ở cùng **tỉ lệ giữ trung bình thực tế** `r̄`, bằng cách nội suy đường của PP1 (`pp2-dsfold-budget/method.md` mục 9) |

Mọi phương pháp dùng **cùng** wrapper, dev subset, 3 benchmark, chỉ số và
bootstrap (`00-common` mục 7–8), và **cùng** bộ đối chứng (`00-common` mục 9).

### 5.2 Tiêu chí (thống nhất trước khi chạy, không đổi sau khi thấy kết quả)

| # | Tiêu chí | Đo bằng | Ngưỡng |
| --- | --- | --- | --- |
| 1 | Chất lượng | % hiệu năng so với full token ở `r̄` ≈ 22,2% và 11,1% | thắng P0 và DivPrune với CI 95% không chứa 0, ở ≥ 2/3 benchmark |
| 2 | Tổng quát hoá | kết quả trên MMBench và MME (không dùng để huấn luyện hay chỉnh tham số) | không thua PP1 quá 1 điểm % trên benchmark nào |
| 3 | Chi phí | thời gian bộ chọn + chi phí huấn luyện | bộ chọn < 5% thời gian prefill |
| 4 | Độ đơn giản | số tham số học, có cần huấn luyện không | **nếu CI của hai phương pháp chồng nhau thì chọn phương pháp đơn giản hơn** |

### 5.3 Các tình huống có thể xảy ra

| Kết quả | Chọn | Khoá luận trình bày |
| --- | --- | --- |
| PP2 thắng PP1 rõ | PP2 | phương pháp thích ứng; PP1 làm ablation |
| PP3 thắng PP1 rõ, PP2 không | PP3 | điểm học được; PP2 là kết quả âm |
| PP2 và PP3 đều thắng | phương pháp tốt hơn; có thể thử ghép PP1 + PP2 + PP3 (chúng thay hai thành phần khác nhau) | |
| Không phương pháp nào hơn PP1 | PP1 | phương pháp không huấn luyện; 3 phương pháp kia là bằng chứng |
| PP1 không thắng đối chứng | xem lại ablation A1–A6 của PP1 | chuyển trọng tâm sang nghiên cứu "cắt token khi có DeepStack" |

## 6. Lịch dùng GPU (một GPU dùng chung)

| Job | Ai | Thời gian ước tính | Ghi chú |
| --- | --- | --- | --- |
| Đối chứng chung | chia đều | khoảng 1–2 ngày | chạy một lần, mọi người dùng chung |
| Chỉnh tham số PP1 | PP1 | khoảng 2 giờ | |
| Nhãn budget head | PP2 | khoảng 9 giờ | chạy qua đêm |
| Nhãn saliency | PP3 | khoảng 6 giờ | chạy qua đêm |
| Huấn luyện cổng, 5 λ | PP4 | khoảng 50 giờ | **xếp lịch sớm nhất**, vì dài nhất |
| Đánh giá full mỗi phương pháp | từng người | khoảng 1,5–2 giờ mỗi mức | |

Quy tắc: ghi lịch vào một bảng chung trước khi chạy job dài hơn 2 giờ.
Không chạy hai job GPU cùng lúc, vì sẽ sai số liệu thời gian và VRAM.

## 7. Quy tắc làm việc nhóm

| Quy tắc | Nội dung |
| --- | --- |
| Sửa tài liệu chung | `00-common/README.md` và `pp1-dsfold/method.md` chỉ được sửa khi **cả nhóm đồng ý**. Đổi bộ D thì phải tăng phiên bản (`D-1.0` → `D-1.1`) và mọi người chạy lại bài kiểm tra PP1 |
| Code | mỗi phương pháp một module trong `src/token_reduction_vl/pruning/`; không import module của phương pháp khác |
| Nhánh git | `feat/pp<N>-<tên>` tách từ `dev`; phần chung trên `feat/common-<tên>` |
| Config và báo cáo | mỗi phương pháp có `configs/prune_<tên>.yaml` và `report.id` riêng; ghi phiên bản bộ D, git SHA, phiên bản dữ liệu chung |
| Dữ liệu huấn luyện | dùng chung **một** danh sách 50k GQA train / 5k GQA val (`00-common` mục 7) |
| Báo cáo baseline | không ai được ghi đè `outputs/reports/baseline/` |
