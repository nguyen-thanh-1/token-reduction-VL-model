# Cách 2 — DSFold-A: mạng nhỏ chọn ngân sách K, DSFold chọn token

- Trạng thái: Đề xuất
- Vai trò: **phần "adaptive" chính của khoá luận**
- Bao nhiêu token: **mạng nhỏ có huấn luyện (budget head)** · Token nào: **DSFold cách 1, tham số đã khoá**
- Điều kiện bắt đầu: cách 1 đạt M5, và bộ tham số DSFold đã khoá
- Ký hiệu và giao thức chung: [`README.md`](README.md)

## 1. Bài toán

Cách 1 dùng cùng một r cho mọi mẫu. Nhưng mức token **đủ dùng** khác nhau giữa các mẫu:

- *"Is the sky blue?"* trên ảnh đơn giản: có thể chỉ cần 5,6%.
- *"What is written on the sign?"* trên ảnh lớn: có thể cần 50–75%.

**Cho:** đặc trưng của cặp (ảnh, câu hỏi) **đã có sẵn** sau ViT và sau S1–S2 của DSFold.

**Tìm:** mức `m*(x) ∈ {L1, …, L7}` (README mục 6) **nhỏ nhất** mà câu trả lời
vẫn giữ nguyên như full-token.

**Ràng buộc:**
- không huấn luyện lại VLM, không cần gradient qua VLM;
- budget head phải rẻ hơn nhiều so với 1 decoder layer.

## 2. Một tính chất quan trọng: greedy DPP cho các tập lồng nhau

Greedy DPP (cách 1, mục 4.2) chọn token **lần lượt**. Tập K₁ token đầu tiên
luôn là tập con của tập K₂ token đầu tiên khi K₁ < K₂. Vì vậy:

- **Chạy greedy một lần** tới `K_max = K(L6 = 75%)` là có ngay `I_keep` cho mọi
  mức L1–L6, theo thứ tự chọn (`sel`), rồi sắp tăng dần.
- Các giá trị `d2` lúc chọn mỗi token cho biết "phần thông tin mới" của token
  đó. Đây là đặc trưng quan trọng cho budget head (mục 4).
- Fold (S3–S4) được tính riêng cho từng mức, vì tập `I_drop` khác nhau.

## 3. Định nghĩa nhãn

Với mỗi mẫu x và mỗi mức m ∈ {1..6} (L7 = 100% luôn đúng với chính nó):

```
y_m(x) = 1[ norm(pred_m(x)) == norm(pred_full(x)) ]
```

- `pred_m` là câu trả lời của DSFold ở mức m; `pred_full` là câu trả lời full-token.
- `norm` là `normalize_answer` trong `evaluation/runner.py`: bỏ khoảng trắng thừa, chuyển chữ thường.
- **Lý do dùng "khớp với full-token" thay cho "đúng đáp án":**
  - Mục tiêu là giữ hành vi của baseline.
  - Khi baseline sai, nhãn "đúng đáp án" sẽ bằng 0 ở mọi mức, gây nhiễu.
  - Nhãn này tạo được mà không cần đáp án.
- Ablation: `y_m = 1[pred_m đúng đáp án]`.

**Nhãn oracle:** `m_oracle(x) = min{ m : y_m(x) = 1 }`, hoặc 7 nếu không có mức
nào đúng. Đây là cận trên: biết trước mức tối ưu cho từng mẫu.

Phải báo cáo **tỉ lệ mẫu có nhãn không đơn điệu** (ví dụ `y_2 = 1` nhưng
`y_3 = 0`). Trường hợp này xảy ra vì cắt token là phép biến đổi phi tuyến.

## 4. Đặc trưng đầu vào x

Mọi đặc trưng đều được tính từ tensor **đã có** trong pipeline cách 1:

| Nhóm | Định nghĩa chính xác | Kích thước |
| --- | --- | ---: |
| Ảnh | `x_v = mean_i ṽ_i` (trung bình vector đơn vị), rồi `P_v: Linear(2048 → 128)` | 128 |
| Câu hỏi | `x_q = mean_j q̃_j`, rồi `P_q: Linear(2048 → 128)` | 128 |
| Tương tác | `P_v(x_v) ⊙ P_q(x_q)` | 128 |
| Kích thước | `log N`, `log(H′/W′)`, `log(1 + N_q)` | 3 |
| Liên quan câu hỏi | mean, std, max của `rel`; entropy chuẩn hoá `H(softmax(rel/0,05)) / log N`; tỉ lệ tổng `rel⁺` nằm trong top-10% token | 5 |
| DeepStack | mean và std của `ms` (biến thể đã khoá ở cách 1) | 2 |
| **Đường cong DPP** | `ρ_m = d2_{sel[K_m − 1]} / d2_{sel[0]}`, với m = 1..6: phần thông tin mới của token cuối cùng ở mỗi mức, so với token đầu tiên | 6 |
| Loại câu hỏi | 3 cờ nhị phân theo danh sách từ khoá cố định: **đếm** ("how many", "number of"), **đọc chữ** ("written", "say", "text", "word", "letter", "brand"), **vị trí** ("left", "right", "above", "below", "behind", "front", "next to") | 3 |

Tổng cộng 403 chiều. **Không** dùng loại tác vụ (open / MC / yes-no) làm đặc
trưng: dữ liệu huấn luyện chỉ có GQA (open), nên đặc trưng đó sẽ là hằng số lúc
huấn luyện và trở thành ngoài miền lúc đánh giá.

**Ý nghĩa của `ρ_m`:** `ρ_m` nhỏ nghĩa là token thứ `K_m` gần như không mang
thông tin mới, tức ảnh đã "bão hoà" ở mức m. Nhiều khả năng mức m đã đủ.

## 5. Kiến trúc: mô hình thứ bậc (cumulative link)

Một điểm "độ dễ" duy nhất `g(x)`, cùng 6 ngưỡng **tăng dần**:

```
h    = GELU(Linear(403 → 256)(x));  h = Dropout(0,1)(h)
h    = GELU(Linear(256 → 128)(h))
g(x) = Linear(128 → 1)(h)                                   # số thực: mẫu càng "dễ" càng lớn
b_1  = tham số tự do;  b_m = b_1 + Σ_{u<m} softplus(δ_u)    # b_1 ≤ b_2 ≤ … ≤ b_6
p_m(x) = σ( g(x) + b_m )                                    # P(mức m là đủ)
```

- **Đơn điệu theo cấu trúc:** vì `b_m` tăng, nên `p_1 ≤ p_2 ≤ … ≤ p_6` với
  **mọi** x. Mức lớn hơn luôn được coi là ít nhất cũng đủ như mức nhỏ hơn.
- **Số tham số:** `P_v` + `P_q` = 2 × 262.272; MLP khoảng 136.000; tổng **khoảng
  0,66M**.
- **Ablation:** 6 đầu ra độc lập `p_m = σ(Linear(128 → 6))` (không có ràng buộc
  đơn điệu).

## 6. Huấn luyện

**Loss:** binary cross-entropy có trọng số, tính cho từng mức:

```
L = (1/6) Σ_{m=1..6} [ −w⁺_m · y_m · log p_m − (1 − y_m) · log(1 − p_m) ]
w⁺_m = (#mẫu y_m = 0) / (#mẫu y_m = 1)       # cân bằng lớp, tính trên tập train
```

| Thiết lập | Giá trị |
| --- | --- |
| Tối ưu | AdamW, lr 1e-3, weight decay 1e-2, cosine decay |
| Batch | 512 |
| Epoch | tối đa 50; early stopping theo loss trên 10% giữ lại (tách theo `image_id`), patience 5 |
| Chuẩn hoá đặc trưng vô hướng | z-score với mean/std tính trên tập train |
| Seed | 3 seed {0, 1, 2}; báo cáo trung bình ± độ lệch chuẩn |
| Phần cứng | CPU là đủ (vài phút) |

## 7. Tạo dữ liệu huấn luyện

1. Lấy **50.000 câu từ GQA train balanced**: chọn ngẫu nhiên theo `image_id`
   với seed 42, mỗi ảnh tối đa 3 câu để tăng độ đa dạng ảnh.
2. Với mỗi câu:
   - chạy ViT **một lần** để có `v`, `f_l`;
   - chạy greedy DPP **một lần** tới `K_max` để có `sel` và `d2`;
   - chạy LLM 7 lần: Full và L1–L6. Mỗi mức tính Fold riêng, dùng lại kết quả ViT;
   - lưu ra `.npz`: đặc trưng x (403 chiều, fp16), các nhãn `y_1..y_6`, 7 câu
     trả lời, `N`, `K_m`.
3. **Chi phí ước tính:** ViT khoảng 46% thời gian một lượt full. Dùng lại ViT
   cho 7 lượt LLM thì mỗi mẫu tốn khoảng `0,155 · (0,46 + 7 · 0,54 · r̄_LLM)`,
   trong đó `r̄_LLM` là tỉ lệ chi phí LLM trung bình qua 7 mức (≈ 0,5). Ra khoảng
   0,37 s mỗi mẫu, tức **khoảng 5–6 giờ** cho 50k mẫu. Tính dư (I/O, DPP) thì
   **khoảng 8 giờ**.
4. **Tập chọn ngưỡng:** làm tương tự cho **5.000 câu GQA val**, khoảng 1 giờ.

## 8. Suy luận và cách chọn ngưỡng θ

**Quy tắc chọn mức:**
```
m*(x) = min{ m ∈ 1..6 : p_m(x) ≥ θ },   hoặc 7 nếu không có m nào thoả
K(x)  = clip(⌈r_{m*} · N⌉, 1, N)
```

θ là núm điều chỉnh duy nhất: θ cao thì giữ nhiều token và an toàn hơn; θ thấp
thì tiết kiệm hơn.

**Chọn θ trên GQA val (5.000 câu):**
1. Quét θ ∈ {0,50; 0,55; …; 0,99}. Với mỗi θ, tính `r̄(θ) = ΣK / ΣN` và `agree(θ)`.
2. Chọn ba **điểm vận hành** θ₁, θ₂, θ₃ sao cho `r̄ ≈ 33,3%`, `22,2%`, `11,1%`
   trên GQA val.
3. **Khoá** θ₁, θ₂, θ₃, rồi chạy trên 3 benchmark đánh giá.

Chi phí lúc suy luận: một MLP 0,66M tham số trên vector 403 chiều, tức **dưới
1 MFLOP**, không đáng kể.

## 9. So sánh công bằng khi r̄ thực tế khác mục tiêu

Trên MMBench và MME, `r̄` thực tế sẽ lệch khỏi mục tiêu vì phân bố dữ liệu khác.
Cách so sánh:

- Với mỗi benchmark, nối các điểm (r, metric) của **DSFold r cố định** (7 mức)
  thành đường gấp khúc tuyến tính.
- DSFold-A tại `(r̄_thực, metric)` được gọi là **tốt hơn** khi nó nằm **trên**
  đường đó, tức metric cao hơn giá trị nội suy tại cùng `r̄_thực`.
- Báo cáo chênh lệch này kèm CI 95% theo bootstrap (README mục 8.4), bootstrap
  đồng thời cả hai phương pháp trên cùng tập ảnh.

## 10. Các phương pháp so sánh (cùng bộ chọn token DSFold)

| Phương pháp | Cách chọn mức | Có huấn luyện |
| --- | --- | --- |
| DSFold r cố định | cùng mức cho mọi mẫu | không |
| **Ngưỡng DPP** | `m* = min{ m : ρ_m ≤ κ }`, κ chọn trên GQA val giống θ | không |
| Ngưỡng entropy | `m*` theo các khoảng entropy chuẩn hoá của `rel`; ranh giới chọn trên GQA val | không |
| **DSFold-A** | budget head | có |
| Oracle | `m_oracle(x)` | cận trên |
| ET-Prune, OccamToken | phương pháp của paper (cài lại nếu khả thi, nếu không thì ghi rõ chỉ trích số liệu) | không |

## 11. Câu hỏi nghiên cứu và chỉ số

| | Câu hỏi | Chỉ số |
| --- | --- | --- |
| Q1 | K thích ứng có hơn K cố định ở cùng `r̄` không? | chênh lệch so với đường nội suy (mục 9) |
| Q2 | Có cần học không, hay ngưỡng DPP là đủ? | DSFold-A so với ngưỡng DPP ở cùng `r̄` |
| Q3 | Học trên GQA có tổng quát hoá sang MMBench và MME không? | Q1 tính riêng trên MMBench và MME |
| Q4 | Đặc trưng nào quan trọng? | bỏ từng nhóm đặc trưng (mục 4), đo AUROC trên GQA val |
| Q5 | Còn cách oracle bao xa? | `r̄` và `agree` của oracle so với DSFold-A |

Chất lượng của budget head (trên GQA val):
- AUROC và Brier score cho từng `p_m`;
- ECE (10 bin);
- confusion matrix giữa `m*` và `m_oracle`.

## 12. Phân tích

- Phân bố `m*` theo benchmark và theo loại câu hỏi. Dùng `category` trong
  canonical record (MME có category; với GQA, nếu metadata nguồn có loại câu
  hỏi thì dùng, nếu không thì dùng 3 cờ từ khoá ở mục 4).
- Tương quan giữa `m*` với `N`, `ρ_m` và các cờ loại câu hỏi.
- Ví dụ minh hoạ mẫu được chọn L1 và mẫu được chọn L6/L7 (đưa vào failure gallery).

## 13. Rủi ro

| Rủi ro | Dấu hiệu | Giảm thiểu |
| --- | --- | --- |
| Lệch miền GQA → MMBench/MME | `r̄` thực tế lệch xa mục tiêu; Q3 âm | đặc trưng không phụ thuộc miền; báo cáo trung thực; ngưỡng DPP làm phương án không học |
| Nhãn mất cân bằng (phần lớn mẫu đúng ở mọi mức ≥ L3) | AUROC cao nhưng vô dụng ở L1–L2 | `w⁺_m`; báo cáo AUROC theo từng mức |
| Lợi ích nhỏ so với ngưỡng DPP | Q2 không có ý nghĩa thống kê | vẫn là kết luận có giá trị; khi đó chọn ngưỡng DPP |
| Nhãn không đơn điệu nhiều | > 10% mẫu | dùng biến thể 6 đầu ra độc lập; báo cáo tỉ lệ |

## 14. Vì sao không dùng RL

Hành động chỉ có 7 lựa chọn, và **phần thưởng của mọi hành động đều tính được
offline** (mục 7). Đây là contextual bandit với thông tin đầy đủ. Học có giám sát
trên nhãn đầy đủ là lời giải trực tiếp, ít phương sai, và cần ít mẫu hơn so với
học thử–sai.

## 15. Tiêu chí thành công

| Mốc | Tiêu chí |
| --- | --- |
| M6a | có 50k mẫu train và 5k mẫu val đã gán nhãn; báo cáo tỉ lệ nhãn không đơn điệu |
| M6b | AUROC ≥ 0,75 cho `p_1`, `p_2`, `p_3` trên GQA val |
| M6c | DSFold-A nằm trên đường nội suy của r cố định (CI không chứa 0) ở ≥ 2/3 benchmark, tại ít nhất một điểm vận hành |
| M6d | trả lời được Q2 (có hay không có ý nghĩa thống kê) |
