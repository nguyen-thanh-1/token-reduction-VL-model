# Cách 4 — Học toàn bộ: mạng quyết cả "bao nhiêu" lẫn "token nào"

- Trạng thái: Đề xuất (**hướng phát triển**; chỉ làm nếu cách 1–3 xong sớm)
- Bao nhiêu token: **học** · Token nào: **học**
- Điều kiện: cách huấn luyện A của cách 3 đã chạy được; GPU đủ VRAM

## 1. Ý tưởng

Một mạng duy nhất dự đoán cho mỗi visual token một **xác suất được giữ**
`π_i`. Tổng các xác suất chính là số token kỳ vọng:

```
E[K] = Σ_i π_i
```

Mạng được huấn luyện để vừa giữ chất lượng câu trả lời, vừa dùng ít token nhất
có thể. Như vậy K tự thích ứng theo từng mẫu, và token nào được giữ cũng do
mạng quyết định. Đây là khung **rate–utility** giống RUTA (2608.04132).

## 2. Vị trí trên sơ đồ

Mạng thay cả hai ô **"Budget" và "S1 (Score)"** trong khối "2½. DSFold".
- S2 trở thành "lấy mẫu hoặc ngưỡng hoá các gate".
- S3–S4 (Assign, Fold DeepStack) **giữ nguyên**. Đây là điểm khác chính so với RUTA.

## 3. Kiến trúc

- Thân mạng: giống cách 3c (1 lớp transformer nhẹ có cross-attention tới câu
  hỏi), khoảng 4M tham số.
- Hai đầu ra:
  - **gate head:** `π_i = sigmoid(s_i)` cho từng token;
  - **ngân sách ngầm định:** không cần đầu ra riêng, vì `E[K] = Σ π_i`.
- Tuỳ chọn: thêm một đầu ra dự đoán độ khó toàn cục để điều chỉnh nhiệt độ
  của các gate.

## 4. Huấn luyện

### 4.1 Gate khả vi

- **Lúc huấn luyện:** Bernoulli relaxation (Gumbel-sigmoid hoặc hard-concrete)
  cho `z_i ∈ {0, 1}`, kết hợp straight-through.
- **Đưa vào model:** như mặt nạ mềm ở cách 3. Cộng `log(z_i + ε)` vào
  attention logits của các key ở vị trí ảnh, và nhân DeepStack của token với
  `z_i`.
- **Lúc suy luận:** giữ token có `π_i ≥ 0,5`, hoặc giữ `⌈Σπ_i⌉` token có `π`
  cao nhất. Sau đó áp Fold DeepStack cho các token bị bỏ.

### 4.2 Hàm mục tiêu rate–utility

```
L = KL( p_full(ans) ‖ p_pruned(ans) )            # utility: giữ hành vi full-token
  + μ · CE(ans_gt)                               # tuỳ chọn: theo đáp án đúng
  + λ · (Σ_i π_i) / N_visual                     # rate: phạt tỉ lệ token dùng
  + η · ‖π − π_DPP‖                              # tuỳ chọn: neo theo lời giải DSFold
```

- Quét **λ** cho ra cả đường Pareto (mỗi λ cho một tỉ lệ giữ trung bình).
- Số hạng neo theo DPP giúp mạng khởi đầu từ lời giải của cách 1, nên ổn định
  hơn và ít overfit hơn.

### 4.3 Dữ liệu và chi phí (ước tính)

- 50k–100k mẫu GQA train, khởi tạo từ trọng số của cách 3 (nếu đã có).
- Backprop qua model 2B đóng băng: khoảng 10–16 GB VRAM với gradient
  checkpointing (cần đo).
- Khoảng **1–2 ngày** cho mỗi giá trị λ. Với 4–5 giá trị λ thì cần khoảng
  **3–5 ngày GPU**.

## 5. Vì sao không dùng RL (nhắc lại)

Ở đây gate có thể được làm khả vi bằng relaxation, nên gradient đi thẳng từ
loss tới mạng. Gradient này **ít nhiễu hơn nhiều** so với policy gradient của
RL.

RL chỉ cần thiết khi quyết định là một chuỗi nhiều bước không khả vi, ví dụ
model tự xin thêm độ phân giải trong lúc suy luận (VisionThink) hay chọn lấy lại
token trong lúc decode (ViMoD). Các hướng đó nằm ngoài phạm vi khoá luận này.

## 6. Thí nghiệm

| Câu hỏi | So sánh |
| --- | --- |
| Học toàn bộ có hơn kết hợp "budget head + DSFold" (cách 2) không? | cùng tỉ lệ giữ trung bình |
| Fold DeepStack có còn giúp khi đã học gate không? | có / không có S4 |
| Neo theo DPP có cần không? | η = 0 / η > 0 |
| Tổng quát hoá | GQA → MMBench, MME |
| So với RUTA | so với số liệu RUTA báo cáo trên Qwen3-VL-8B (khác kích thước model, ghi rõ) |

## 7. Rủi ro

| Rủi ro | Mức độ | Giảm thiểu |
| --- | --- | --- |
| Huấn luyện không ổn định (gate về toàn 0 hoặc toàn 1) | cao | khởi tạo từ cách 3; tăng λ dần theo lịch; neo theo DPP |
| Tốn GPU nhất trong 4 cách | cao | làm sau cùng; ít giá trị λ |
| Cạnh tranh trực tiếp với RUTA | cao | điểm khác: Fold DeepStack, tín hiệu DeepStack, model nhỏ 2B |
| Overfit GQA | trung bình | neo theo DPP; đánh giá ngoài miền |

## 8. Đóng góp nếu làm

- Khung rate–utility có **Fold DeepStack**: token bị bỏ vẫn đóng góp qua kênh
  DeepStack, khác RUTA (RUTA gộp vào các anchor token ở main stream).
- So sánh ba mức học trên cùng một khung, cùng dữ liệu và cùng đối chứng:
  không học (cách 1), học một phần (cách 2, 3), học toàn bộ (cách 4).

## 9. Khuyến nghị

Trong khoá luận, cách 4 nên được trình bày ở phần **hướng phát triển**, kèm
thiết kế ở trên. Chỉ triển khai nếu cách 1 và 2 đã có kết quả đầy đủ trước
thời hạn và còn ít nhất khoảng 3–4 tuần.
