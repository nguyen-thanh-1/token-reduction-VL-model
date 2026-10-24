# Cách 2 — DSFold-A: mạng nhỏ chọn ngân sách K, công thức toán chọn token

- Trạng thái: Đề xuất
- Vai trò: **phần "adaptive" chính của khoá luận**
- Bao nhiêu token: **MLP nhỏ có huấn luyện** · Token nào: **DSFold (cách 1)**
- Điều kiện: cách 1 đã chạy ổn định (đã qua M1–M5)

## 1. Ý tưởng

Mỗi ảnh và câu hỏi cần một lượng thông tin khác nhau:
- *"Is it daytime?"* với ảnh đơn giản: vài chục token là đủ.
- Đọc chữ, đếm vật thể, hay câu hỏi về vị trí trên ảnh lớn: cần nhiều token hơn.

Cách 1 dùng một tỉ lệ r cho **mọi** mẫu, nên lãng phí ở mẫu dễ và thiếu ở mẫu
khó. Cách 2 thêm một **budget head**: một MLP nhỏ nhìn vào ảnh và câu hỏi,
rồi chọn **mức r riêng cho từng mẫu**. Việc chọn token nào vẫn do DSFold đảm
nhiệm.

## 2. Vị trí trên sơ đồ

Budget head là một ô mới **bên trong khối "2½. DSFold"**, nằm thay vào ô
"Budget". Nó nhận thống kê từ các ô đầu vào và từ S1 (Score), rồi xuất ra K
cho S2 (Select).

## 3. Không gian lựa chọn

Có 6 mức rời rạc, trùng với các mức cố định của cách 1:

```
R = {5,6%, 11,1%, 22,2%, 33,3%, 50%, 100%}
```

Dùng mức rời rạc thay vì một số thực vì: (1) tạo nhãn offline được cho từng
mức; (2) so sánh trực tiếp được với cách 1; (3) không cần RL.

## 4. Đặc trưng đầu vào của budget head

Mọi đặc trưng đều đã có sẵn sau ViT và sau S1 của DSFold, nên gần như không
tốn thêm chi phí.

| Nhóm | Đặc trưng | Kích thước |
| --- | --- | --- |
| Ảnh | `mean(v)` qua một lớp chiếu tuyến tính 2048 → 256 | 256 |
| Câu hỏi | `mean(q)` qua một lớp chiếu tuyến tính 2048 → 256 | 256 |
| Kích thước | `log N_visual`, tỉ lệ khung hình của lưới | 2 |
| Mức liên quan | entropy của `softmax(rel)`, tỉ lệ khối lượng của top-10% `rel` | 2 |
| DeepStack | mean và std của `ms` | 2 |
| Độ dư thừa | **đường cong DPP**: phần log-det tăng thêm (đã chuẩn hoá) tại 6 mức K | 6 |
| Câu hỏi | số token câu hỏi; có từ khoá đếm hoặc đọc chữ không ("how many", "what does … say") | 2 |

Đường cong DPP là đặc trưng quan trọng nhất. Nếu log-det bão hoà sớm, ảnh có
ít nội dung khác nhau, nên ít token là đủ.

## 5. Kiến trúc

```
x = concat(proj_v(mean v), proj_q(mean q), scalars)   # ~530 chiều
h = GELU(Linear(530 → 512)); h = GELU(Linear(512 → 256))
p = sigmoid(Linear(256 → 6))     # p_m = P(mức m vẫn đủ tốt)
```

Khoảng **1,2 triệu tham số**, chủ yếu nằm ở hai lớp chiếu 2048 → 256.

**Quy tắc chọn lúc suy luận:** chọn mức **nhỏ nhất** m sao cho `p_m ≥ θ`. Nếu
không có mức nào đạt thì dùng 100%. θ là núm điều chỉnh đánh đổi chất lượng –
hiệu năng. Quét θ cho ra cả một đường Pareto.

## 6. Tạo nhãn (offline, không cần RL)

1. Lấy **50.000 câu hỏi ngẫu nhiên cố định từ GQA train balanced**, ghi lại seed.
2. Với mỗi câu, chạy DSFold (cách 1, siêu tham số đã khoá) ở **cả 6 mức**.
3. Nhãn cho mức m: `y_m = 1` nếu câu trả lời ở mức m **trùng với câu trả lời
   full-token** (sau khi chuẩn hoá), ngược lại `y_m = 0`.
   - Dùng "trùng full-token" thay vì "đúng đáp án" vì mục tiêu là **giữ hành
     vi của baseline**, và nhãn này ít nhiễu hơn khi baseline sai.
   - Ablation: dùng nhãn "đúng đáp án".
4. Lưu kèm logit hoặc xác suất của token trả lời đầu tiên ở từng mức, để dùng
   cho biến thể hồi quy (mục 7).

**Chi phí ước tính:** 50k × 6 lượt × khoảng 0,14 s ≈ **12 giờ GPU**. Không
cần gradient qua model 2B. Toàn bộ đặc trưng (mục 4) được lưu cùng lúc với nhãn.

Vì sao không cần RL: chỉ có 6 hành động và ta **đánh giá được tất cả offline**.
Bài toán là một contextual bandit với đầy đủ thông tin, nên học có giám sát cho
lời giải tốt nhất với ít dữ liệu nhất.

## 7. Huấn luyện

- **Loss chính:** binary cross-entropy độc lập cho 6 đầu ra `p_m`.
- **Biến thể:** hồi quy độ tụt xác suất `Δ_m = p_full(ans) − p_m(ans)`, rồi
  chọn mức nhỏ nhất có `Δ̂_m ≤ ε`.
- **Tính đơn điệu:** khuyến khích `p_m` tăng theo m bằng một penalty
  `Σ max(0, p_m − p_{m+1})`, hoặc dự đoán theo kiểu cumulative logit.
- **Thiết lập:** AdamW, lr 1e-3, 20 epoch, batch 512. Chạy trên CPU hoặc GPU
  vài phút.
- **Chọn θ:** trên GQA val. Có thể chọn θ tại tỉ lệ giữ trung bình 22,2% để
  so sánh công bằng với mức cố định 22,2%.

## 8. Thí nghiệm

### So sánh chính: cùng tỉ lệ giữ trung bình thực tế

| Phương pháp | Loại |
| --- | --- |
| DSFold r cố định (cách 1) | không thích ứng |
| **DSFold ngưỡng DPP** (dừng khi phần log-det tăng thêm < ε) | thích ứng, không huấn luyện |
| ET-Prune, OccamToken (cài lại hoặc so với số liệu báo cáo) | thích ứng, không huấn luyện |
| **DSFold-A** | thích ứng, có huấn luyện |
| **Oracle** (biết trước mức nhỏ nhất đúng cho từng mẫu) | cận trên, không đạt được thực tế |

### Câu hỏi nghiên cứu

| | Câu hỏi | Cách trả lời |
| --- | --- | --- |
| Q1 | K thích ứng có tốt hơn K cố định ở cùng ngân sách trung bình không? | so đường Pareto |
| Q2 | Có cần học không, hay ngưỡng DPP là đủ? | DSFold-A so với ngưỡng DPP |
| Q3 | Học trên GQA có tổng quát hoá sang MMBench, MME không? | đánh giá ngoài miền |
| Q4 | Đặc trưng nào quan trọng? | ablation từng nhóm đặc trưng |
| Q5 | Còn cách oracle bao xa? | khoảng cách so với oracle |

### Phân tích

- Phân bố mức K được chọn theo benchmark và theo loại câu hỏi (GQA có nhãn
  loại câu hỏi; MME có category).
- Ví dụ minh hoạ mẫu được chọn ít token và mẫu được chọn nhiều token (dùng
  failure gallery).

## 9. Rủi ro

| Rủi ro | Giảm thiểu |
| --- | --- |
| Lệch miền: GQA khác MMBench/MME (trắc nghiệm, yes/no, ảnh lớn) | đặc trưng phải độc lập miền; báo cáo trung thực; có ngưỡng DPP làm phương án không học |
| Nhãn mất cân bằng (phần lớn mẫu đúng ở mọi mức) | dùng trọng số lớp; đánh giá bằng AUROC theo từng mức |
| Lợi ích nhỏ so với ngưỡng DPP | vẫn là kết quả đáng báo cáo (Q2); khi đó chọn ngưỡng DPP |
| Chi phí tạo nhãn | bắt đầu với 10k mẫu; tăng lên nếu đường học chưa bão hoà |

## 10. Đóng góp khoa học

- Budget head nhẹ (khoảng 1M tham số), học từ nhãn oracle offline, dùng tín
  hiệu DeepStack và đường cong DPP có sẵn, không cần RL và không cần gradient
  qua VLM.
- So sánh có hệ thống giữa ngân sách cố định, thích ứng không học, và thích
  ứng có học, trên cùng một bộ chọn token.

## 11. Mốc hoàn thành

| Mốc | Điều kiện |
| --- | --- |
| M6a | tạo xong nhãn và đặc trưng cho 50k mẫu GQA train |
| M6b | budget head có AUROC > 0,75 ở các mức ≤ 22,2% trên GQA val |
| M6c | DSFold-A thắng DSFold r cố định ở cùng ngân sách trung bình trên ≥ 2/3 benchmark |
