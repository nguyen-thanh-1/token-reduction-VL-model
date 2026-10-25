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
           · thành viên PP1: bộ D (configs/dsfold_D-1.0.yaml) + bộ kiểm tra "cài đúng PP1"
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
| Code | mỗi phương pháp **một folder riêng** trong `src/token_reduction_vl/pruning/` (mục 8). Chỉ sửa folder của mình; **không import** code của phương pháp khác |
| Nhánh git | `feat/pp<N>-<tên>` tách từ `dev`; phần chung trên `feat/common-<tên>` |
| Config | mỗi phương pháp có `configs/methods/pp<N>_<tên>.yaml` riêng (mục 8.3) |
| Kết quả | mỗi phương pháp ghi vào folder riêng `outputs/pp<N>_<tên>/` (mục 9). Mỗi lần chạy có `run_meta.json`, ghi phiên bản bộ D, git SHA, phiên bản danh sách dữ liệu |
| Dữ liệu huấn luyện | dùng chung **một** bộ danh sách trong `configs/splits/` (mục 9.3) |
| Báo cáo baseline | không ai được ghi đè `outputs/baseline/` và `outputs/reports/baseline/` |

## 8. Cấu trúc code

### 8.1 Cây thư mục

```
src/token_reduction_vl/
├── pruning/
│   ├── base.py                    ← CHUNG: PrunerInput, PrunerOutput, VisualTokenPruner (00-common mục 4.0)
│   ├── baselines/                 ← CHUNG: các đối chứng (00-common mục 9)
│   │   ├── random_k.py   grid_k.py   pool.py
│   │   ├── divprune.py   cdpruner.py fastv.py
│   │   └── resolution_p0.py       (chỉ đổi max_pixels/min_pixels, không cắt token)
│   │
│   ├── pp1_dsfold/                ← thành viên PP1
│   │   ├── scoring.py             S1: q, rel, ms, quality
│   │   ├── dpp.py                 S2: greedy DPP + max–min (N > N_cap)
│   │   ├── fold.py                S3 Assign + S4 Fold
│   │   └── pruner.py              DSFoldPruner(VisualTokenPruner)
│   │
│   ├── pp2_budget/                ← thành viên PP2
│   │   ├── dsfold_base/           bản PP1 do PP2 tự cài (scoring.py, dpp.py, fold.py)
│   │   ├── features.py            403 đặc trưng
│   │   ├── model.py               budget head (cumulative link)
│   │   ├── train.py
│   │   └── pruner.py              BudgetDSFoldPruner
│   │
│   ├── pp3_scorer/                ← thành viên PP3
│   │   ├── dsfold_base/           bản PP1 do PP3 tự cài
│   │   ├── features.py            u_i (467 chiều)
│   │   ├── saliency.py            nhãn gradient × input
│   │   ├── models.py              3a MLP, 3b CNN, 3c transformer
│   │   ├── train.py               ListNet (tuyến B), chưng cất (tuyến A)
│   │   └── pruner.py              ScoredDSFoldPruner
│   │
│   └── pp4_gate/                  ← thành viên PP4
│       ├── dsfold_base/           bản PP1 do PP4 tự cài
│       ├── features.py
│       ├── hard_concrete.py
│       ├── model.py               mạng cổng
│       ├── losses.py              KL + rate + chống trùng lặp + neo DPP
│       ├── attention_bias.py      custom 4D mask (spike)
│       ├── train.py
│       └── pruner.py              GatedDSFoldPruner
│
├── models/
│   ├── qwen3_vl.py                (baseline có sẵn, không sửa)
│   └── qwen3_vl_pruned.py         ← CHUNG: wrapper (00-common mục 4.2)
│
└── evaluation/
    ├── metrics.py  runner.py      (có sẵn)
    ├── agreement.py               ← CHUNG: chỉ số agree (00-common mục 8.2)
    ├── bootstrap.py               ← CHUNG: paired cluster bootstrap (mục 8.4)
    ├── timing.py                  ← CHUNG: đo ViT / bộ chọn / prefill / decode
    └── run_meta.py                ← CHUNG: ghi run_meta.json (mục 9.2)

scripts/
├── common/   run_pruned.py  run_controls.py  make_splits.py  make_conformance_fixtures.py
├── pp1/      tune_dsfold.py
├── pp2/      make_budget_labels.py  train_budget_head.py  pick_thresholds.py
├── pp3/      make_saliency_labels.py  train_score_head.py
└── pp4/      spike_attention_bias.py  make_teacher_logits.py  train_gates.py

tests/
├── common/   test_wrapper_noop.py  test_bootstrap.py  test_interface.py
├── test_pp1_conformance.py        ← CHUNG: chạy cho cả 4 bản PP1 (pp1_dsfold + 3 dsfold_base)
├── pp1/  pp2/  pp3/  pp4/         ← unit test riêng (tensor giả, không cần model)
└── fixtures/pp1_conformance/reference.json   (kết quả tham chiếu, nhỏ, theo dõi bằng git)
```

### 8.2 Quyền sửa

| Vùng | Ai được sửa |
| --- | --- |
| `pruning/base.py`, `pruning/baselines/`, `models/qwen3_vl_pruned.py`, `evaluation/{agreement,bootstrap,timing,run_meta}.py`, `scripts/common/`, `tests/common/`, `tests/test_pp1_conformance.py` | **cả nhóm đồng ý** (đây là code của "luật chơi" `00-common`) |
| `pruning/pp<N>_*/`, `scripts/pp<N>/`, `tests/pp<N>/`, `configs/methods/pp<N>_*.yaml` | **chỉ thành viên PP<N>** |

Có **4 bản PP1** (`pp1_dsfold/` và 3 folder `dsfold_base/`). Việc lặp code là
**cố ý**, để mỗi người làm độc lập. `tests/test_pp1_conformance.py` chạy trên
cả 4 bản, đảm bảo chúng cho cùng kết quả.

### 8.3 Config

```
configs/
├── baseline_qwen3_vl.yaml  data_v1.yaml  report_baseline.yaml   (có sẵn, không sửa)
├── dsfold_D-1.0.yaml           ← CHUNG: bộ tham số D của PP1 (PP1 mục 8.0); đổi thì tạo file D-1.1 mới
├── controls.yaml               ← CHUNG: cấu hình các đối chứng
├── splits/                     ← CHUNG: danh sách mẫu (mục 9.3)
└── methods/
    ├── pp1_dsfold.yaml         trỏ tới dsfold_D-1.0.yaml (hoặc D-tuned)
    ├── pp2_budget.yaml         θ₁..θ₃, đường dẫn checkpoint, phiên bản D
    ├── pp3_scorer.yaml         kiến trúc, đường dẫn checkpoint, phiên bản D
    └── pp4_gate.yaml           λ, lịch huấn luyện, phiên bản D
```

## 9. Cấu trúc lưu kết quả

### 9.1 Phân biệt `outputs/` và `data/`

`.gitignore` hiện tại **theo dõi `outputs/`** (trừ checkpoint, `*.pt`,
`*.safetensors`, `smoke_*`) và **bỏ qua `data/`**. Vì vậy:

| Loại | Để ở | Git | Ví dụ |
| --- | --- | --- | --- |
| Prediction, metric, thời gian, báo cáo, hình | `outputs/` | **có**: cả nhóm xem được | `outputs/pp2_budget/runs/…/gqa_testdev.jsonl` |
| Checkpoint mạng nhỏ | `outputs/pp<N>_*/checkpoints/` | **không** (đã bị ignore) | `budget_head_seed0.pt` |
| Nhãn, đặc trưng, teacher logits, tensor lớn | `data/features/pp<N>_*/` | **không** | `labels_gqa_train_50k.npz` |
| Tensor kiểm tra PP1 | `data/features/pp1_conformance/` | **không** (quá lớn) | `sample_000.npz` |
| Danh sách mẫu (chỉ là ID) | `configs/splits/` | **có** | `gqa_train_50k.txt` |

Lưu ý: file lớn hơn khoảng 10 MB **không** được để trong `outputs/`. Nếu cần chia
sẻ file lớn trong nhóm thì dùng ổ chung, không đưa lên Git.

### 9.2 Cây thư mục kết quả

```
outputs/
├── baseline/                       (có sẵn, KHÔNG ghi đè)
├── reports/baseline/               (có sẵn, KHÔNG ghi đè)
│
├── controls/                       ← CHUNG: chạy một lần, mọi người dùng
│   └── <control>/<level>/<split>/  ví dụ: divprune/L3/dev_1k/  ·  random_k_s0/L1/full/  ·  p0/L3/full/
│       ├── gqa_testdev.jsonl  mmb_validation.jsonl  mme_test.jsonl
│       ├── metrics.json  timing.json  run_meta.json
│
├── pp1_dsfold/
│   ├── tuning/                     log 32 lần chạy chỉnh tham số → bộ D-tuned
│   ├── runs/<run_id>/              (cấu trúc file giống controls)
│   ├── ablations/<A1..A6>/<run_id>/
│   └── reports/<report_id>/        report.md, figures/, tables/, summary.json, manifest.json
│
├── pp2_budget/
│   ├── runs/<run_id>/  reports/<report_id>/
│   ├── training/                   log huấn luyện, AUROC, chọn θ (json, png)
│   └── checkpoints/                (không lên Git)
│
├── pp3_scorer/
│   ├── runs/<run_id>/  reports/<report_id>/
│   ├── training/
│   └── checkpoints/                (không lên Git)
│
├── pp4_gate/
│   ├── runs/<run_id>/  reports/<report_id>/
│   ├── spike/                      kết quả spike 4D mask
│   ├── training/                   log theo λ: E[K]/N, KL, histogram π
│   └── checkpoints/                (không lên Git)
│
└── comparison/                     ← CHUNG: so sánh cuối cùng 4 phương pháp
    └── <report_id>/                Pareto plots, bảng chọn phương pháp, input cho ADR 0002

data/features/                      (KHÔNG lên Git)
├── pp1_conformance/                tensor v, f của 100 mẫu kiểm tra
├── pp2_budget/                     labels_gqa_train_50k.npz, labels_gqa_val_5k.npz
├── pp3_scorer/                     saliency_gqa_train_50k.npz
└── pp4_gate/                       teacher_gqa_train_50k.npz
```

### 9.3 Quy ước đặt tên

| Đối tượng | Quy ước | Ví dụ |
| --- | --- | --- |
| `<level>` | `L1` … `L7` cho mức cố định; `adaptive` cho PP2/PP4 | `L3`, `adaptive` |
| `<split>` | `dev_1k` hoặc `full` | |
| `<run_id>` | `<YYYYMMDD>_<method>_<variant>_<level>_<split>` | `20261102_pp1_D-1.0_L3_dev_1k` · `20261110_pp2_theta2_adaptive_full` · `20261115_pp4_lambda1_adaptive_full` |
| `<report_id>` | `<method>-<variant>` | `pp1-dsfold-D-tuned`, `pp3-scorer-3b-routeB` |
| File danh sách mẫu | `configs/splits/<tên>.txt`, mỗi dòng một `sample_id`, dòng đầu là comment ghi seed | `gqa_train_50k.txt`, `gqa_val_5k.txt`, `gqa_val_tune_2k.txt`, `dev_1k_gqa.txt`, `dev_1k_mmb.txt`, `dev_1k_mme.txt`, `pp1_conformance_100.txt` |

### 9.4 Nội dung bắt buộc

**Mỗi dòng prediction JSONL** giữ mọi trường của baseline, và thêm các trường sau:

```json
{
  "sample_id": "...", "prediction": "...", "references": [...],
  "method": "pp2_budget", "variant": "theta2", "level": "adaptive",
  "ratio_target": null, "N": 260, "K": 58, "keep_ratio": 0.2231,
  "input_tokens": 84, "output_tokens": 2,
  "latency_seconds": 0.11, "timing": {"vit": 0.05, "selector": 0.002, "prefill": 0.03, "decode": 0.028},
  "peak_vram_mb": 3900.0, "pruning_method": "pp2_budget"
}
```

**`run_meta.json`** cho mỗi lần chạy:

```json
{
  "run_id": "20261110_pp2_theta2_adaptive_full",
  "method": "pp2_budget", "variant": "theta2",
  "config": "configs/methods/pp2_budget.yaml", "dsfold_params": "D-1.0",
  "git_sha": "...", "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
  "transformers": "5.x.y", "torch": "2.14.1",
  "gpu": "...", "driver": "...", "seed": 0,
  "splits": {"gqa": "full", "mmb": "full", "mme": "full"},
  "checkpoint": "outputs/pp2_budget/checkpoints/budget_head_seed0.pt",
  "started_at": "...", "finished_at": "..."
}
```

**`metrics.json`**: kết quả của `scripts/evaluate_predictions.py` cho từng
benchmark, cộng `agree`, `keep_ratio` trung bình, `r̄`, và với MME thì chia theo
3 nhóm kích thước ảnh (`00-common` mục 5.1).
