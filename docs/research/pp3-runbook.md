# Runbook PP3 — bản 1 (lệnh chạy từng bước)

Ngày lập: 2026-11-03. Chạy trong PowerShell ở thư mục gốc repo, mỗi lệnh một
terminal. Mọi lệnh `run_diagnostics` dừng giữa chừng được và chạy lại sẽ tiếp
tục (dòng đã xong được bỏ qua).

**Quy tắc song song**

- Các lệnh trong cùng một **nhóm song song** chạy cùng lúc được (mỗi lệnh ghi
  run khác nhau). Mỗi tiến trình cần khoảng 4,3 GB VRAM; xem `nvidia-smi`.
- Một nhóm chỉ bắt đầu khi nhóm **phải chờ** của nó đã xong.
- Không chạy cùng lúc hai lệnh trùng `--source` + `--prefix` + tên trong `--only`.
- Số thời gian (selector / prefill) chỉ lấy từ bước 5, chạy một mình.

| Bước | Trạng thái |
|---|---|
| 1. Giai đoạn A (tune2k) | xong toàn bộ (1a–1e) |
| 2. Xác nhận (gqa_val_confirm) | xong (2a, 2b, 2c) |
| 3. Đánh giá cuối | xong 3-I, 3-II (3-III bỏ qua) |
| 4. Bảng cuối | xong (`outputs/reports/pp3_final/tables.md`) |
| 5. Đo thời gian | xong (5a, 5b) |

Cấu hình đã chốt (2026-11-03): **3d_attn50k có fold** (bỏ fold không khác
biệt: L2 −1.25 / L3 −0.25 / L4 +0.15 so với −1.20 / −0.20 / −0.20).

---

## Bước 1 — Giai đoạn A trên tune2k

### Nhóm 1-I (đã xong): 1a, 1b, 1c-L3, 1c-L4 — song song

**1a** — baseline DivPrune / FastV, L1, không fold

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --only 3d_attn50k_L1 random_L1 random_L2 divprune_L1 divprune_L2 divprune_L3 divprune_L4 fastv_L1 fastv_L2 fastv_L3 fastv_L4 3d_attn50k_nofold_L2 3d_attn50k_nofold_L3 3d_attn50k_nofold_L4 3d_attn50k_nofold_L5
```

**1b** — ablation 3c

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_3c.yaml --tag 3c --only 3c_L3 3c_L4
```

**1c-L3** — ablation DSFold-D

```powershell
uv run python scripts/common/run_pruned.py --method-config configs/methods/pp3_scorer_dsfold_D.yaml --source gqa_val_balanced --split configs/splits/gqa_val_tune_2k.txt --level L3 --variant dsfoldD --run-id tune2k_dsfoldD_L3_gqa
```

**1c-L4**

```powershell
uv run python scripts/common/run_pruned.py --method-config configs/methods/pp3_scorer_dsfold_D.yaml --source gqa_val_balanced --split configs/splits/gqa_val_tune_2k.txt --level L4 --variant dsfoldD --run-id tune2k_dsfoldD_L4_gqa
```

### Nhóm 1-II (đã xong): 1d — chờ nhóm 1-I

**1d-1** — paired report

```powershell
uv run python scripts/pp3/paired_report.py --runs random_L1 divprune_L1 fastv_L1 3d_attn50k_L1 random_L2 divprune_L2 fastv_L2 3d_attn50k_L2 3d_attn50k_nofold_L2 random_L3 divprune_L3 fastv_L3 dsfoldD_L3 3a_L3 3c_L3 3d_attn50k_L3 3d_attn50k_nofold_L3 random_L4 divprune_L4 fastv_L4 dsfoldD_L4 3a_L4 3c_L4 3d_attn50k_L4 3d_attn50k_nofold_L4 3d_attn50k_L5 3d_attn50k_nofold_L5 --pairs 3d_attn50k_L2:divprune_L2 3d_attn50k_L2:fastv_L2 3d_attn50k_L3:divprune_L3 3d_attn50k_L3:fastv_L3 --types
```

**1d-2** — accuracy theo dạng câu

```powershell
uv run python scripts/pp3/simulate_gate.py --candidates 3d_attn50k_L3 3d_attn50k_nofold_L3 3d_attn50k_L4 3d_attn50k_nofold_L4 3d_attn50k_L5 3d_attn50k_nofold_L5 --top 1
```

### Nhóm 1-III (đã xong): 1e-1, 1e-2, 1e-3 — song song

Ablation ở budget thấp: 3a và DSFold-D ở L1/L2.

**1e-1**

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer.yaml --tag 3a --only 3a_L1 3a_L2
```

**1e-2**

```powershell
uv run python scripts/common/run_pruned.py --method-config configs/methods/pp3_scorer_dsfold_D.yaml --source gqa_val_balanced --split configs/splits/gqa_val_tune_2k.txt --level L1 --variant dsfoldD --run-id tune2k_dsfoldD_L1_gqa
```

**1e-3**

```powershell
uv run python scripts/common/run_pruned.py --method-config configs/methods/pp3_scorer_dsfold_D.yaml --source gqa_val_balanced --split configs/splits/gqa_val_tune_2k.txt --level L2 --variant dsfoldD --run-id tune2k_dsfoldD_L2_gqa
```

### Nhóm 1-IV (đã xong): 1e-4 — chờ nhóm 1-III

**1e-4**

```powershell
uv run python scripts/pp3/paired_report.py --runs dsfoldD_L1 3a_L1 3d_attn50k_L1 dsfoldD_L2 3a_L2 3d_attn50k_L2 --pairs 3d_attn50k_L1:3a_L1 3d_attn50k_L2:3a_L2
```

---

## Bước 2 — Xác nhận trên gqa_val_confirm (4890 câu, không trùng tune2k)

### 2a (đã xong)

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --split configs/splits/gqa_val_confirm.txt --prefix confirm --only full 3d_attn50k_L2 3d_attn50k_L3 3d_attn50k_L4 divprune_L3 fastv_L3 random_L3
```

### 2b — thêm L1; song song được với nhóm 1-III

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --split configs/splits/gqa_val_confirm.txt --prefix confirm --only 3d_attn50k_L1 divprune_L1 fastv_L1 random_L1
```

### 2c — báo cáo; chờ 2b (dán kết quả cho Claude)

```powershell
uv run python scripts/pp3/paired_report.py --prefix confirm --runs random_L1 divprune_L1 fastv_L1 3d_attn50k_L1 3d_attn50k_L2 random_L3 divprune_L3 fastv_L3 3d_attn50k_L3 3d_attn50k_L4 --pairs 3d_attn50k_L1:divprune_L1 3d_attn50k_L3:divprune_L3 3d_attn50k_L3:fastv_L3
```

---

## Bước 3 — Đánh giá cuối (qua đêm)

### Nhóm 3-I: 3a-MME, 3a-MMB, 3a-GQA — song song

Có thể bắt đầu ngay, song song với bước 1e và 2b nếu còn VRAM.
Thiếu thời gian: bỏ `_L2`, `_L4` khỏi `--only`, chạy lại đủ sau.

**3a-MME** (2374 câu)

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --source mme_test --split all --prefix final --only full random_L1 random_L2 random_L3 random_L4 divprune_L1 divprune_L2 divprune_L3 divprune_L4 fastv_L1 fastv_L2 fastv_L3 fastv_L4 3d_attn50k_L1 3d_attn50k_L2 3d_attn50k_L3 3d_attn50k_L4
```

**3a-MMB** (4329 câu)

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --source mmb_validation --split all --prefix final --only full random_L1 random_L2 random_L3 random_L4 divprune_L1 divprune_L2 divprune_L3 divprune_L4 fastv_L1 fastv_L2 fastv_L3 fastv_L4 3d_attn50k_L1 3d_attn50k_L2 3d_attn50k_L3 3d_attn50k_L4
```

**3a-GQA** (12578 câu, lâu nhất)

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --source gqa_testdev_balanced --split all --prefix final --only full random_L1 random_L2 random_L3 random_L4 divprune_L1 divprune_L2 divprune_L3 divprune_L4 fastv_L1 fastv_L2 fastv_L3 fastv_L4 3d_attn50k_L1 3d_attn50k_L2 3d_attn50k_L3 3d_attn50k_L4
```

### Nhóm 3-II: 3b-MME, 3b-MMB, 3b-GQA — song song với nhau và với nhóm 3-I

PP3-fc (biến thể diễn giải được), chỉ ghi run `fcd25_L5`.

**3b-MME**

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_fc.yaml --tag fcd25 --fc beta=1,share_min=0.25,share_max=0.25,context_dpp=1 --source mme_test --split all --prefix final --only fcd25_L5
```

**3b-MMB**

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_fc.yaml --tag fcd25 --fc beta=1,share_min=0.25,share_max=0.25,context_dpp=1 --source mmb_validation --split all --prefix final --only fcd25_L5
```

**3b-GQA**

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_fc.yaml --tag fcd25 --fc beta=1,share_min=0.25,share_max=0.25,context_dpp=1 --source gqa_testdev_balanced --split all --prefix final --only fcd25_L5
```

### Nhóm 3-III (tùy chọn, ưu tiên thấp): 3c-MME, 3c-MMB, 3c-GQA — song song với 3-I, 3-II

Head 3a (nhãn gradient) trên tập test, để chứng minh hai nguồn nhãn đều thắng
baseline. Chỉ chạy nếu còn GPU sau 3-I và 3-II.

**3c-MME**

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer.yaml --tag 3a --source mme_test --split all --prefix final --only 3a_L1 3a_L2 3a_L3
```

**3c-MMB**

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer.yaml --tag 3a --source mmb_validation --split all --prefix final --only 3a_L1 3a_L2 3a_L3
```

**3c-GQA**

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer.yaml --tag 3a --source gqa_testdev_balanced --split all --prefix final --only 3a_L1 3a_L2 3a_L3
```

---

## Bước 4 — Bảng cuối; chờ toàn bộ bước 3 (dán kết quả cho Claude)

Hai lệnh song song được (chỉ đọc file).

**4-1** — chất lượng và hiệu năng

```powershell
uv run python scripts/pp3/final_report.py --runs random_L1 divprune_L1 fastv_L1 3d_attn50k_L1 random_L2 divprune_L2 fastv_L2 3d_attn50k_L2 random_L3 divprune_L3 fastv_L3 3d_attn50k_L3 random_L4 divprune_L4 fastv_L4 3d_attn50k_L4 fcd25_L5 --output outputs/reports/pp3_final/tables.md
```

**4-2** — chỉ số vùng (tune2k)

```powershell
uv run python scripts/pp3/region_report.py --runs random_L3 3d_attn50k_L3 random_L4 3d_attn50k_L4 fcd25_L4 fcd25_L5
```

---

## Bước 5 — Đo thời gian; chạy MỘT MÌNH (không có lệnh GPU nào khác)

**5a** — 300 câu tune2k (ảnh nhỏ), dùng cho bảng hiệu năng của bài báo.

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --prefix timing --limit 300 --only full random_L3 divprune_L1 divprune_L3 fastv_L1 fastv_L3 3d_attn50k_L1 3d_attn50k_L3 3d_attn50k_L4
```

**5b** — ảnh lớn (300 câu đầu của MME: code_reasoning + artwork, khoảng 1000 token ảnh); chạy sau 5a, cũng một mình

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --source mme_test --split all --prefix timing --limit 300 --only full random_L3 divprune_L1 divprune_L3 fastv_L1 fastv_L3 3d_attn50k_L1 3d_attn50k_L3 3d_attn50k_L4
```

Sau đó (dán kết quả cho Claude):

```powershell
uv run python scripts/pp3/final_report.py --prefix timing --benchmark gqa:gqa_val_balanced --benchmark mme:mme_test --runs random_L3 divprune_L1 divprune_L3 fastv_L1 fastv_L3 3d_attn50k_L1 3d_attn50k_L3 3d_attn50k_L4
```

---

## Bước 6 — Baseline CDPruner (sau phản biện, 04/11)

CDPruner (NeurIPS 2025) là công trình gần nhất: DPP có điều kiện theo câu hỏi,
điểm liên quan thủ công. Cài ở `baselines.py` (`cdpruner_L1..L5`, ADR 0005).
Mỗi câu chỉ có 4 run nên khoảng 25–30% thời gian của nhóm 3-I.

### Nhóm 6-I: 6a-GQA, 6a-MMB — song song

**6a-GQA** (khoảng 3 giờ)

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --source gqa_testdev_balanced --split all --prefix final --only full cdpruner_L1 cdpruner_L2 cdpruner_L3 cdpruner_L4
```

**6a-MMB** (khoảng 1 giờ)

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --source mmb_validation --split all --prefix final --only full cdpruner_L1 cdpruner_L2 cdpruner_L3 cdpruner_L4
```

`full` đã có đủ dòng nên được bỏ qua; nó chỉ có mặt để `run_meta` ghi đúng.

### Nhóm 6-II: 6a-MME (khoảng 1,5 giờ) — chờ 6a-MMB xong (ảnh landmark lớn)

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --source mme_test --split all --prefix final --only full cdpruner_L1 cdpruner_L2 cdpruner_L3 cdpruner_L4
```

### 6b — CDPruner trên tune2k (khoảng 15 phút, cho bảng ablation); song song với 6-I

```powershell
uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --only cdpruner_L1 cdpruner_L2 cdpruner_L3 cdpruner_L4
```

### 6c — bảng (dán kết quả cho Claude); chờ 6-I, 6-II

```powershell
uv run python scripts/pp3/final_report.py --runs cdpruner_L1 3d_attn50k_L1 cdpruner_L2 3d_attn50k_L2 cdpruner_L3 3d_attn50k_L3 cdpruner_L4 3d_attn50k_L4 --output outputs/reports/pp3_final/tables_cdpruner.md
```

```powershell
uv run python scripts/pp3/paired_report.py --prefix final --source gqa_testdev_balanced --dataset gqa --runs cdpruner_L1 cdpruner_L3 --pairs 3d_attn50k_L1:cdpruner_L1 3d_attn50k_L2:cdpruner_L2 3d_attn50k_L3:cdpruner_L3 3d_attn50k_L4:cdpruner_L4
```

---

## Tùy chọn — hình minh họa cho paper (bất cứ lúc nào)

Trước khi chạy, sửa `focus_context` trong `configs/methods/pp3_scorer_fc.yaml`
thành `share_min: 0.25`, `share_max: 0.25`, `context_dpp: 1`. Ảnh lưu ở
`data/demo/interactive/`.

```powershell
uv run python scripts/common/interactive_pruned.py --method-config configs/methods/pp3_scorer_fc.yaml --level L4 --level L5
```
