"""Build outputs/pp3_scorer/reports/test10/index.html from the test10_* runs."""

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from token_reduction_vl.data.splits import iter_split_samples  # noqa: E402
from token_reduction_vl.reporting.pruning_report import (  # noqa: E402
    quality, read_jsonl, render_kept_tokens, score_prediction_record,
)

BENCH = [("gqa", "GQA testdev-balanced", "gqa_testdev_balanced"),
         ("mmb", "MMBench validation", "mmb_validation"),
         ("mme", "MME test", "mme_test")]
RUNS = [("Full tokens", "none", "test10_full_{ds}")] + [
    (f"{name} {lvl}", "pp3_scorer", f"test10_{tag}_{lvl}_{{ds}}")
    for lvl in ("L5", "L4", "L3", "L1") for name, tag in (("PP3 3a", "3a"), ("DSFold-D", "D"))
]
OUT = Path("outputs/pp3_scorer/reports/test10")


def load(method, run_id, source):
    path = Path(f"outputs/{method}/runs/{run_id}/{source}.jsonl")
    return {r["sample_id"]: r for r in read_jsonl(path)} if path.exists() else None


def mark(row):
    ok = score_prediction_record(row)
    return '<span class="ok">✓</span>' if ok else '<span class="bad">✗</span>'


CANONICAL = {"gqa": "gqa/testdev.jsonl", "mmb": "mmbench/validation.jsonl", "mme": "mme/test.jsonl"}
THUMB = 360


def save_jpg(image, name):
    image = image.copy()
    image.thumbnail((THUMB, THUMB * 2))
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    image.save(OUT / "img" / name, quality=82)
    return f"img/{name}"


def gallery(ds, title, source, ids, runs):
    """One card per sample: original image, then each run's kept-token overlay with K/N and answer."""

    samples = {s.sample_id: s for s in iter_split_samples(
        Path("data/processed/v1") / CANONICAL[ds], ids, project_root=Path("."))}
    cards = []
    for number, sid in enumerate(ids, 1):
        sample, full = samples[sid], runs[0][1][sid]
        image = sample.image if not callable(getattr(sample, "image", None)) else sample.image()
        image = image.convert("RGB")
        refs = ", ".join(full.get("references") or [])
        tiles = [f"""<figure class=tile><img loading=lazy src="{save_jpg(image, f'{ds}_{number:02d}_orig.jpg')}" alt="original">
<figcaption><b>Original image</b><br>{image.width}×{image.height} px → N = {full['N']} visual tokens</figcaption></figure>"""]
        for label, rows in runs:
            row = rows[sid]
            k, n = row.get("K", row["N"]), row["N"]
            keep = row.get("keep_indices")
            if keep is None:
                keep = range(n)
            overlay = render_kept_tokens(image, tuple(row["grid_hw"]), keep)
            tag = label.replace(" ", "_").replace("-", "")
            src = save_jpg(overlay, f"{ds}_{number:02d}_{tag}.jpg")
            name = "Baseline (full tokens)" if label == "Full tokens" else label
            tiles.append(f"""<figure class=tile><img loading=lazy src="{src}" alt="{html.escape(label)}">
<figcaption><b>{html.escape(name)}</b><br>{k}/{n} tokens ({100 * k / n:.1f}%) · LLM input {row['input_tokens']}<br>
{mark(row)} <span class=ans>{html.escape(row['prediction'])}</span></figcaption></figure>""")
        cards.append(f"""<article class=card><h3>#{number} <span class=sid>{html.escape(sid)}</span></h3>
<pre class=prompt>{html.escape(sample.question)}</pre>
<p><b>Reference:</b> {html.escape(refs)}</p><div class=tiles>{''.join(tiles)}</div></article>""")
    return f"<section><h2>{title} — kept tokens per sample</h2><p class=note>Green outline = token kept, dark = dropped. One cell = one merged visual token (2×2 patches of 16 px).</p>{''.join(cards)}</section>"


sections, galleries, meta = [], [], None
for ds, title, source in BENCH:
    runs = [(label, load(m, rid.format(ds=ds), source)) for label, m, rid in RUNS]
    runs = [(label, rows) for label, rows in runs if rows]
    if not runs:
        continue
    full = runs[0][1]
    ids = list(full)
    base_rows = {r["sample_id"]: r for r in read_jsonl(Path(f"outputs/baseline/{source}.jsonl"))}
    ref_acc = quality(full[ids[0]]["dataset"], [base_rows[i] for i in ids if i in base_rows])["accuracy"]
    body = []
    for label, rows in runs:
        rs = [rows[i] for i in ids]
        q = quality(rs[0]["dataset"], rs)
        correct = sum(bool(score_prediction_record(r)) for r in rs)
        delta = correct - sum(bool(score_prediction_record(full[i])) for i in ids)
        mme = f"{q['mme_score']:.1f}" if q.get("mme_score") is not None else "—"
        k = sum(r.get("K", 0) for r in rs) / len(rs)
        n = sum(r.get("N", 0) for r in rs) / len(rs)
        tok = sum(r["input_tokens"] for r in rs) / len(rs)
        lat = sum(r["latency_seconds"] for r in rs[1:]) / max(1, len(rs) - 1)
        cls = "pos" if delta > 0 else "neg" if delta < 0 else ""
        body.append(
            f"<tr><td>{label}</td><td class=num>{correct}/{len(rs)}</td><td class=num>{q['accuracy']:.2f}</td>"
            f"<td class='num {cls}'>{delta:+d}</td><td class=num>{mme}</td><td class=num>{k:.0f}/{n:.0f}</td>"
            f"<td class=num>{tok:.0f}</td><td class=num>{lat:.3f}</td></tr>")
    detail_head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in runs)
    detail = []
    for i in ids:
        r0 = full[i]
        cells = "".join(f"<td>{mark(rows[i])} {html.escape(rows[i]['prediction'][:40])}</td>" for _, rows in runs)
        question = html.escape(r0.get("question", "")[:160])
        refs = html.escape(", ".join(r0.get("references") or []))
        detail.append(f"<tr><td class=q title='{html.escape(i)}'>{question}</td><td>{refs}</td>{cells}</tr>")
    galleries.append(gallery(ds, title, source, ids, runs))
    sections.append(f"""
<section><h2>{title} — accuracy</h2>
<p class=note>Full-token accuracy on these {len(ids)} ids in the existing baseline run: {ref_acc:.2f} (sanity check for the fresh full-token run).</p>
<div class=scroll><table><thead><tr><th>Run</th><th>Correct</th><th>Acc</th><th>Δ vs full</th><th>MME score</th><th>K/N</th><th>Input tok</th><th>Latency s*</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<details><summary>Per-sample answers</summary><div class=scroll><table class=detail><thead><tr><th>Question</th><th>Reference</th>{detail_head}</tr></thead>
<tbody>{''.join(detail)}</tbody></table></div></details></section>""")
    meta = meta or json.loads(Path(f"outputs/pp3_scorer/runs/test10_3a_L3_{ds}/run_meta.json").read_text(encoding="utf-8"))

page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>PP3 Test10 Comparison</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e8;--head:#f5f5f7;--ok:#1a7f37;--bad:#c62828}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16161a;--fg:#ececf1;--muted:#9a9aa2;--line:#2e2e35;--head:#1f1f25;--ok:#4cc26d;--bad:#ff6b6b}}}}
:root[data-theme=dark]{{--bg:#16161a;--fg:#ececf1;--muted:#9a9aa2;--line:#2e2e35;--head:#1f1f25;--ok:#4cc26d;--bad:#ff6b6b}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 6px}}
.note,.meta{{color:var(--muted);font-size:13px}} .scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;margin:8px 0}} th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}}
th{{background:var(--head);font-weight:600;font-size:13px}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.pos{{color:var(--ok)}} .neg{{color:var(--bad)}} .ok{{color:var(--ok);font-weight:700}} .bad{{color:var(--bad);font-weight:700}}
.detail td{{font-size:13px}} .q{{max-width:320px}} summary{{cursor:pointer;color:var(--muted);margin:6px 0}}
.card{{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}} .card h3{{margin:0 0 6px;font-size:16px}}
.sid{{color:var(--muted);font-weight:400;font-size:12px}} .prompt{{white-space:pre-wrap;background:var(--head);padding:8px 10px;border-radius:6px;font:13px/1.45 ui-monospace,monospace;margin:6px 0}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}} .tile{{margin:0}}
.tile img{{width:100%;height:auto;border-radius:4px;display:block}} .tile figcaption{{font-size:12.5px;margin-top:4px}} .ans{{font-family:ui-monospace,monospace}}
.warn{{border-left:3px solid var(--bad);padding:8px 12px;background:var(--head);font-size:14px}}
</style></head><body>
<h1>PP3 score head vs DSFold-D vs full tokens — 10 held-out samples per benchmark</h1>
<p class=meta>Qwen3-VL-2B-Instruct · checkpoint {meta['checkpoint']} · DSFold params {meta['dsfold_params']} · git {meta['git_sha'][:7]} · {meta['gpu']} · torch {meta['torch']} · transformers {meta['transformers']}<br>
Samples: random seed 0 from GQA testdev-balanced, MMBench validation and MME test, disjoint from the dev_1k splits (MME: 5 images × 2 questions).</p>
<p class=warn>Smoke comparison only. With 10 samples one answer moves accuracy by 0.10, so these numbers cannot show a difference from the full-token model; use the dev_1k paired results for conclusions. Accuracies of the three benchmarks are not averaged; MME score is not a percentage.</p>
{''.join(sections)}
{''.join(galleries)}
<p class=note>* Mean latency excludes the first sample of each run (GPU warm-up). Δ vs full = correct answers minus the full-token run's correct answers on the same ids.</p>
</body></html>"""
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "index.html").write_text(page, encoding="utf-8")
print(OUT / "index.html")
