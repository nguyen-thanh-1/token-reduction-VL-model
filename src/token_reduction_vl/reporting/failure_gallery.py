"""Build a self-contained HTML gallery of benchmark prediction failures."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from token_reduction_vl.evaluation.metrics import score_prediction_record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    return rows


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def _category(record: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    for source in (record, _metadata(record), sample, _metadata(sample)):
        value = source.get("category")
        if value not in (None, ""):
            return str(value)
    types = _metadata(sample).get("types", {})
    if isinstance(types, Mapping):
        for key in ("detailed", "semantic", "structural"):
            if types.get(key) not in (None, ""):
                return str(types[key])
    return "uncategorized"


def _select_diverse(
    failures: Sequence[tuple[dict[str, Any], dict[str, Any]]], maximum: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select failures round-robin across categories in deterministic input order."""

    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    category_order: list[str] = []
    for record, sample in failures:
        category = _category(record, sample)
        if category not in groups:
            category_order.append(category)
        groups[category].append((record, sample))

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    offset = 0
    while len(selected) < maximum:
        added = False
        for category in category_order:
            rows = groups[category]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        offset += 1
    return selected


def _image_data_uri(path: Path, *, max_width: int, jpeg_quality: int) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
        if image.mode in {"RGBA", "LA"}:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _reference_text(record: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    answer_label = sample.get("answer_label") or _metadata(record).get("answer_label")
    answers = sample.get("answers") or record.get("references") or ()
    if isinstance(answers, str):
        answers = [answers]
    answer_text = " / ".join(str(value) for value in answers)
    if answer_label:
        return f"{answer_label} — {answer_text}" if answer_text else str(answer_label)
    return answer_text or "(no reference)"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def _render_html(
    *,
    title: str,
    model_id: str,
    cases: list[dict[str, Any]],
    images: dict[str, str],
    summaries: list[dict[str, Any]],
) -> str:
    cases_json = _safe_json(cases)
    images_json = _safe_json(images)
    summaries_json = _safe_json(summaries)
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — câu trả lời sai</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#d9dfeb;
      --panel:#fff; --bg:#f3f6fb; --blue:#2855d9; --red:#b42318; --green:#067647; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ background:linear-gradient(135deg,#172554,#243b77 55%,#3157b7); color:white; padding:34px 5vw 28px; }}
    header h1 {{ margin:0 0 8px; font-size:clamp(25px,3vw,40px); }}
    header p {{ margin:5px 0; color:#dbe6ff; }}
    main {{ width:min(1480px,94vw); margin:24px auto 60px; }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:18px; }}
    .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:15px 17px; box-shadow:0 5px 18px #243b7710; }}
    .stat strong {{ display:block; font-size:25px; margin-top:5px; }}
    .toolbar {{ position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:10px; align-items:center;
      padding:13px; margin-bottom:18px; background:#ffffffed; backdrop-filter:blur(10px); border:1px solid var(--line); border-radius:14px; }}
    button,input,select {{ font:inherit; border:1px solid #b9c3d6; border-radius:9px; background:white; padding:9px 12px; }}
    button {{ cursor:pointer; }} button.active {{ color:white; background:var(--blue); border-color:var(--blue); }}
    input {{ flex:1 1 290px; min-width:220px; }} select {{ min-width:190px; }}
    #count {{ margin-left:auto; color:var(--muted); font-size:14px; }}
    .gallery {{ display:grid; gap:18px; }}
    .case {{ display:grid; grid-template-columns:minmax(300px,44%) 1fr; overflow:hidden; background:var(--panel);
      border:1px solid var(--line); border-radius:16px; box-shadow:0 8px 28px #243b7712; }}
    .visual {{ min-height:320px; display:flex; align-items:center; justify-content:center; padding:14px; background:#e9edf5; }}
    .visual img {{ max-width:100%; max-height:520px; object-fit:contain; cursor:zoom-in; border-radius:9px; box-shadow:0 3px 14px #0002; }}
    .missing {{ color:var(--muted); text-align:center; padding:30px; }}
    .content {{ padding:20px 22px; min-width:0; }}
    .badges {{ display:flex; flex-wrap:wrap; gap:7px; margin-bottom:12px; }}
    .badge {{ padding:4px 8px; border-radius:999px; font-size:12px; font-weight:650; background:#eef2ff; color:#3447a5; }}
    .badge.fail {{ background:#fee4e2; color:var(--red); }}
    .question {{ white-space:pre-wrap; font-size:18px; line-height:1.48; margin:8px 0 16px; font-weight:650; }}
    .qa {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .answer {{ border-radius:11px; padding:12px 14px; border:1px solid; white-space:pre-wrap; overflow-wrap:anywhere; }}
    .answer small {{ display:block; font-weight:700; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px; }}
    .truth {{ background:#ecfdf3; border-color:#abefc6; color:#05603a; }}
    .prediction {{ background:#fff1f0; border-color:#fecdca; color:#912018; }}
    .choices {{ margin:14px 0; padding:0; list-style:none; display:grid; gap:6px; }}
    .choices li {{ padding:8px 10px; background:#f7f8fb; border-radius:8px; border:1px solid #e4e7ec; }}
    .choices li.correct {{ border-color:#6ce9a6; background:#edfcf2; }}
    details {{ margin-top:13px; border-top:1px solid var(--line); padding-top:11px; }}
    summary {{ cursor:pointer; color:#344054; font-weight:650; }}
    .context {{ white-space:pre-wrap; line-height:1.5; color:#475467; margin-top:10px; }}
    .meta {{ margin-top:14px; color:var(--muted); font-size:13px; display:flex; flex-wrap:wrap; gap:12px; }}
    dialog {{ width:min(94vw,1500px); max-height:94vh; border:0; border-radius:14px; padding:12px; background:#111827; }}
    dialog::backdrop {{ background:#0b1020dd; }} dialog img {{ display:block; max-width:100%; max-height:88vh; margin:auto; object-fit:contain; }}
    dialog button {{ position:absolute; right:18px; top:18px; background:#ffffffdd; }}
    .empty {{ text-align:center; padding:60px 20px; color:var(--muted); background:white; border-radius:14px; }}
    @media (max-width:850px) {{ .case {{ grid-template-columns:1fr; }} .visual {{ min-height:240px; }} .qa {{ grid-template-columns:1fr; }} #count {{ width:100%; margin-left:0; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Phân tích các câu trả lời sai</h1>
    <p>{title}</p><p>Model: <code>{model_id}</code> · Ảnh được nhúng trong file, có thể mở offline.</p>
  </header>
  <main>
    <section class="stats" id="stats"></section>
    <section class="toolbar">
      <button class="active" data-dataset="all">Tất cả</button>
      <span id="datasetButtons"></span>
      <input id="search" type="search" placeholder="Tìm câu hỏi, đáp án, category, sample ID…">
      <select id="category"><option value="all">Tất cả category</option></select>
      <span id="count"></span>
    </section>
    <section class="gallery" id="gallery"></section>
  </main>
  <dialog id="lightbox"><button type="button">Đóng ✕</button><img alt="Ảnh phóng to"></dialog>
  <script>
    const CASES={cases_json};
    const IMAGES={images_json};
    const SUMMARIES={summaries_json};
    const state={{dataset:'all',category:'all',query:''}};
    const esc=(value)=>String(value??'').replace(/[&<>"']/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[ch]));
    const stats=document.querySelector('#stats');
    stats.innerHTML=SUMMARIES.map(s=>`<div class="stat"><span>${{esc(s.label)}}</span><strong>${{s.selected}} / ${{s.wrong_total.toLocaleString()}}</strong><small>mẫu đang hiển thị / tổng số trả lời sai</small></div>`).join('');
    const datasetButtons=document.querySelector('#datasetButtons');
    datasetButtons.innerHTML=SUMMARIES.map(s=>`<button data-dataset="${{esc(s.dataset)}}">${{esc(s.dataset.toUpperCase())}}</button>`).join(' ');
    const category=document.querySelector('#category');
    [...new Set(CASES.map(c=>c.category))].sort().forEach(value=>category.insertAdjacentHTML('beforeend',`<option value="${{esc(value)}}">${{esc(value)}}</option>`));
    function choicesHTML(item) {{
      if (!item.choices.length) return '';
      return `<ul class="choices">${{item.choices.map(c=>`<li class="${{c.label===item.answer_label?'correct':''}}"><b>${{esc(c.label)}}.</b> ${{esc(c.text)}}</li>`).join('')}}</ul>`;
    }}
    function card(item) {{
      const image=IMAGES[item.image_key];
      const visual=image?`<img loading="lazy" src="${{image}}" alt="${{esc(item.image_id)}}">`:`<div class="missing">Không đọc được ảnh<br>${{esc(item.image_error||'')}}</div>`;
      const details=item.hint?`<details><summary>Ngữ cảnh / hint</summary><div class="context">${{esc(item.hint)}}</div></details>`:'';
      return `<article class="case"><div class="visual">${{visual}}</div><div class="content">
        <div class="badges"><span class="badge fail">SAI</span><span class="badge">${{esc(item.dataset.toUpperCase())}} · ${{esc(item.split)}}</span><span class="badge">${{esc(item.category)}}</span></div>
        <div class="question">${{esc(item.question)}}</div>${{choicesHTML(item)}}
        <div class="qa"><div class="answer truth"><small>Đáp án chuẩn</small>${{esc(item.reference)}}</div><div class="answer prediction"><small>Model trả lời</small>${{esc(item.prediction||'(rỗng)')}}</div></div>
        ${{details}}<div class="meta"><span>ID: ${{esc(item.sample_id)}}</span><span>Image: ${{esc(item.image_id)}}</span><span>Input: ${{item.input_tokens??'—'}} tokens</span><span>Output: ${{item.output_tokens??'—'}} tokens</span><span>Latency: ${{item.latency_seconds==null?'—':Number(item.latency_seconds).toFixed(4)+' s'}}</span></div>
      </div></article>`;
    }}
    function render() {{
      const query=state.query.toLocaleLowerCase();
      const rows=CASES.filter(item=>(state.dataset==='all'||item.dataset===state.dataset)&&(state.category==='all'||item.category===state.category)&&(!query||[item.sample_id,item.image_id,item.question,item.reference,item.prediction,item.category,item.hint].join(' ').toLocaleLowerCase().includes(query)));
      document.querySelector('#gallery').innerHTML=rows.length?rows.map(card).join(''):'<div class="empty">Không có mẫu phù hợp với bộ lọc.</div>';
      document.querySelector('#count').textContent=`Hiển thị ${{rows.length}} / ${{CASES.length}} mẫu`;
      document.querySelectorAll('.visual img').forEach(img=>img.addEventListener('click',()=>{{const dialog=document.querySelector('#lightbox');dialog.querySelector('img').src=img.src;dialog.showModal();}}));
    }}
    document.querySelectorAll('[data-dataset]').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('[data-dataset]').forEach(b=>b.classList.remove('active'));button.classList.add('active');state.dataset=button.dataset.dataset;render();}}));
    document.querySelector('#search').addEventListener('input',event=>{{state.query=event.target.value;render();}});
    category.addEventListener('change',event=>{{state.category=event.target.value;render();}});
    const lightbox=document.querySelector('#lightbox');lightbox.querySelector('button').addEventListener('click',()=>lightbox.close());lightbox.addEventListener('click',event=>{{if(event.target===lightbox)lightbox.close();}});
    render();
  </script>
</body>
</html>
"""


def build_failure_gallery(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Generate the configured standalone HTML failure gallery."""

    report = config["report"]
    gallery = config["failure_gallery"]
    predictions = config["predictions"]
    canonical_sources = gallery["canonical_sources"]
    maximum = int(gallery.get("max_cases_per_dataset", 12))
    max_width = int(gallery.get("image_max_width", 720))
    jpeg_quality = int(gallery.get("jpeg_quality", 82))
    if maximum < 1:
        raise ValueError("max_cases_per_dataset must be at least 1.")
    if max_width < 64:
        raise ValueError("image_max_width must be at least 64.")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be between 1 and 95.")

    destination = output_path or (
        _resolve(project_root, report["output_dir"])
        / str(gallery.get("filename", "failure_cases.html"))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    images: dict[str, str] = {}
    path_to_key: dict[Path, str] = {}
    summaries: list[dict[str, Any]] = []

    for dataset, prediction_source in predictions.items():
        if dataset not in canonical_sources:
            raise KeyError(f"Missing canonical source for {dataset!r}.")
        prediction_path = _resolve(project_root, prediction_source["path"])
        canonical_path = _resolve(project_root, canonical_sources[dataset])
        canonical_rows = _read_jsonl(canonical_path)
        canonical_by_id = {str(row["sample_id"]): row for row in canonical_rows}
        failures: list[tuple[dict[str, Any], dict[str, Any]]] = []
        missing_samples = 0
        for record in _read_jsonl(prediction_path):
            if score_prediction_record(record) is not False:
                continue
            sample = canonical_by_id.get(str(record.get("sample_id", "")))
            if sample is None:
                missing_samples += 1
                continue
            failures.append((record, sample))

        selected = _select_diverse(failures, maximum)
        summaries.append(
            {
                "dataset": dataset,
                "label": str(prediction_source.get("label", dataset)),
                "wrong_total": len(failures),
                "selected": len(selected),
                "missing_canonical_samples": missing_samples,
            }
        )
        for record, sample in selected:
            image_path = _resolve(project_root, str(sample["image"])).resolve()
            image_error: str | None = None
            image_key = ""
            if image_path in path_to_key:
                image_key = path_to_key[image_path]
            elif image_path.is_file():
                image_key = f"image-{len(images) + 1}"
                path_to_key[image_path] = image_key
                try:
                    images[image_key] = _image_data_uri(
                        image_path, max_width=max_width, jpeg_quality=jpeg_quality
                    )
                except OSError as exc:
                    image_error = str(exc)
                    image_key = ""
            else:
                image_error = f"Missing file: {image_path}"

            choices = sample.get("choices", ())
            if not isinstance(choices, list):
                choices = []
            cases.append(
                {
                    "dataset": dataset,
                    "split": str(sample.get("split", record.get("split", ""))),
                    "sample_id": str(record.get("sample_id", "")),
                    "image_id": str(sample.get("image_id", record.get("image_id", ""))),
                    "image_key": image_key,
                    "image_error": image_error,
                    "category": _category(record, sample),
                    "question": str(sample.get("question", record.get("question", ""))),
                    "hint": str(sample.get("hint") or ""),
                    "choices": choices,
                    "answer_label": str(sample.get("answer_label") or ""),
                    "reference": _reference_text(record, sample),
                    "prediction": str(record.get("prediction", "")),
                    "input_tokens": record.get("input_tokens"),
                    "output_tokens": record.get("output_tokens"),
                    "latency_seconds": record.get("latency_seconds"),
                }
            )

    html = _render_html(
        title=str(report["title"]),
        model_id=str(report["model_id"]),
        cases=cases,
        images=images,
        summaries=summaries,
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(destination)
    return {
        "path": destination,
        "cases": len(cases),
        "unique_images": len(images),
        "datasets": summaries,
    }
