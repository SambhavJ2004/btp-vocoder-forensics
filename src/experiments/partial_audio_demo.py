"""Small offline review artifact with original sources, null controls and timelines."""

from __future__ import annotations

import csv
import html
import io
import json
import shutil
from pathlib import Path

from vocoders.checkpoints import sha256_file


def preview_records(rows: list[dict], sources: dict, count: int) -> list[dict]:
    by_id = {row["example_id"]: row for row in rows}
    mixed = [row for row in rows if row["has_synthetic"]]
    chosen = []
    for condition in sorted({row["source_condition"] for row in mixed}):
        chosen.append(next(row for row in mixed if row["source_condition"] == condition))
    # Prefer additional utterances before additional views of an already shown group.
    groups = {row["group_id"] for row in chosen}
    remainder = [row for row in mixed if row not in chosen]
    remainder.sort(key=lambda row: (row["group_id"] in groups, row["group_id"], row["example_id"]))
    chosen = (chosen + remainder)[:count]
    examples = []
    for row in chosen:
        control = by_id[row["matched_control_id"]]
        example = {
            "example_id": row["example_id"], "utt_id": row["group_id"] , "split": row["split"],
            "vocoder": row["source_condition"], "num_samples": row["num_samples"],
            "sample_rate": row["sample_rate"], "start_sample": row["start_sample"],
            "end_sample": row["end_sample"], "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "fade_samples": row["crossfade_samples_per_boundary"],
            "mixed_file": f"audio/{row['example_id']}.wav",
            "control_file": f"audio/{control['example_id']}.wav",
            "real_file": f"sources/{row['source_real_id']}.wav",
            "donor_file": f"sources/{row['source_donor_id']}.wav",
            "file_sha256": {
                f"audio/{row['example_id']}.wav": row["generated_sha256"],
                f"audio/{control['example_id']}.wav": control["generated_sha256"],
                f"sources/{row['source_real_id']}.wav": sources[row["source_real_id"]]["sha256"],
                f"sources/{row['source_donor_id']}.wav": sources[row["source_donor_id"]]["sha256"],
            }, "development_only": True, "citable": False,
        }
        if row.get("matched_real_to_real_control_id") is not None:
            real_control = by_id[row["matched_real_to_real_control_id"]]
            relative = f"audio/{real_control['example_id']}.wav"
            example.update(real_to_real_file=relative,
                           real_donor_start_sample=real_control["donor_start_sample"],
                           real_donor_end_sample=real_control["donor_end_sample"])
            example["file_sha256"][relative] = real_control["generated_sha256"]
        examples.append(example)
    return examples


def timeline_svg(examples: list[dict]) -> str:
    width, left = 900, 40
    review = any("real_to_real_file" in example for example in examples)
    spacing = 150 if review else 100
    height = 95 + spacing * len(examples)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}" '
             f'viewBox="0 0 980 {height}" role="img" aria-label="Ground-truth edit timelines">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#17202a">',
             '<text x="40" y="25">VocoPatch ground truth — development only, non-citable</text>',
             '<text x="40" y="48">Blue: real • red: synthetic • gold: transition/ignore</text>']
    for i, example in enumerate(examples):
        y = 82 + i * spacing
        start, end, fade, n = (example[key] for key in (
            "start_sample", "end_sample", "fade_samples", "num_samples",
        ))
        def x(sample, total=n):
            return left + width * sample / total
        title = html.escape(f"{example['utt_id']} · {example['vocoder']} · {example['split']}")
        parts.extend([
            f'<text x="40" y="{y}">{title}</text>',
            f'<rect x="40" y="{y + 10}" width="900" height="22" fill="#3877ac"/>',
            f'<rect x="{x(start + fade):.9f}" y="{y + 10}" '
            f'width="{width * (end - start - 2 * fade) / n:.9f}" height="22" fill="#b33445"/>',
        ])
        for a, b in ((start, start + fade), (end - fade, end)):
            if a < b:
                parts.append(f'<rect x="{x(a):.9f}" y="{y + 10}" '
                             f'width="{width * (b - a) / n:.9f}" height="22" fill="#e8ac36"/>')
        text = (f"edit [{start}, {end}) samples = "
                f"[{example['start_seconds']:.6f}, {example['end_seconds']:.6f}) s; "
                f"{fade} ignored samples at each boundary; total {n} samples")
        parts.append(f'<text x="40" y="{y + 55}">{html.escape(text)}</text>')
        if "real_to_real_file" in example:
            parts.extend([
                f'<rect x="40" y="{y + 68}" width="900" height="14" fill="#3877ac"/>',
                f'<rect x="{x(start + fade):.9f}" y="{y + 68}" '
                f'width="{width * (end - start - 2 * fade) / n:.9f}" '
                'height="14" fill="#2d8059"/>',
            ])
            for a, b in ((start, start + fade), (end - fade, end)):
                if a < b:
                    parts.append(f'<rect x="{x(a):.9f}" y="{y + 68}" '
                                 f'width="{width * (b-a) / n:.9f}" height="14" fill="#e8ac36"/>')
            origin = (f"Real-only control: green = relocated real, label 0; donor "
                      f"[{example['real_donor_start_sample']}, {example['real_donor_end_sample']}) "
                      "samples; gold = ignore")
            parts.append(f'<text x="40" y="{y + 105}">{html.escape(origin)}</text>')
    parts.append('</g></svg>\n')
    return "\n".join(parts)


def timestamps_csv(examples: list[dict]) -> str:
    stream = io.StringIO(newline="")
    fields = ["example_id", "utt_id", "split", "vocoder", "start_sample", "end_sample",
              "start_seconds", "end_seconds", "fade_samples", "sample_rate", "num_samples",
              "mixed_file", "control_file", "real_file", "donor_file",
              "development_only", "citable"]
    if any("real_to_real_file" in example for example in examples):
        fields += ["real_to_real_file", "real_donor_start_sample", "real_donor_end_sample"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(examples)
    return stream.getvalue()


def _index(examples: list[dict]) -> str:
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>VocoPatch local review</title><style>body{font:17px sans-serif;'
             'max-width:1050px;margin:35px auto;padding:0 18px;color:#17202a}'
             'article{border-top:1px solid #aaa;padding:20px 0}audio{width:100%}'
             '.audio{display:grid;grid-template-columns:1fr 1fr;gap:15px}'
             'img{max-width:100%;height:auto}code{font-size:14px}</style>',
             '<h1>VocoPatch partial-audio pilot</h1><p><strong>Development only — non-citable. '
             'These are ground-truth edits, not model predictions.</strong></p>',
             '<p>The same selected real-reference region is replaced by its paired vocoder '
             'samples. Complementary fades stay inside the edit. Gold buffers are ignored '
             'in labels. Real-only controls use the same editor with real as its own donor; '
             'they do not establish freedom from real-to-synthetic splice artifacts.</p>',
             '<p><a href="README.md">README</a> · <a href="timestamps.csv">Exact timestamps</a> · '
             '<a href="examples.json">Source/output hashes</a></p>',
             '<img src="timeline.svg" alt="Real, synthetic and transition sample regions">']
    for example in examples:
        title = html.escape(f"{example['utt_id']} — {example['vocoder']} — {example['split']}")
        parts.append(f'<article><h2>{title}</h2><p>Edit '
                     f'[{example["start_sample"]}, {example["end_sample"]}) samples; '
                     f'[{example["start_seconds"]:.6f}, {example["end_seconds"]:.6f}) s. '
                     f'{example["fade_samples"]} transition samples per boundary.</p>'
                     '<div class="audio">')
        for title, field in (("Partial synthetic edit", "mixed_file"),
                             ("Real-only null control", "control_file"),
                             ("Original real source", "real_file"),
                             ("Original vocoder source", "donor_file")):
            parts.append(f'<div><p>{title}</p><audio controls preload="none" '
                         f'src="{html.escape(example[field], quote=True)}"></audio></div>')
        parts.append('</div></article>')
        if "real_to_real_file" in example:
            parts.append(
                '<p>Real-to-real control: a different real excerpt from the same utterance, '
                f'donor [{example["real_donor_start_sample"]}, '
                f'{example["real_donor_end_sample"]}) samples. '
                'Content/time correspondence changes; this is not alignment repair. '
                'Core label 0; transition label -1; no synthetic speech.</p>'
                f'<audio controls preload="none" src="{example["real_to_real_file"]}"></audio>'
            )
    return "\n".join([*parts, '</html>\n'])


def write_demo(root: Path, rows: list[dict], sources: dict, source_root: Path, count: int) -> None:
    if not count:
        return
    from experiments.partial_audio import write_json

    examples = preview_records(rows, sources, count)
    folder = root / "demo"
    folder.mkdir()
    by_id = {row["example_id"]: row for row in rows}
    for example in examples:
        row = by_id[example["example_id"]]
        pairs = [(example["mixed_file"], root / row["output_filename"]),
                 (example["control_file"],
                  root / by_id[row["matched_control_id"]]["output_filename"]),
                 (example["real_file"], source_root / sources[row["source_real_id"]]["path"]),
                 (example["donor_file"], source_root / sources[row["source_donor_id"]]["path"])]
        if "real_to_real_file" in example:
            control = by_id[row["matched_real_to_real_control_id"]]
            pairs.append((example["real_to_real_file"], root / control["output_filename"]))
        for relative, source in pairs:
            target = folder / relative
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
    write_json(folder / "examples.json", {"examples": examples,
                                         "development_only": True, "citable": False})
    (folder / "timeline.svg").write_text(timeline_svg(examples), encoding="utf-8")
    (folder / "timestamps.csv").write_text(timestamps_csv(examples), encoding="utf-8", newline="")
    (folder / "index.html").write_text(_index(examples), encoding="utf-8")
    (folder / "README.md").write_text(
        "# VocoPatch offline review\n\n**Development only; non-citable.**\n\n"
        "Open index.html in a browser. Audio, original sources, controls, timeline.svg "
        "and timestamps.csv work offline with no Python or server. Each partial clip "
        "uses the original paired vocoder at identical sample indices. The gold fade "
        "buffers are mixed/ignored; the red core is pure donor audio.\n\n"
        "The same-source null control is decoded-sample-identical to real. It checks "
        "the editing/encoding path and cannot eliminate real-to-synthetic seam cues. "
        "A shared control/source may appear beside several mixed clips. Preview "
        "copies are not independent dataset examples. No model, localization "
        "accuracy or research eligibility is claimed. Full parameters and validation "
        "are in the parent package's provenance.json and validation.json.\n",
        encoding="utf-8",
    )
    if any("real_to_real_file" in example for example in examples):
        with (folder / "README.md").open("a", encoding="utf-8") as stream:
            stream.write(
                "\nReal-to-real controls use disjoint real excerpts from the SAME utterance. "
                "They may repeat/remove words and introduce phase/prosody/level differences; "
                "they are not content matched. All samples are real. Green marks a relocated "
                "real core (label 0), gold marks ignored editing boundaries (label -1). "
                "These controls still cannot eliminate all splice confounds.\n"
            )


def validate_demo(root: Path, rows: list[dict], sources: dict, count: int) -> None:
    if not count:
        if (root / "demo").exists():
            raise ValueError("Unexpected disabled demo directory.")
        return
    folder = root / "demo"
    examples = preview_records(rows, sources, count)
    expected = {"examples": examples, "development_only": True, "citable": False}
    if json.loads((folder / "examples.json").read_text(encoding="utf-8")) != expected:
        raise ValueError("Demo examples differ from the validated dataset records.")
    for name, content in (("timeline.svg", timeline_svg(examples)),
                          ("timestamps.csv", timestamps_csv(examples)),
                          ("index.html", _index(examples))):
        if (folder / name).read_text(encoding="utf-8") != content:
            raise ValueError(f"Demo {name} does not match exact sample labels.")
    for example in examples:
        for relative, digest in example["file_sha256"].items():
            if sha256_file(folder / relative) != digest:
                raise ValueError("Demo audio/source copy hash mismatch.")
