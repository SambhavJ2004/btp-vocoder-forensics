"""Generate clearly labeled external-source-format fixtures, never research audio.

soundfile writes here deliberately simulate an acquired 48-kHz FLAC source.
Production ingestion only reads it; processed audio writers remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def native_fixture(seed, seconds=4.6, rate=48000):
    rng = np.random.default_rng(seed)
    t = np.arange(round(rate * seconds)) / rate
    signal = (
        0.07 * np.sin(2 * np.pi * (170 + seed % 1000) * t)
        + 0.025 * np.sin(2 * np.pi * (2200 + seed % 2000) * t)
        + 0.004 * rng.standard_normal(len(t))
    )
    signal[: round(0.08 * rate)] = 0
    signal[-round(0.08 * rate) :] = 0
    return signal.astype(np.float32)


def make_fixture(root, *, speakers=6, utterances=3, invalid=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "fixture.json").write_text(
        json.dumps(
            {
                "kind": "synthetic_audio_fixture",
                "speech_or_rights_evidence": False,
                "citable": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "README.txt").write_text(
        "VCTK 0.92 layout SYNTHETIC FIXTURE. "
        "Tones/noise, not real speech or an official release.\n",
        encoding="utf-8",
    )
    (root / "license_text.txt").write_text(
        "Synthetic parser fixture only. Does not verify external source rights.\n"
        "Expected source license identifier: https://creativecommons.org/licenses/by/4.0/legalcode\n",
        encoding="utf-8",
    )
    info = ["ID AGE GENDER ACCENT REGION"]
    for i in range(speakers):
        p = f"p{201 + i}"
        info.append(f"{201 + i} unknown unknown synthetic_fixture")
        (root / "txt" / p).mkdir(parents=True)
        (root / "wav48_silence_trimmed" / p).mkdir(parents=True)
        for u in range(1, utterances + 1):
            nid = f"{p}_{u:03d}"
            put_source(root, nid, native_fixture((i + 1) * 101 + u))
    (root / "speaker-info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    if invalid:
        if speakers < 6:
            raise ValueError("Invalid fixture requires six identity records.")
        put_source(root, "p201_900", np.zeros(48000 * 5, dtype=np.float32))
        put_source(root, "p202_901", native_fixture(9001, seconds=2))
        put_source(root, "p203_902", native_fixture(9002))
        (root / "wav48_silence_trimmed/p203/p203_902_mic2.flac").write_bytes(b"fLaCinvalid")
        put_source(root, "p204_903", native_fixture(9003))
        (root / "wav48_silence_trimmed/p204/p204_903_mic2.flac").unlink()
        put_source(root, "p205_904", native_fixture(9004, rate=44100), rate=44100)
        duplicate = native_fixture(9005)
        put_source(root, "p206_905", duplicate)
        put_source(root, "p201_906", duplicate)
    return root


def put_source(root, native_id, wav, *, rate=48000, text=None, subtype="PCM_16"):
    root = Path(root)
    p = native_id.split("_")[0]
    sf.write(
        root / "wav48_silence_trimmed" / p / f"{native_id}_mic2.flac",
        wav,
        rate,
        format="FLAC",
        subtype=subtype,
    )
    (root / "txt" / p / f"{native_id}.txt").write_text(
        text if text is not None else f"Synthetic fixture content identifier {native_id}.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--speakers", type=int, default=6)
    parser.add_argument("--utterances-per-speaker", type=int, default=3)
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args()
    print(
        make_fixture(
            args.out_dir,
            speakers=args.speakers,
            utterances=args.utterances_per_speaker,
            invalid=args.include_invalid,
        )
    )
