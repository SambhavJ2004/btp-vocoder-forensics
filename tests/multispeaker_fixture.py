"""Reuse existing speech/room and native-layout fixtures, never genuine VCTK."""

from pathlib import Path

import librosa
from test_phase_a_run import _speechlike
from vctk_fixture import make_fixture, put_source

from data.source_ingestion import read_json, write_json


def make_generation_fixture(root):
    root = make_fixture(root, speakers=8, utterances=4)
    for speaker in range(8):
        for utterance in range(4):
            # Fixture acquisition simulation only, not a production intermediate.
            wav = librosa.resample(
                _speechlike(4.7, seed=speaker * 31 + utterance),
                orig_sr=22050,
                target_sr=48000,
                res_type="soxr_hq",
                fix=True,
                scale=False,
            )
            put_source(root, f"p{201 + speaker}_{utterance + 1:03d}", wav)
    marker = read_json(root / "fixture.json")
    marker.update(
        producer="tests/multispeaker_fixture.py using existing _speechlike/make_fixture",
        real_vctk_speakers=0,
    )
    write_json(root / "fixture.json", marker)
    return root


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(make_generation_fixture(args.out_dir))
