"""The sanctioned reader accepts a verified snapshot and existing paths."""

import io
import struct
import wave

import numpy as np

from data.audio_io import probe, read_native


def test_read_native_and_probe_match_for_a_snapshot_and_a_path(tmp_path):
    pcm = (-32768, -123, 0, 123, 32767)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setparams((1, 2, 22_050, len(pcm), "NONE", "not compressed"))
        writer.writeframes(struct.pack("<5h", *pcm))
    path = tmp_path / "synthetic.wav"
    path.write_bytes(stream.getvalue())
    stream.seek(0)
    snapshot_header = probe(stream)
    path_header = probe(path)
    assert snapshot_header["sample_rate"] == path_header["sample_rate"] == 22_050
    assert {k: v for k, v in snapshot_header.items() if k != "path"} == {
        k: v for k, v in path_header.items() if k != "path"
    }
    stream.seek(0)
    in_memory, memory_rate = read_native(stream)
    on_disk, disk_rate = read_native(path)
    assert memory_rate == disk_rate == 22_050
    np.testing.assert_array_equal(in_memory, on_disk)
    np.testing.assert_array_equal(in_memory, np.asarray(pcm, dtype=np.float32) / 32768)
