"""CPU checks for the data-independent base training protocol."""

import copy
import hashlib
import io
import json
import math
import sys
from dataclasses import FrozenInstanceError, replace
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from vocoroute import losses, train_base  # noqa: E402
from vocoroute.spectral import SpectralFrontend  # noqa: E402
from vocoroute.train_base import BaseTrainingModel, main  # noqa: E402
from vocoroute.training_checkpoint import (  # noqa: E402
    RECEIPT_FILE,
    WEIGHTS_FILE,
    initialize_backbone,
    load_base_checkpoint,
    run_provenance,
    save_base_checkpoint,
)
from vocoroute.training_config import LossWeights, TrainingConfig, canonical_json  # noqa: E402
from vocoroute.training_data import (  # noqa: E402
    MANIFEST_SCHEMA,
    TrainingAudio,
    load_manifest,
    prepare_example,
)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _signal(kind, length):
    t = torch.arange(length, dtype=torch.float32)
    if kind == "sinusoid":
        return 0.25 * torch.sin(2 * math.pi * 440 * t / 22_050)
    if kind == "transient":
        return 0.3 * torch.cos(t) * torch.exp(-(((t - length / 2) / 15) ** 2))
    result = torch.zeros(length)
    if kind == "impulse":
        result[length // 2] = 0.5
    return result


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _manifest_document():
    rows = []
    for index, role in enumerate(["train"] * 8 + ["dev", "eval"]):
        source = f"source-{index}"
        digest = _digest(source.encode())
        path = f"recordings/{source}.bin"
        rows.append(
            {
                "utterance_id": f"utt-{index}",
                "speaker_id": f"speaker-{role}",
                "source_group": source,
                "role": role,
                "native_path": path,
                "native_sha256": digest,
                "native_rate": 22_050,
                "native_samples": 1025,
                "prepared_path": path,
                "prepared_sha256": digest,
                "prepared_samples": 1025,
                "resample_steps": 0,
                "resampling_profile": "identity",
                "resampling_receipt_sha256": None,
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": "synthetic-metadata-test",
        "source_kind": "native_generator",
        "target_rate": 22_050,
        "rows": rows,
    }


def _write_manifest(tmp_path, doc, filename="manifest.json"):
    path = tmp_path / filename
    data = canonical_json(doc)
    path.write_bytes(data)
    return path, _digest(data)


@pytest.mark.parametrize("kind", ["silence", "impulse", "sinusoid", "transient"])
@pytest.mark.parametrize("length", [1, 257, 1025])
def test_loss_identical_and_perturbed_have_finite_cpu_gradients(kind, length):
    target = _signal(kind, length)[None].requires_grad_()
    criterion = losses.ReconstructionLoss(LossWeights(phase_enabled=True))
    identical = criterion(target.detach().clone().requires_grad_(), target)
    torch.testing.assert_close(torch.stack(identical), torch.zeros(len(identical)), rtol=0, atol=0)
    predicted = (0.81 * target.detach().roll(1, dims=-1) + 0.025).requires_grad_()
    terms = criterion(predicted, target)
    assert terms.waveform > 0 and terms.magnitude > 0 and terms.log_magnitude >= 0
    assert torch.isfinite(torch.stack(terms)).all()
    terms.total.backward()
    assert target.grad is None
    assert torch.isfinite(predicted.grad).all()
    if kind == "silence":
        assert terms.phase_time == 0 and terms.phase_frequency == 0
    if length == 1:
        assert terms.phase_time == 0


def test_phase_distance_wrap_and_reference_energy_mask():
    epsilon = 0.02
    predicted = torch.polar(torch.ones(1, 1, 2), torch.tensor([[[0.0, -math.pi + epsilon]]]))
    reference = torch.polar(torch.ones(1, 1, 2), torch.tensor([[[0.0, math.pi - epsilon]]]))
    magnitude = torch.ones(1, 1, 2)
    temporal = losses._phase_pair_loss(predicted, reference, magnitude, dim=2)
    spectral = losses._phase_pair_loss(predicted.mT, reference.mT, magnitude.mT, dim=1)
    assert temporal.item() == pytest.approx(1 - math.cos(2 * epsilon), abs=1e-8)
    torch.testing.assert_close(temporal, spectral)
    magnitude[:, :, 1] = 1e-8
    assert losses._phase_pair_loss(predicted, reference, magnitude, dim=2) == 0


def test_phase_ablation_skips_phase_path_and_applies_only_selected_weights(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("the disabled phase calculation executed")

    monkeypatch.setattr(losses, "_phase_pair_loss", unexpected)
    criterion = losses.ReconstructionLoss(LossWeights(2.0, 0.5, 0.0, False, 99.0, 99.0))
    target = _signal("transient", 1025)[None]
    terms = criterion(target.roll(2, dims=-1), target)
    torch.testing.assert_close(terms.total, 2 * terms.waveform + 0.5 * terms.magnitude)
    assert terms.phase_time == 0 and terms.phase_frequency == 0


@pytest.mark.parametrize("length", [1, 255, 256, 1025])
def test_example_uses_the_same_sample_origin_with_exact_synthesis_length(length):
    frontend = SpectralFrontend()
    signal = _signal("impulse", length)
    original = signal.clone()
    example = prepare_example(frontend, signal)
    assert example.length == length
    assert example.mel.shape == (1, 80, 1 + length // 256)
    assert torch.equal(example.target[0], original)
    expected = frontend.condition(frontend.analyze(original[None]))
    torch.testing.assert_close(example.mel, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        frontend.synthesize(frontend.analyze(example.target), length),
        example.target,
        atol=2e-7,
        rtol=0,
    )
    assert torch.equal(signal, original)
    signal.zero_()
    assert torch.equal(example.target[0], original)


def test_base_parameter_shape_and_full_backward_without_optimizer(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("an optimizer was constructed")

    monkeypatch.setattr(torch.optim, "AdamW", unexpected)
    config = TrainingConfig(loss=LossWeights(phase_enabled=True))
    rng = torch.random.get_rng_state().clone()
    model = BaseTrainingModel(config)
    clone = initialize_backbone(config.seed)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert sum(param.numel() for param in model.parameters()) == 566_144
    for left, right in zip(model.backbone.parameters(), clone.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    signal = _signal("transient", 1025)
    example = prepare_example(model.frontend, signal)
    spectrum = model.backbone(example.mel)
    assert spectrum.shape == (1, 513, 5)
    assert spectrum[:, [0, -1]].imag.count_nonzero() == 0
    wave = model.frontend.synthesize(spectrum, example.length)
    assert wave.shape == (1, 1025)
    expected = model.loss(wave, example.target)
    computed = model(signal)
    torch.testing.assert_close(torch.stack(computed), torch.stack(expected))
    computed.total.backward()
    for param in model.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
    assert model.backbone.head.weight.grad.count_nonzero() > 0


def test_config_is_canonical_strict_and_readonly():
    cfg = TrainingConfig(epochs=2, loss=LossWeights(phase_enabled=True))
    wire = canonical_json(cfg.document())
    restored = TrainingConfig.from_document(json.loads(wire))
    assert restored == cfg
    assert restored.sha256 == _digest(wire)
    assert restored.sha256 != replace(restored, seed=1).sha256
    with pytest.raises(FrozenInstanceError):
        restored.epochs = 3
    for group, key, value in (
        ("frontend", "pad_mode", "reflect"),
        ("frontend", "center", 1),
        ("roles", "fit", "eval"),
        ("optimizer", "name", "SGD"),
        ("scheduler", "gamma", 3),
        ("loss", "phase_enabled", 1),
    ):
        doc = cfg.document()
        doc[group][key] = value
        with pytest.raises(ValueError):
            TrainingConfig.from_document(doc)


def test_pinned_manifest_ordering_and_immutable_roles(tmp_path):
    doc = _manifest_document()
    path, pin = _write_manifest(tmp_path, doc)
    manifest = load_manifest(path, pin)
    reverse = copy.deepcopy(doc)
    reverse["rows"].reverse()
    rev_path, rev_pin = _write_manifest(tmp_path, reverse, "reverse.json")
    reversed_manifest = load_manifest(rev_path, rev_pin)
    assert manifest.examples("train", seed=17, epoch=3) == reversed_manifest.examples(
        "train", seed=17, epoch=3
    )
    assert manifest.examples("train", seed=17, epoch=3) != manifest.examples(
        "train", seed=17, epoch=4
    )
    assert all(row.role == "train" for row in manifest.examples("train"))
    assert all(row.role == "dev" for row in manifest.examples("dev"))
    with pytest.raises(ValueError, match="only exposes train and dev"):
        manifest.examples("eval")
    with pytest.raises(FrozenInstanceError):
        manifest.rows[0].role = "dev"
    doc["rows"][0]["role"] = "eval"
    _write_manifest(tmp_path, doc)
    with pytest.raises(ValueError, match="approved SHA-256"):
        load_manifest(path, pin)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("speaker_id", "SPEAKER-TRAIN", "speaker-disjoint"),
        ("source_group", "source-0", "source groups"),
        ("utterance_id", "utt-0", "unique"),
        ("native_path", "../recording.wav", "relative paths"),
        ("native_path", "recordings/source-0.bin", "unique"),
        ("resample_steps", 2, "resampling"),
        ("prepared_samples", 1024, "sample count"),
        ("role", "dev", "must contain"),
    ],
)
def test_rejects_invalid_manifest_rows(tmp_path, field, value, error):
    doc = _manifest_document()
    doc["rows"][-1][field] = value
    path, pin = _write_manifest(tmp_path, doc)
    with pytest.raises(ValueError, match=error):
        load_manifest(path, pin)


def test_resampling_provenance_and_byte_identity(tmp_path):
    doc = _manifest_document()
    row = doc["rows"][0]
    row.update(
        native_rate=48_000,
        native_samples=48_001,
        prepared_samples=22_051,
        resample_steps=1,
        resampling_profile="soxr_hq_profile_v1",
        resampling_receipt_sha256="1" * 64,
        prepared_path="derived/0.wav",
        prepared_sha256="2" * 64,
    )
    path, pin = _write_manifest(tmp_path, doc)
    assert load_manifest(path, pin).rows[0].resampling_receipt_sha256 == "1" * 64
    row["resampling_receipt_sha256"] = None
    path, pin = _write_manifest(tmp_path, doc)
    with pytest.raises(ValueError, match="pinned resampling"):
        load_manifest(path, pin)

    doc = _manifest_document()
    duplicate = doc["rows"][0]["native_sha256"]
    doc["rows"][-1].update(native_sha256=duplicate, prepared_sha256=duplicate)
    path, pin = _write_manifest(tmp_path, doc)
    with pytest.raises(ValueError, match="byte identities"):
        load_manifest(path, pin)


def test_dry_run_and_missing_real_never_invoke_training(tmp_path, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        pytest.fail("dry-run or missing real data never enters training")

    monkeypatch.setattr("vocoroute.train_base._run_training", forbidden)
    assert main(["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["base_parameters"] == 566_144
    doc = _manifest_document()
    path, pin = _write_manifest(tmp_path, doc)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_bytes(canonical_json(TrainingConfig(manifest_sha256=pin).document()))
    args = ["--config", str(cfg_path), "--manifest", str(path), "--data-root", str(tmp_path)]
    with pytest.raises(SystemExit, match="2"):
        main(["--dry-run", *args])
    assert "recordings" in capsys.readouterr().err
    out = tmp_path / "unused-output"
    with pytest.raises(SystemExit, match="2"):
        main(["--train", *args, "--out-dir", str(out)])
    assert not out.exists()

    # Only opaque integrity-test bytes are on disk. Waveforms stay in memory.
    (tmp_path / "recordings").mkdir()
    for row in doc["rows"][:-1]:
        (tmp_path / row["native_path"]).write_bytes(row["source_group"].encode())
    assert main(["--dry-run", *args]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["role_counts"] == {"train": 8, "dev": 1, "eval": 1}
    manifest = load_manifest(path, pin)
    access = TrainingAudio(manifest, tmp_path)
    with pytest.raises(ValueError, match="train/dev"):
        access.waveform(manifest.rows[-1])
    (tmp_path / manifest.rows[0].prepared_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        TrainingAudio(manifest, tmp_path)


def test_audio_interface_enforces_headers_geometry_and_pinned_bytes(tmp_path, monkeypatch):
    doc = _manifest_document()
    manifest_path, pin = _write_manifest(tmp_path, doc)
    (tmp_path / "recordings").mkdir()
    for row in doc["rows"][:-1]:
        (tmp_path / row["native_path"]).write_bytes(row["source_group"].encode())
    manifest = load_manifest(manifest_path, pin)
    source = _signal("impulse", 1025)
    seen = []
    headers = {"channels": 1, "sample_rate": 22_050, "frames": source.numel()}
    stub = ModuleType("data.audio_io")

    def probe(stream):
        seen.append(stream.read().decode())
        return headers

    stub.probe = probe
    decoded = []
    decoded_rate = 22_050

    def read_native(stream):
        decoded.append(stream.read())
        return source.numpy(), decoded_rate

    stub.read_native = read_native
    monkeypatch.setitem(sys.modules, "data.audio_io", stub)
    audio = TrainingAudio(manifest, tmp_path)
    audio.validate_headers()
    assert "source-9" not in seen
    row = manifest.rows[0]
    torch.testing.assert_close(audio.waveform(row), source)
    assert decoded == [b"source-0"]
    headers["channels"] = 2
    with pytest.raises(ValueError, match="header disagrees"):
        audio.validate_headers()
    with pytest.raises(ValueError, match="header disagrees"):
        audio.waveform(row)
    assert decoded == [b"source-0"]
    headers["channels"] = 1
    decoded_rate = 16_000
    with pytest.raises(ValueError, match="decoded waveform disagrees"):
        audio.waveform(row)

    def replace_during_probe(stream):
        info = probe(stream)
        (tmp_path / row.prepared_path).write_bytes(b"replacement")
        return info

    decoded_rate = 22_050
    stub.probe = replace_during_probe
    torch.testing.assert_close(audio.waveform(row), source)
    assert decoded == [b"source-0"] * 3
    (tmp_path / row.prepared_path).write_bytes(b"changed after preflight")
    with pytest.raises(ValueError, match="SHA-256 mismatch after preflight"):
        audio.waveform(row)
    assert decoded == [b"source-0"] * 3


@pytest.mark.parametrize("field", ["native_path", "prepared_path"])
def test_preflight_checks_distinct_native_and_prepared_files(tmp_path, field, monkeypatch):
    doc = _manifest_document()
    row = doc["rows"][0]
    row.update(
        native_rate=48_000,
        native_samples=2231,
        resample_steps=1,
        resampling_profile="soxr_hq_profile_v1",
        resampling_receipt_sha256="1" * 64,
        prepared_path="prepared/source-0.bin",
        prepared_sha256=_digest(b"prepared"),
    )
    doc["rows"] = [row, *doc["rows"][-2:]]
    path, pin = _write_manifest(tmp_path, doc)
    (tmp_path / "recordings").mkdir()
    (tmp_path / "prepared").mkdir()
    for example in doc["rows"][:-1]:
        (tmp_path / example["native_path"]).write_bytes(example["source_group"].encode())
    (tmp_path / row["prepared_path"]).write_bytes(b"prepared")
    manifest = load_manifest(path, pin)
    access = TrainingAudio(manifest, tmp_path)
    (tmp_path / row[field]).write_bytes(b"mismatch")
    with pytest.raises(ValueError, match="recording SHA-256 mismatch"):
        TrainingAudio(manifest, tmp_path)
    stub = ModuleType("data.audio_io")
    stub.probe = lambda stream: {
        b"source-0": {"channels": 1, "sample_rate": 48_000, "frames": 2231},
    }[stream.read()]
    monkeypatch.setitem(sys.modules, "data.audio_io", stub)
    with pytest.raises(ValueError, match="SHA-256 mismatch after preflight"):
        access.validate_headers()


def test_checkpoint_round_trip_and_rejects_tampering(tmp_path):
    path, pin = _write_manifest(tmp_path, _manifest_document())
    cfg = TrainingConfig(manifest_sha256=pin, epochs=2)
    model = initialize_backbone(cfg.seed)
    run = run_provenance(cfg, load_manifest(path, pin))
    history = [
        {"epoch": 1, "train_total": 2.0, "dev_total": 1.0},
        {"epoch": 2, "train_total": 1.0, "dev_total": 1.0},
    ]
    target = tmp_path / "checkpoint"
    receipt_pin = save_base_checkpoint(target, cfg, run, history, 1, model.state_dict())
    loaded, receipt = load_base_checkpoint(target, cfg, receipt_sha256=receipt_pin)
    assert receipt["run"]["config_sha256"] == cfg.sha256 and not loaded.training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        save_base_checkpoint(target, cfg, run, history, 1, model.state_dict())
    with pytest.raises(ValueError, match="first minimum"):
        save_base_checkpoint(tmp_path / "wrong-epoch", cfg, run, history, 2, model.state_dict())
    with pytest.raises(ValueError, match="configuration"):
        load_base_checkpoint(target, replace(cfg, seed=19), receipt_sha256=receipt_pin)
    with pytest.raises(ValueError, match="external SHA-256"):
        load_base_checkpoint(target, cfg, receipt_sha256="0" * 64)

    old_receipt = (target / RECEIPT_FILE).read_bytes()
    incompatible = copy.deepcopy(receipt)
    incompatible["run"]["source_sha256"]["vocoroute/spectral.py"] = "0" * 64
    raw = canonical_json(incompatible)
    (target / RECEIPT_FILE).write_bytes(raw)
    with pytest.raises(ValueError, match="frontend source differs"):
        load_base_checkpoint(target, cfg, receipt_sha256=_digest(raw))
    (target / RECEIPT_FILE).write_bytes(old_receipt)

    weights_path = target / WEIGHTS_FILE
    weights_path.write_bytes(weights_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="weights SHA-256"):
        load_base_checkpoint(target, cfg, receipt_sha256=receipt_pin)
    state = model.state_dict()
    state["stem.weight"] = state["stem.weight"][:1]
    buffer = io.BytesIO()
    torch.save(state, buffer)
    weights_path.write_bytes(buffer.getvalue())
    receipt["weights_sha256"] = _digest(buffer.getvalue())
    raw = canonical_json(receipt)
    (target / RECEIPT_FILE).write_bytes(raw)
    with pytest.raises(ValueError, match="invalid checkpoint tensor"):
        load_base_checkpoint(target, cfg, receipt_sha256=_digest(raw))

    buffer = io.BytesIO()
    torch.save({"unsupported": target}, buffer)
    weights_path.write_bytes(buffer.getvalue())
    receipt["weights_sha256"] = _digest(buffer.getvalue())
    raw = canonical_json(receipt)
    (target / RECEIPT_FILE).write_bytes(raw)
    with pytest.raises(ValueError, match="rejected by the weights-only loader"):
        load_base_checkpoint(target, cfg, receipt_sha256=_digest(raw))


def test_frontend_config_matches_the_live_analysis():
    from torchaudio.functional import melscale_fbanks

    settings = TrainingConfig().document()["frontend"]
    waveform = _signal("transient", 257)[None]
    frontend = SpectralFrontend()
    assert settings["window"] == "periodic_hann" and settings["pad_mode"] == "constant_zero"
    stft = torch.stft(
        waveform,
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
        win_length=settings["win_length"],
        window=torch.hann_window(settings["win_length"]),
        center=settings["center"],
        pad_mode="constant",
        normalized=settings["normalized"],
        onesided=settings["onesided"],
        return_complex=True,
    )
    torch.testing.assert_close(frontend.analyze(waveform), stft, atol=2e-6, rtol=2e-6)
    bank = melscale_fbanks(
        stft.shape[1],
        settings["f_min"],
        settings["f_max"],
        settings["n_mels"],
        settings["sample_rate"],
        settings["mel_norm"],
        settings["mel_scale"],
    )
    mel = (bank.T @ stft.abs().pow(settings["mel_power"])).clamp_min(settings["mel_floor"]).log()
    torch.testing.assert_close(frontend.condition(stft), mel, atol=2e-6, rtol=2e-6)


def test_bounded_synthetic_optimizer_smoke(tmp_path, monkeypatch, capsys):
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    step_budget = 8
    data = tmp_path / "synthetic"
    (data / "recordings").mkdir(parents=True)
    doc = _manifest_document()
    doc["dataset_id"] = "synthetic-n2a1-optimizer-smoke"
    doc["rows"] = [doc["rows"][0], *doc["rows"][-2:]]
    path, pin = _write_manifest(tmp_path, doc)
    config = TrainingConfig(epochs=step_budget, manifest_sha256=pin)
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_json(config.document()))
    train = _signal("sinusoid", 1025) + 0.1 * _signal("transient", 1025)
    dev = 0.98 * train + 0.003 * _signal("impulse", 1025)
    waves = {b"source-0": train, b"source-8": dev}
    for row in doc["rows"][:-1]:
        (data / row["prepared_path"]).write_bytes(row["source_group"].encode())
    stub = ModuleType("data.audio_io")
    stub.probe = lambda stream: {
        "channels": 1,
        "sample_rate": 22_050,
        "frames": waves[stream.read()].numel(),
    }
    decode_order = []

    def read_native(stream):
        content = stream.read()
        decode_order.append(content)
        return waves[content].numpy(), 22_050

    stub.read_native = read_native
    monkeypatch.setitem(sys.modules, "data.audio_io", stub)
    initial_model = BaseTrainingModel(config).eval()
    with torch.no_grad():
        initial = initial_model(train)
    reference = prepare_example(initial_model.frontend, train)

    runner = []
    forward = BaseTrainingModel.forward

    def observe_forward(model, waveform):
        runner[:] = [model]
        return forward(model, waveform)

    monkeypatch.setattr(BaseTrainingModel, "forward", observe_forward)
    before_reload = []
    save = train_base.save_base_checkpoint

    def observe_save(directory, cfg, run, history, best_epoch, weights):
        saved_model = initialize_backbone(cfg.seed).eval()
        saved_model.load_state_dict(weights, strict=True)
        with torch.no_grad():
            before_reload.append(
                initial_model.frontend.synthesize(saved_model(reference.mel), reference.length)
            )
        return save(directory, cfg, run, history, best_epoch, weights)

    monkeypatch.setattr(train_base, "save_base_checkpoint", observe_save)
    gradients = []

    def observe_step(optimizer, _args, _kwargs):
        assert len(gradients) < step_budget
        params = [param for group in optimizer.param_groups for param in group["params"]]
        assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in params)
        gradients.append(max(param.grad.abs().max().item() for param in params))

    handle = register_optimizer_step_pre_hook(observe_step)
    output = tmp_path / "checkpoint"
    try:
        assert (
            main(
                [
                    "--train",
                    "--config",
                    str(config_path),
                    "--manifest",
                    str(path),
                    "--data-root",
                    str(data),
                    "--out-dir",
                    str(output),
                ]
            )
            == 0
        )
    finally:
        handle.remove()
    summary = json.loads(capsys.readouterr().out)
    assert len(gradients) == step_budget and all(gradient > 0 for gradient in gradients)
    assert decode_order == [b"source-0", b"source-8"] * step_budget
    last = runner[0]
    with torch.no_grad():
        final = last(train)
    assert torch.isfinite(torch.stack(final)).all()
    delta = max(
        (end.detach() - start.detach()).abs().max().item()
        for start, end in zip(initial_model.parameters(), last.parameters(), strict=True)
    )
    assert delta > 0
    assert final.total < initial.total
    loaded, receipt = load_base_checkpoint(
        output, config, receipt_sha256=summary["checkpoint_receipt_sha256"]
    )
    with torch.no_grad():
        reloaded = initial_model.frontend.synthesize(loaded(reference.mel), reference.length)
    assert reloaded.shape == (1, reference.length)
    torch.testing.assert_close(before_reload[0], reloaded, rtol=0, atol=0)
    print(
        json.dumps(
            {
                "synthetic_optimizer_steps": len(gradients),
                "initial_total": initial.total.item(),
                "final_total": final.total.item(),
                "initial_waveform_l1": initial.waveform.item(),
                "final_waveform_l1": final.waveform.item(),
                "max_gradient": max(gradients),
                "max_parameter_delta": delta,
                "best_dev_epoch": receipt["best_epoch"],
                "dev_totals": [row["dev_total"] for row in receipt["history"]],
                "reload_max_waveform_error": (before_reload[0] - reloaded).abs().max().item(),
            },
            sort_keys=True,
        )
    )
