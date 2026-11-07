"""Thin CLI. Kaggle notebooks are drivers; the logic lives in the package.

Usage:
    btpvf audit                       # which conditions are runnable yet
    btpvf resynth --corpus-root ... --conditions ... --out-dir ... --n-utterances N
                                      # Phase A, every gate, output only if all pass
    btpvf validate --manifest ...     # manifest-level invariant gate
    btpvf sanity   --manifest ...     # confound-leakage probes, both tiers
    btpvf kaggle run --mode iteration|record [--sha ...]
                                      # Phase A on Kaggle from a code snapshot
    btpvf silence-diag --manifest ... # diagnose a silence-probe hit (not a gate)
    btpvf provenance --manifest ...   # INV-01 derivation paths per tier
    btpvf loudness --manifest ...     # INV-04 gain delta and in-band offset vs real
    btpvf merge --manifest A --manifest B --out-dir ...
                                      # INV-13: per-condition runs -> one dataset
    btpvf quality  --manifest ...     # Phase B, X-axis
    btpvf plot     --quality ... --detection ...
    btpvf pilot train|evaluate|predict # non-citable CPU forensic development pilot
    btpvf partial build|validate      # non-citable paired partial-audio pilot
    btpvf temporal train|predict      # non-citable train/dev localization baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="btpvf", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit", help="checkpoint / mel-config readiness table")

    ingest = sub.add_parser("ingest", help="non-citable native source intake; no generation")
    intake_sub = ingest.add_subparsers(dest="ingest_cmd", required=True)
    intake = intake_sub.add_parser("vctk", help="read-only VCTK 0.92 mic2 source selection")
    intake.add_argument("--source-root", required=True)
    intake.add_argument("--out-dir", required=True)
    intake.add_argument(
        "--evidence-dir", default=None,
        help="directory containing official README.txt and license_text.txt",
    )
    intake.add_argument("--speakers", type=int, default=50)
    intake.add_argument("--utterances-per-speaker", type=int, default=10)
    intake.add_argument("--source-kind", choices=["local_dataset", "fixture"], required=True)
    check_intake = intake_sub.add_parser("verify", help="remeasure source intake and integrity")
    check_intake.add_argument("--ingestion-dir", required=True)
    check_intake.add_argument("--source-root", default=None, help="explicit relocated source root")
    check_intake.add_argument("--evidence-dir", default=None)

    a = sub.add_parser(
        "resynth",
        help="Phase A: gated resynthesis -> full-band archive, derived tier, manifest",
    )
    a.add_argument("--corpus-root", required=True, help="LJSpeech-1.1 directory")
    a.add_argument("--conditions", nargs="+", required=True)
    a.add_argument("--out-dir", required=True, help="must not exist yet")
    a.add_argument("--n-utterances", type=int, required=True)
    a.add_argument("--device", default="auto", help="auto | cpu | cuda[:N]")
    a.add_argument(
        "--keep-failed", action="store_true",
        help="on a failed gate, keep staging as <out-dir>.failed instead of deleting it",
    )

    v = sub.add_parser("validate", help="manifest invariant gate")
    v.add_argument("--manifest", required=True)

    pr = sub.add_parser("provenance", help="INV-01 derivation paths and mel configs")
    pr.add_argument("--manifest", required=True)

    ld = sub.add_parser("loudness", help="INV-04: per-condition loudness gain vs real")
    ld.add_argument("--manifest", required=True)
    ld.add_argument("--out", default="results/loudness_gain.csv")

    s = sub.add_parser("sanity", help="confound-leakage probes")
    s.add_argument("--manifest", required=True)
    s.add_argument("--split", default="eval")
    s.add_argument("--out", default="results/sanity.csv")

    sd = sub.add_parser(
        "silence-diag",
        help="diagnose a silence-probe hit: noise floor in silent regions, threshold sweep",
    )
    sd.add_argument("--manifest", required=True)
    sd.add_argument("--out-dir", default="results")
    sd.add_argument("--limit", type=int, default=None, help="utterances to read (default all)")

    kg = sub.add_parser("kaggle", help="run Phase A on Kaggle from a code snapshot")
    kgsub = kg.add_subparsers(dest="kaggle_cmd", required=True)
    kgrun = kgsub.add_parser("run", help="snapshot the code, push a kernel, fetch the output")
    kgrun.add_argument("--mode", choices=["iteration", "record"], required=True,
                       help="iteration = working tree, marked dirty; record = git archive of --sha")
    kgrun.add_argument("--sha", default=None, help="commit to archive (record mode only)")
    kgrun.add_argument("--conditions", nargs="+", default=["bigvgan_112m", "vocos", "griffin_lim"])
    kgrun.add_argument("--n-utterances", type=int, default=200)
    kgrun.add_argument("--corpus", default="auto",
                       help="LJSpeech root on Kaggle; 'auto' finds it by shape under /kaggle/input")
    kgrun.add_argument("--dataset", action="append", dest="datasets",
                       default=["mathurinache/the-lj-speech-dataset"],
                       help="extra owner/slug to attach; repeatable")
    kgrun.add_argument("--slug", default="btpvf-v0", help="kernel slug (kept private)")
    kgrun.add_argument("--out-name", default=None,
                       help="output directory name under /kaggle/working "
                            "(default: resynth_v0). Use a v1 name so a v1 run can "
                            "never be mistaken for, or overwrite, v0.")
    kgrun.add_argument("--title", default="btpvf v0 resynthesis")
    kgrun.add_argument("--gpu", action="store_true",
                       help="request a GPU; omit for CPU, which costs no weekly quota")
    kgrun.add_argument("--timeout", type=int, default=32400, help="kernel run limit, seconds")
    kgcode = kgsub.add_parser("code", help="snapshot and version the code dataset only")
    kgcode.add_argument("--mode", choices=["iteration", "record"], required=True)
    kgcode.add_argument("--sha", default=None)
    kgsub.add_parser("notebook", help="regenerate notebooks/kaggle_v0.ipynb from V0_SPEC")

    mg = sub.add_parser("merge", help="INV-13: combine per-condition runs into one dataset")
    mg.add_argument("--manifest", action="append", required=True, dest="manifests",
                    help="a per-condition run's manifest.csv; repeat once per run")
    mg.add_argument("--out-dir", required=True, help="must not exist yet")
    mg.add_argument("--allow-differing-real", action="store_true",
                    help="report instead of refusing when real is not byte-identical "
                         "across runs. For inspecting a mismatch, never for shipping one.")

    q = sub.add_parser("quality", help="Phase B: quality metrics")
    q.add_argument("--manifest", required=True)
    q.add_argument("--out-dir", default="results")
    q.add_argument("--device", default="cpu")

    m = sub.add_parser("plot", help="main plot: quality vs detectability")
    m.add_argument("--quality", required=True, help="quality_by_condition.csv")
    m.add_argument("--detection", required=True, help="detection results csv")
    m.add_argument("--out", default="results/main_plot.png")
    m.add_argument("--quality-col", default="utmos_mean")

    pilot = sub.add_parser("pilot", help="non-citable multi-vocoder CPU forensic pilot")
    pilots = pilot.add_subparsers(dest="pilot_cmd", required=True)
    pt = pilots.add_parser("train", help="train on train, select on dev, evaluate eval")
    pt.add_argument("--manifest", required=True)
    pt.add_argument("--out-dir", required=True, help="must not exist yet")
    pt.add_argument("--dataset-id", default="btpvf-v0")
    from data.invariants import GLOBAL_SEED

    pt.add_argument("--seed", type=int, default=GLOBAL_SEED)
    pe = pilots.add_parser("evaluate", help="reproduce frozen eval using a trusted local bundle")
    pe.add_argument("--bundle", required=True)
    pe.add_argument("--manifest", required=True)
    pe.add_argument("--out-dir", required=True, help="must not exist yet")
    pp = pilots.add_parser("predict", help="score one mono PCM_16 archive-rate WAV")
    pp.add_argument("--bundle", required=True, help="trusted local bundle directory")
    pp.add_argument("--wav", required=True)
    pp.add_argument("--work-dir", default=None, help="parent for temporary analysis WAV")

    partial = sub.add_parser("partial", help="local partial-audio development dataset")
    partial_sub = partial.add_subparsers(dest="partial_cmd", required=True)
    pb = partial_sub.add_parser("build", help="validate sources, splice and validate a local pilot")
    pb.add_argument("--manifest", required=True)
    pb.add_argument("--out-dir", required=True, help="must not exist yet")
    pb.add_argument("--n-utterances", type=int, default=None,
                    help="default 12, or the sum of explicit split counts")
    pb.add_argument("--all-utterances", action="store_true",
                    help="scan every frozen source group (max 200); requires both controls")
    pb.add_argument("--seed", type=int, default=GLOBAL_SEED)
    pb.add_argument("--segment-seconds", type=float, default=1.0)
    pb.add_argument("--crossfade-ms", type=float, default=10.0)
    pb.add_argument("--boundary-margin-seconds", type=float, default=0.25)
    pb.add_argument("--variants", type=int, choices=(1, 2), default=1)
    pb.add_argument("--real-to-real-controls", action="store_true",
                    help="also splice a different region of the same real utterance")
    pb.add_argument("--split-count", action="append", default=None,
                    help="explicit frozen-split count, e.g. train=4; repeat for dev/eval")
    pv = partial_sub.add_parser("validate", help="read-only reconstruction and integrity checks")
    pv.add_argument("--dataset-dir", required=True)
    pv.add_argument("--source-manifest", required=True)
    pu = partial_sub.add_parser("prepare-upload", help="local private-upload folder; no network")
    pu.add_argument("--dataset-dir", required=True)
    pu.add_argument("--source-manifest", required=True)
    pu.add_argument("--out-dir", required=True, help="must not exist yet")
    pu.add_argument("--dataset-card", required=True)
    pu.add_argument("--protocol", required=True)
    uv = partial_sub.add_parser("validate-upload", help="read-only local upload-folder checks")
    uv.add_argument("--dataset-dir", required=True)
    uv.add_argument("--source-manifest", required=True)

    temporal = sub.add_parser("temporal", help="CPU development temporal localization")
    temporal_sub = temporal.add_subparsers(dest="temporal_cmd", required=True)
    tt = temporal_sub.add_parser("train", help="fit on train and select/report on dev only")
    tt.add_argument("--dataset-dir", required=True)
    tt.add_argument("--source-manifest", required=True)
    tt.add_argument("--out-dir", required=True)
    tt.add_argument("--setting", action="append", help="window_ms:hop_ms; repeat for search")
    tt.add_argument("--c", action="append", type=float)
    tt.add_argument("--threshold", action="append", type=float)
    tt.add_argument("--smoothing", action="append", type=int, choices=(1, 3))
    tt.add_argument("--merge-ms", action="append", type=float)
    tt.add_argument("--seed", type=int, default=GLOBAL_SEED)
    tp = temporal_sub.add_parser("predict", help="single-file model intervals, no truth input")
    tp.add_argument("--bundle", required=True)
    tp.add_argument("--wav", required=True)
    tp.add_argument("--work-dir", required=True)

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        from data.source_ingestion import IntakePolicy, ingest_vctk, verify_ingestion

        if args.ingest_cmd == "verify":
            result = verify_ingestion(
                args.ingestion_dir, source_root=args.source_root, evidence_dir=args.evidence_dir,
            )
        else:
            result = ingest_vctk(
                args.source_root, args.out_dir,
                policy=IntakePolicy(args.speakers, args.utterances_per_speaker),
                evidence_dir=args.evidence_dir, source_kind=args.source_kind,
            )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "temporal":
        if args.temporal_cmd == "train":
            from experiments.temporal_localization import TemporalExperimentConfig, train_temporal

            default = TemporalExperimentConfig()
            settings = (tuple(tuple(map(float, value.split(":"))) for value in args.setting)
                        if args.setting else default.settings)
            result = train_temporal(args.dataset_dir, args.source_manifest, args.out_dir,
                TemporalExperimentConfig(settings=settings, c_grid=tuple(args.c or default.c_grid),
                    thresholds=tuple(args.threshold or default.thresholds),
                    smoothing_widths=tuple(args.smoothing or default.smoothing_widths),
                    merge_ms=tuple(args.merge_ms or default.merge_ms), seed=args.seed))
        else:
            from detectors.temporal_model import predict_temporal_file

            result = predict_temporal_file(args.bundle, args.wav, args.work_dir)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.cmd == "partial":
        from data.partial_audio import PartialAudioConfig
        from experiments.partial_audio import build_partial_dataset, validate_partial_dataset

        if args.partial_cmd == "build":
            from data.invariants import SPLIT_NAMES

            split_counts = None
            if args.all_utterances and (args.split_count or args.n_utterances is not None):
                raise ValueError("--all-utterances cannot be combined with counts or quotas.")
            if args.split_count:
                requested = {}
                for specification in args.split_count:
                    name, separator, count = specification.partition("=")
                    if not separator or name not in SPLIT_NAMES or name in requested:
                        raise ValueError("Use each frozen split once: --split-count train=4")
                    requested[name] = int(count)
                split_counts = tuple(requested.get(name, 0) for name in SPLIT_NAMES)
            n_utterances = args.n_utterances
            if n_utterances is None:
                n_utterances = sum(split_counts) if split_counts is not None else 12
            if args.all_utterances:
                from experiments.partial_audio import source_frame

                n_utterances = int(source_frame(args.manifest).utt_id.nunique())
            result = build_partial_dataset(args.manifest, args.out_dir, PartialAudioConfig(
                seed=args.seed, n_utterances=n_utterances,
                segment_seconds=args.segment_seconds, crossfade_ms=args.crossfade_ms,
                boundary_margin_seconds=args.boundary_margin_seconds, variants=args.variants,
                real_to_real_controls=args.real_to_real_controls, split_counts=split_counts,
                all_utterances=args.all_utterances,
            ))
        elif args.partial_cmd == "prepare-upload":
            from experiments.partial_audio_upload import prepare_upload_directory

            result = prepare_upload_directory(args.dataset_dir, args.source_manifest,
                                              args.out_dir, args.dataset_card, args.protocol)
        elif args.partial_cmd == "validate-upload":
            from experiments.partial_audio_upload import validate_upload_directory

            result = validate_upload_directory(args.dataset_dir, args.source_manifest)
        else:
            result = validate_partial_dataset(args.dataset_dir, args.source_manifest)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.cmd == "pilot":
        if args.pilot_cmd == "predict":
            from detectors.forensic_pilot import predict_file

            result = predict_file(args.bundle, args.wav, work_dir=args.work_dir)
        else:
            from experiments.forensic_pilot import evaluate_pilot, train_pilot

            if args.pilot_cmd == "train":
                result = train_pilot(args.manifest, args.out_dir,
                                     dataset_id=args.dataset_id, seed=args.seed)
            else:
                result = evaluate_pilot(args.bundle, args.manifest, args.out_dir)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.cmd == "audit":
        import pandas as pd

        from vocoders.registry import checkpoint_audit

        print(pd.DataFrame(checkpoint_audit()).to_string(index=False))
        return 0

    if args.cmd == "resynth":
        from experiments.phase_a_resynthesis import run_gated

        result = run_gated(
            args.corpus_root,
            args.conditions,
            args.out_dir,
            args.n_utterances,
            device=None if args.device == "auto" else args.device,
            keep_failed=args.keep_failed,
        )
        print(result.sanity.to_string(index=False))
        print()
        print(result.loudness.to_string(index=False))
        print(f"\nall gates passed; dataset at {result.out_dir}")
        print(f"manifest: {result.manifest_path}")
        return 0

    if args.cmd == "validate":
        from data.manifest import read_manifest, validate_manifest

        validate_manifest(read_manifest(args.manifest))
        print("manifest OK: all table-level invariants hold")
        return 0

    if args.cmd == "provenance":
        from data.manifest import (
            band_report,
            mel_config_table,
            provenance_table,
            read_manifest,
        )

        df = read_manifest(args.manifest)
        print(provenance_table(df).to_string(index=False))
        print()
        print(band_report(df).to_string(index=False))
        print()
        print(mel_config_table(df).to_string(index=False))
        return 0

    if args.cmd == "loudness":
        from data.manifest import loudness_gain_report, read_manifest

        report = loudness_gain_report(read_manifest(args.manifest))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False)
        print(report.to_string(index=False))
        return 0

    if args.cmd == "sanity":
        from data.manifest import read_manifest
        from experiments.sanity_checks import run_all

        df = run_all(read_manifest(args.manifest), split=args.split)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(df.to_string(index=False))
        if df["leaked"].any():
            print("\nCONFOUND LEAK: fix the pipeline and regenerate. Do not relax the floor.")
            return 1
        return 0

    if args.cmd == "silence-diag":
        import pandas as pd

        from data.manifest import resolve_audio_paths
        from experiments.sanity_checks import silence_diagnostic

        # Read tolerantly: this runs on manifests written by earlier versions,
        # including the failed run being diagnosed. Paths still go through the
        # one resolution, which is what makes a manifest whose directory was
        # renamed under it -- `<out>.staging` -> `<out>.failed` -- readable.
        df = pd.read_csv(args.manifest)
        missing = {"utt_id", "condition", "tier", "path", "split"} - set(df.columns)
        if missing:
            raise SystemExit(f"manifest lacks {sorted(missing)}")
        df = resolve_audio_paths(df, args.manifest)
        if df.attrs.get("paths_relocated"):
            print(
                f"[paths] {df.attrs['paths_relocated']} rows record an absolute path "
                f"from another machine or directory; read from "
                f"{df.attrs['manifest_root']}, the manifest's own, instead."
            )
        floors, sweep, ref = silence_diagnostic(df, limit=args.limit)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        floors.to_csv(out_dir / "silence_floors.csv", index=False)
        sweep.to_csv(out_dir / "silence_threshold_sweep.csv", index=False)
        pd.DataFrame([ref.as_record()]).to_csv(
            out_dir / "silence_calibration.csv", index=False
        )
        print("the derived silence threshold for this dataset (INV-03/INV-12):")
        for key, value in ref.as_record().items():
            print(f"  {key:<28} {value}")
        if not ref.ok:
            print(
                "  ^ the threshold does NOT read the trim pad; `btpvf sanity` raises on "
                "this dataset rather than reporting a silence EER."
            )
        print("\nlevel inside the reference's silent regions:")
        print(floors.to_string(index=False))
        print("\nthe probe's own feature, swept over thresholds:")
        print(sweep.to_string(index=False))
        print(f"\nwritten to {out_dir}/silence_floors.csv, silence_threshold_sweep.csv "
              "and silence_calibration.csv")
        print("Diagnosis only: INV-12's floor and verdicts are unchanged.")
        return 0

    if args.cmd == "kaggle":
        from experiments.kaggle_harness import (
            KernelSpec,
            check_auth,
            push_code_dataset,
            run_kernel,
            stage_snapshot,
        )

        if args.kaggle_cmd == "notebook":
            from experiments.kaggle_harness import V0_SPEC, write_v0_notebook

            path = write_v0_notebook()
            print(f"regenerated {path} from V0_SPEC: {V0_SPEC.conditions} "
                  f"n={V0_SPEC.n_utterances} gpu={V0_SPEC.gpu}")
            return 0

        auth = check_auth()
        print(f"[auth] {auth['username']} via {auth['auth_method']}")  # never the token
        if args.kaggle_cmd == "code":
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                snapshot = stage_snapshot(Path(tmp), mode=args.mode, sha=args.sha)
                result = push_code_dataset(Path(tmp), snapshot, username=auth["username"])
            print(json.dumps({"snapshot": snapshot.as_record(), "dataset": result}, indent=2))
            return 0

        spec = KernelSpec(
            slug=args.slug,
            title=args.title,
            conditions=tuple(args.conditions),
            n_utterances=args.n_utterances,
            corpus=args.corpus,
            gpu=args.gpu,
            dataset_sources=tuple(args.datasets),
            timeout_s=args.timeout,
            **({"out_dir": f"/kaggle/working/{args.out_name}"} if args.out_name else {}),
        )
        record = run_kernel(spec, mode=args.mode, sha=args.sha)
        print(f"\nrun record: runs/{record['stamp']}/run.json")
        if not record["snapshot"]["citable"]:
            print("ITERATION MODE: debugging output. Do not cite it in docs/measurements.md.")
        return 0

    if args.cmd == "merge":
        from data.merge import merge_runs, summarise

        record = merge_runs(
            args.manifests, args.out_dir,
            require_identical_real=not args.allow_differing_real,
        )
        print(summarise(record))
        print(f"\nmerge record: {args.out_dir}/manifest.merge.json")
        return 0

    if args.cmd == "quality":
        from experiments.phase_b_quality import run

        paths = run(args.manifest, args.out_dir, device=args.device)
        print("\n".join(str(x) for x in paths))
        return 0

    if args.cmd == "plot":
        import pandas as pd

        from experiments.analysis import main_plot, summarise

        quality = pd.read_csv(args.quality)
        detection = pd.read_csv(args.detection)
        _, stats = main_plot(quality, detection, args.out, quality_col=args.quality_col)
        print(summarise(quality, detection, quality_col=args.quality_col))
        print(f"figure: {args.out}  rho={stats['rho']:+.3f}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
