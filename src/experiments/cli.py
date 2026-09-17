"""Thin CLI. Kaggle notebooks are drivers; the logic lives in the package.

Usage:
    btpvf audit                       # which conditions are runnable yet
    btpvf resynth --corpus-root ... --conditions ... --out-dir ... --n-utterances N
                                      # Phase A, every gate, output only if all pass
    btpvf validate --manifest ...     # manifest-level invariant gate
    btpvf sanity   --manifest ...     # confound-leakage probes, both tiers
    btpvf provenance --manifest ...   # INV-01 derivation paths per tier
    btpvf loudness --manifest ...     # INV-04 gain delta and in-band offset vs real
    btpvf quality  --manifest ...     # Phase B, X-axis
    btpvf plot     --quality ... --detection ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="btpvf", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit", help="checkpoint / mel-config readiness table")

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

    q = sub.add_parser("quality", help="Phase B: quality metrics")
    q.add_argument("--manifest", required=True)
    q.add_argument("--out-dir", default="results")
    q.add_argument("--device", default="cpu")

    m = sub.add_parser("plot", help="main plot: quality vs detectability")
    m.add_argument("--quality", required=True, help="quality_by_condition.csv")
    m.add_argument("--detection", required=True, help="detection results csv")
    m.add_argument("--out", default="results/main_plot.png")
    m.add_argument("--quality-col", default="utmos_mean")

    args = p.parse_args(argv)

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
