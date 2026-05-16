#!/usr/bin/env python3
"""Build a tab-separated summary of per-epoch evaluation results.

Scans ``<eval_per_epoch_dir>/epoch*/eval/evaluation_results.json`` and writes
one row per epoch into ``summary.tsv``. Also prepends a ``baseline`` row from
``<model_root>/eval_zeroshot/evaluation_results.json`` (zero-shot DINOv3) when
available, so you can directly compare each epoch against the un-fine-tuned
baseline.

Columns:
    epoch  num_pairs  auc@5  auc@10  auc@20  precision  Δauc@10  Δprecision

The two ``Δ`` columns are written *only* in the on-screen preview; the TSV
itself stays compact (epoch + 5 metrics) so it stays trivially parseable.

Missing files / fields are written as ``-``.

Usage:
    python scripts/build_eval_summary.py <path> [output.tsv] [options]

``<path>`` may point to either:
  * a model root (``output/navi_small``)               — recommended
  * an ``eval_per_epoch`` dir (``.../eval_per_epoch``) — also fine

Options:
  --baseline <file>     Override baseline JSON path.
  --no-baseline         Don't include a baseline row.
  -o / --output <file>  Override output TSV path.

Examples:
    # Most common: just point at the model root.
    python scripts/build_eval_summary.py \\
        /root/autodl-tmp/cv-project/output/navi_small

    # Also works:
    python scripts/build_eval_summary.py \\
        /root/autodl-tmp/cv-project/output/navi_middle/eval_per_epoch
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

COLUMNS = ("num_pairs", "auc@5", "auc@10", "auc@20", "precision")
HEADER = ("epoch",) + COLUMNS
# Extra columns that only show up in the on-screen preview (not in TSV).
PREVIEW_DELTA_COLUMNS = ("Δauc@10", "Δprecision")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_delta(cur: Any, base: Any) -> str:
    if not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
        return "-"
    d = float(cur) - float(base)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4g}"


def _parse_legacy_txt(path: Path) -> dict:
    """Best-effort parse of the older plain-text evaluation_results.txt."""
    text = path.read_text(errors="ignore")

    def grab(pat: str):
        m = re.search(pat, text, re.IGNORECASE)
        return m.group(1) if m else None

    out: dict = {}
    for key, pat in (
        ("num_pairs", r"num[_ ]?pairs[^0-9\-]*([0-9]+)"),
        ("auc@5", r"auc@?5[^0-9\-]*([0-9.]+)"),
        ("auc@10", r"auc@?10[^0-9\-]*([0-9.]+)"),
        ("auc@20", r"auc@?20[^0-9\-]*([0-9.]+)"),
        ("precision", r"precision[^0-9\-]*([0-9.]+)"),
    ):
        v = grab(pat)
        if v is not None:
            try:
                out[key] = float(v) if "." in v else int(v)
            except ValueError:
                out[key] = v
    return out


def _load_metrics(json_path: Path | None,
                  txt_path: Path | None = None) -> dict | None:
    if json_path and json_path.exists():
        try:
            return json.loads(json_path.read_text())
        except json.JSONDecodeError:
            print(f"[warn] failed to parse {json_path}, falling back to legacy",
                  file=sys.stderr)
    if txt_path and txt_path.exists():
        return _parse_legacy_txt(txt_path)
    return None


def _load_epoch_metrics(epoch_dir: Path) -> dict | None:
    return _load_metrics(epoch_dir / "eval" / "evaluation_results.json",
                         epoch_dir / "eval" / "evaluation_results.txt")


def _load_baseline_metrics(path: Path) -> dict | None:
    """Path may be a JSON file directly, or a dir containing it."""
    if path.is_dir():
        return _load_metrics(path / "evaluation_results.json",
                             path / "evaluation_results.txt")
    return _load_metrics(path if path.suffix == ".json" else None,
                         path if path.suffix == ".txt" else None)


def _epoch_tag_to_int(tag: str) -> int:
    digits = re.sub(r"\D", "", tag)
    return int(digits) if digits else -1


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _resolve_paths(input_path: Path,
                   baseline_override: Path | None,
                   no_baseline: bool,
                   output_override: Path | None
                   ) -> tuple[Path, Path | None, Path]:
    """Resolve (eval_per_epoch_dir, baseline_json_or_dir, output_tsv)."""
    p = input_path.expanduser().resolve()
    if not p.is_dir():
        raise SystemExit(f"[error] not a directory: {p}")

    # Heuristic: input is either a model root (contains eval_per_epoch/)
    # or an eval_per_epoch dir itself.
    if (p / "eval_per_epoch").is_dir():
        model_root = p
        eval_per_epoch = p / "eval_per_epoch"
    elif p.name == "eval_per_epoch":
        eval_per_epoch = p
        model_root = p.parent
    else:
        # Fallback: treat as eval_per_epoch even if name doesn't match.
        eval_per_epoch = p
        model_root = p.parent

    if no_baseline:
        baseline = None
    elif baseline_override is not None:
        baseline = baseline_override.expanduser().resolve()
    else:
        # Default baseline location produced by the pipeline.
        candidate = model_root / "eval_zeroshot"
        baseline = candidate if candidate.exists() else None

    out_tsv = (output_override.expanduser().resolve()
               if output_override is not None
               else eval_per_epoch / "summary.tsv")

    return eval_per_epoch, baseline, out_tsv


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_summary(eval_per_epoch: Path,
                  baseline_path: Path | None,
                  out_tsv: Path) -> int:
    # --- baseline ---
    base_metrics: dict | None = None
    if baseline_path is not None:
        base_metrics = _load_baseline_metrics(baseline_path)
        if base_metrics is None:
            print(f"[warn] baseline path given but no results found: {baseline_path}",
                  file=sys.stderr)

    # --- per-epoch ---
    epoch_dirs = sorted(
        (p for p in eval_per_epoch.glob("epoch*") if p.is_dir()),
        key=lambda p: _epoch_tag_to_int(p.name.replace("epoch", "")),
    )

    # rows: list of (label, metrics_dict_or_None)
    labelled: list[tuple[str, dict | None]] = []
    if base_metrics is not None or baseline_path is not None:
        # Always include the baseline label even if file was missing — keeps
        # the TSV self-describing about what comparison is intended.
        labelled.append(("baseline", base_metrics))

    n_ok = n_missing = 0
    for ep_dir in epoch_dirs:
        tag = ep_dir.name.replace("epoch", "")
        m = _load_epoch_metrics(ep_dir)
        if m is None:
            n_missing += 1
        else:
            n_ok += 1
        labelled.append((tag, m))

    # --- TSV (compact, no Δ columns) ---
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    rows_tsv: list[tuple[str, ...]] = []
    for label, m in labelled:
        if m is None:
            rows_tsv.append((label, *("-" for _ in COLUMNS)))
        else:
            rows_tsv.append((label, *(_fmt(m.get(c)) for c in COLUMNS)))

    with out_tsv.open("w") as f:
        f.write("\t".join(HEADER) + "\n")
        for row in rows_tsv:
            f.write("\t".join(row) + "\n")

    print(f"[ok] wrote {out_tsv}  "
          f"({n_ok} epoch(s) with results, {n_missing} missing"
          f"{', baseline ✓' if base_metrics is not None else ''})")

    # --- on-screen preview WITH delta columns vs baseline ---
    preview_header = HEADER + (PREVIEW_DELTA_COLUMNS if base_metrics else ())
    preview_rows: list[tuple[str, ...]] = []
    for (label, m), row in zip(labelled, rows_tsv):
        extra: tuple[str, ...] = ()
        if base_metrics:
            if label == "baseline" or m is None:
                extra = ("-", "-")
            else:
                extra = (
                    _fmt_delta(m.get("auc@10"), base_metrics.get("auc@10")),
                    _fmt_delta(m.get("precision"), base_metrics.get("precision")),
                )
        preview_rows.append(row + extra)

    if preview_rows:
        widths = [max(len(h), *(len(r[i]) for r in preview_rows))
                  for i, h in enumerate(preview_header)]
        sep = "  "
        print(sep.join(h.ljust(w) for h, w in zip(preview_header, widths)))
        print(sep.join("-" * w for w in widths))
        for row in preview_rows:
            print(sep.join(c.ljust(w) for c, w in zip(row, widths)))

    return 0 if n_ok > 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_argv(argv: list[str]) -> tuple[Path, Path | None, bool, Path | None]:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0)

    input_path: Path | None = None
    baseline: Path | None = None
    no_baseline = False
    output: Path | None = None

    positional: list[str] = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--baseline":
            baseline = Path(argv[i + 1]); i += 2; continue
        if a == "--no-baseline":
            no_baseline = True; i += 1; continue
        if a in ("-o", "--output"):
            output = Path(argv[i + 1]); i += 2; continue
        if a.startswith("--"):
            print(f"[error] unknown option: {a}", file=sys.stderr); sys.exit(2)
        positional.append(a); i += 1

    if not positional:
        print("[error] missing <path>", file=sys.stderr); sys.exit(2)
    input_path = Path(positional[0])
    if len(positional) >= 2 and output is None:
        # legacy 2nd positional = output tsv
        output = Path(positional[1])
    if len(positional) > 2:
        print("[error] too many positional arguments", file=sys.stderr); sys.exit(2)

    return input_path, baseline, no_baseline, output


def main(argv: list[str]) -> int:
    input_path, baseline_override, no_baseline, output_override = _parse_argv(argv)
    eval_per_epoch, baseline_path, out_tsv = _resolve_paths(
        input_path, baseline_override, no_baseline, output_override,
    )
    return build_summary(eval_per_epoch, baseline_path, out_tsv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
