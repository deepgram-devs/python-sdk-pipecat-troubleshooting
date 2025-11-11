"""
Deepgram SDK v5 Synchronous generate() TTS Benchmark with Telemetry enabled

This script benchmarks Deepgram TTS using the v5 SDK's synchronous generate() method,
measuring TTFB, TTLB, and detailed timing breakdown.
"""

import argparse
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepgram import DeepgramClient


def _summaries(metrics: List[Dict[str, float]], key: str) -> Dict[str, float]:
    """Calculate summary statistics for a metric across all runs."""
    values = [m[key] for m in metrics]
    return {
        "min": min(values),
        "max": max(values),
        "avg": statistics.mean(values),
        "median": statistics.median(values),
    }


def _timing_breakdown_summaries(metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate summary statistics for timing breakdown fields."""
    timing_keys = ["iterator_creation_ms", "first_chunk_ms", "remaining_chunks_ms"]
    summaries = {}

    for key in timing_keys:
        values = [m["timing_breakdown"][key] for m in metrics if "timing_breakdown" in m]
        if values:
            summaries[key] = {
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }

    return summaries


def _run_once(
    client: DeepgramClient,
    text: str,
    model: str,
    encoding: str,
    sample_rate: int,
) -> Dict[str, Any]:
    """Execute one TTS request and measure detailed timing.

     V5 SDK uses lazy evaluation - the iterator is created instantly,
    but the HTTP request doesn't execute until first iteration.

    """
    start = time.perf_counter()
    timing_breakdown = {}

    ttfb: Optional[float] = None
    total_bytes = 0
    first_chunk_time: Optional[float] = None

    try:
        # Phase 1: Iterator creation (lazy - no HTTP request yet)
        response = client.speak.v1.audio.generate(
            text=text,
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            container="none",
        )

        # Iterator created (no network activity yet)
        iterator_created = time.perf_counter()
        timing_breakdown["iterator_creation_ms"] = (iterator_created - start) * 1000.0

        # Phase 2: First iteration triggers actual HTTP request + receives first chunk
        # This includes: request preparation, network send, server processing, first chunk receive
        first_iteration_start = time.perf_counter()

        for chunk in response:
            if not chunk:
                continue

            if ttfb is None:
                # First chunk received - this is when TTFB is measured
                ttfb = (time.perf_counter() - start) * 1000.0
                first_chunk_time = time.perf_counter()
                # Time from starting iteration to receiving first chunk (SDK request exec + network)
                timing_breakdown["first_chunk_ms"] = (first_chunk_time - iterator_created) * 1000.0
                total_bytes += len(chunk)
                break  # Exit after first chunk to separate from remaining chunks

        # Phase 3: Process remaining chunks
        remaining_start = time.perf_counter()
        for chunk in response:
            if not chunk:
                continue
            total_bytes += len(chunk)

        timing_breakdown["remaining_chunks_ms"] = (time.perf_counter() - remaining_start) * 1000.0

        ttlb = (time.perf_counter() - start) * 1000.0

        return {
            "ttfb_ms": ttfb if ttfb is not None else ttlb,
            "ttlb_ms": ttlb,
            "bytes": float(total_bytes),
            "timing_breakdown": timing_breakdown,
        }

    except Exception as exc:
        raise RuntimeError(f"SDK request failed: {exc}") from exc


def _write_results(
    metrics: List[Dict[str, Any]],
    summary: Dict[str, Any],
    timing_summary: Dict[str, Dict[str, float]],
    client_init_ms: float,
    args: argparse.Namespace,
    telemetry_enabled: bool,
) -> None:
    """Write metrics and summary to timestamped output directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path(args.output_dir) / "sdk_v5_sync_tts" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "benchmark": "sdk_v5_sync_tts",
        "generated_at": timestamp,
        "iterations": len(metrics),
        "client_init_ms": client_init_ms,
        "telemetry_enabled": telemetry_enabled,
        "parameters": {
            "text": args.text,
            "model": args.model,
            "encoding": args.encoding,
            "sample_rate": args.sample_rate,
        },
        "runs": metrics,
        "summary": summary,
        "timing_breakdown_summary": timing_summary,
    }

    # Write metrics.json
    with (base_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    # Write summary.txt
    with (base_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Deepgram SDK v5 sync benchmark\n")
        handle.write(f"Telemetry: {'enabled' if telemetry_enabled else 'disabled'}\n")
        handle.write(f"Iterations: {len(metrics)}\n")
        handle.write(f"Client initialization (one-time): {client_init_ms:.1f} ms\n\n")

        # Individual run results
        for idx, metric in enumerate(metrics, start=1):
            handle.write(
                f"[{idx:02d}] TTFB {metric['ttfb_ms']:.1f} ms  "
                f"TTLB {metric['ttlb_ms']:.1f} ms  bytes {int(metric['bytes'])}\n"
            )

        handle.write(f"\nSummary (Deepgram SDK v5 sync)\n")
        handle.write(
            f"TTFB  avg {summary['ttfb']['avg']:.1f} ms  "
            f"median {summary['ttfb']['median']:.1f} ms  "
            f"min {summary['ttfb']['min']:.1f}  max {summary['ttfb']['max']:.1f}\n"
        )
        handle.write(
            f"TTLB  avg {summary['ttlb']['avg']:.1f} ms  "
            f"median {summary['ttlb']['median']:.1f} ms  "
            f"min {summary['ttlb']['min']:.1f}  max {summary['ttlb']['max']:.1f}\n"
        )

        # Timing breakdown averages
        handle.write(f"\nPer-Request Timing Breakdown (averages):\n")
        for key, stats in timing_summary.items():
            label = key.replace("_", " ").replace(" ms", "").title()
            handle.write(f"  {label}: {stats['avg']:.1f} ms\n")


def _run(args: argparse.Namespace) -> None:
    """Execute benchmark suite."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY is required", file=sys.stderr)
        sys.exit(1)

    # Measure client initialization (one-time cost)
    # Note: In v5, telemetry is DISABLED by default, so we explicitly enable it
    client_init_start = time.perf_counter()
    client = DeepgramClient(api_key=api_key, telemetry_opt_out=False)
    client_init_ms = (time.perf_counter() - client_init_start) * 1000.0
    print(f"Client initialization: {client_init_ms:.1f} ms\n")

    metrics: List[Dict[str, Any]] = []

    for idx in range(1, args.iterations + 1):
        try:
            result = _run_once(
                client,
                args.text,
                args.model,
                args.encoding,
                args.sample_rate,
            )
            metrics.append(result)
            print(
                f"[{idx:02d}] TTFB {result['ttfb_ms']:.1f} ms  "
                f"TTLB {result['ttlb_ms']:.1f} ms  bytes {int(result['bytes'])}"
            )
        except Exception as exc:
            print(f"[{idx:02d}] ERROR: {exc}", file=sys.stderr)

    if not metrics:
        print("All runs failed; nothing to report", file=sys.stderr)
        sys.exit(2)

    # Calculate summaries
    ttfb_stats = _summaries(metrics, "ttfb_ms")
    ttlb_stats = _summaries(metrics, "ttlb_ms")
    timing_breakdown_stats = _timing_breakdown_summaries(metrics)

    print("\nSummary (Deepgram SDK v5 sync)")
    print(
        f"TTFB  avg {ttfb_stats['avg']:.1f} ms  "
        f"median {ttfb_stats['median']:.1f} ms  "
        f"min {ttfb_stats['min']:.1f}  max {ttfb_stats['max']:.1f}"
    )
    print(
        f"TTLB  avg {ttlb_stats['avg']:.1f} ms  "
        f"median {ttlb_stats['median']:.1f} ms  "
        f"min {ttlb_stats['min']:.1f}  max {ttlb_stats['max']:.1f}"
    )

    print("\nPer-Request Timing Breakdown (averages):")
    for key, stats in timing_breakdown_stats.items():
        label = key.replace("_", " ").replace(" ms", "").title()
        print(f"  {label}: {stats['avg']:.1f} ms")

    summary = {"ttfb": ttfb_stats, "ttlb": ttlb_stats}
    _write_results(metrics, summary, timing_breakdown_stats, client_init_ms, args, telemetry_enabled=True)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Deepgram TTS via SDK v5 synchronous mode with timing instrumentation."
    )
    parser.add_argument("--text", default="Testing Deepgram SDK streaming TTS.")
    parser.add_argument("--model", default="aura-2-andromeda-en")
    parser.add_argument("--encoding", default="linear16")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where benchmark results will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    _run(_parse_args())


if __name__ == "__main__":
    main()
