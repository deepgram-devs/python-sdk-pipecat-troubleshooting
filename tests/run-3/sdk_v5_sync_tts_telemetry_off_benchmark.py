"""
Deepgram SDK v5 Synchronous TTS Benchmark with Granular Timing - Telemetry Disabled (Run 3)

This script benchmarks Deepgram SDK v5's synchronous generate() method with
telemetry DISABLED, measuring 6 granular timing milestones to understand
exactly where latency occurs and whether telemetry has any performance impact.
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
    timing_keys = [
        "time_until_request_ms",
        "time_until_response_ms",
        "time_until_results_ms",
    ]
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
    """Execute one TTS request and measure granular timing at 6 milestones.

    V5 SDK uses lazy evaluation - the iterator is created instantly,
    but the HTTP request doesn't execute until first iteration.

    Milestones (chronological):
    1. Start
    2. Time until request (when first iteration starts, HTTP is sent)
    3. Time until response (after HTTP completes, headers received)
    4. TTFB (first data chunk arrives)
    5. Time until response results (all chunks processed, iterator closed)
    6. TTLB (last data chunk received)
    """
    start = time.perf_counter()
    timing_breakdown = {}

    ttfb: Optional[float] = None
    total_bytes = 0

    try:
        # Create iterator (lazy - no HTTP request yet)
        response = client.speak.v1.audio.generate(
            text=text,
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            container="none",
        )

        # Milestone 2: Time until request (first iteration starts, HTTP is sent)
        # Note: We need to enter the iteration to trigger the HTTP request
        iterator = iter(response)
        request_sent_time = time.perf_counter()
        timing_breakdown["time_until_request_ms"] = (request_sent_time - start) * 1000.0

        # Get first chunk (this triggers HTTP request and receives response)
        first_chunk = next(iterator)

        # Milestone 3: Time until response (HTTP complete, headers received, before chunk captured)
        response_received_time = time.perf_counter()
        timing_breakdown["time_until_response_ms"] = (response_received_time - start) * 1000.0

        # Milestone 4: TTFB - First data chunk (already received above)
        if first_chunk:
            ttfb = (time.perf_counter() - start) * 1000.0
            total_bytes += len(first_chunk)

        # Collect remaining chunks
        for chunk in iterator:
            if not chunk:
                continue
            total_bytes += len(chunk)

        # Milestone 6: TTLB - Last chunk received
        ttlb = (time.perf_counter() - start) * 1000.0

        # Milestone 5: Time until response results (iterator exhausted, data finalized)
        results_ready_time = time.perf_counter()
        timing_breakdown["time_until_results_ms"] = (results_ready_time - start) * 1000.0

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
    base_dir = Path(args.output_dir) / "run-3" / "sdk_v5_sync_tts_telemetry_off" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "benchmark": "sdk_v5_sync_tts_telemetry_off_run3",
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
        handle.write("Deepgram SDK v5 sync benchmark (Run 3 - Granular Timing, Telemetry Disabled)\n")
        handle.write(f"Telemetry: {'enabled' if telemetry_enabled else 'disabled'}\n")
        handle.write(f"Iterations: {len(metrics)}\n")
        handle.write(f"Client initialization (one-time): {client_init_ms:.1f} ms\n\n")

        # Individual run results
        for idx, metric in enumerate(metrics, start=1):
            handle.write(
                f"[{idx:02d}] TTFB {metric['ttfb_ms']:.1f} ms  "
                f"TTLB {metric['ttlb_ms']:.1f} ms  bytes {int(metric['bytes'])}\n"
            )

        # Consolidated timing breakdown (chronological order)
        handle.write(f"\nPer-Request Timing Breakdown (chronological):\n")
        handle.write(
            f"  Time Until Request: "
            f"avg {timing_summary['time_until_request_ms']['avg']:.1f} ms  "
            f"median {timing_summary['time_until_request_ms']['median']:.1f} ms  "
            f"min {timing_summary['time_until_request_ms']['min']:.1f}  "
            f"max {timing_summary['time_until_request_ms']['max']:.1f}\n"
        )
        handle.write(
            f"  Time Until Response: "
            f"avg {timing_summary['time_until_response_ms']['avg']:.1f} ms  "
            f"median {timing_summary['time_until_response_ms']['median']:.1f} ms  "
            f"min {timing_summary['time_until_response_ms']['min']:.1f}  "
            f"max {timing_summary['time_until_response_ms']['max']:.1f}\n"
        )
        handle.write(
            f"  Ttfb: "
            f"avg {summary['ttfb']['avg']:.1f} ms  "
            f"median {summary['ttfb']['median']:.1f} ms  "
            f"min {summary['ttfb']['min']:.1f}  "
            f"max {summary['ttfb']['max']:.1f}\n"
        )
        handle.write(
            f"  Ttlb: "
            f"avg {summary['ttlb']['avg']:.1f} ms  "
            f"median {summary['ttlb']['median']:.1f} ms  "
            f"min {summary['ttlb']['min']:.1f}  "
            f"max {summary['ttlb']['max']:.1f}\n"
        )
        handle.write(
            f"  Time Until Results: "
            f"avg {timing_summary['time_until_results_ms']['avg']:.1f} ms  "
            f"median {timing_summary['time_until_results_ms']['median']:.1f} ms  "
            f"min {timing_summary['time_until_results_ms']['min']:.1f}  "
            f"max {timing_summary['time_until_results_ms']['max']:.1f}\n"
        )


def _run(args: argparse.Namespace) -> None:
    """Execute benchmark suite."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY is required", file=sys.stderr)
        sys.exit(1)

    # Milestone 1: Measure client initialization (one-time cost)
    # Note: In v5, telemetry is DISABLED by default, explicitly setting it for clarity
    client_init_start = time.perf_counter()
    client = DeepgramClient(api_key=api_key, telemetry_opt_out=True)
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

    # Consolidated timing breakdown (chronological order)
    print("\nPer-Request Timing Breakdown (chronological):")
    print(
        f"  Time Until Request: "
        f"avg {timing_breakdown_stats['time_until_request_ms']['avg']:.1f} ms  "
        f"median {timing_breakdown_stats['time_until_request_ms']['median']:.1f} ms  "
        f"min {timing_breakdown_stats['time_until_request_ms']['min']:.1f}  "
        f"max {timing_breakdown_stats['time_until_request_ms']['max']:.1f}"
    )
    print(
        f"  Time Until Response: "
        f"avg {timing_breakdown_stats['time_until_response_ms']['avg']:.1f} ms  "
        f"median {timing_breakdown_stats['time_until_response_ms']['median']:.1f} ms  "
        f"min {timing_breakdown_stats['time_until_response_ms']['min']:.1f}  "
        f"max {timing_breakdown_stats['time_until_response_ms']['max']:.1f}"
    )
    print(
        f"  Ttfb: "
        f"avg {ttfb_stats['avg']:.1f} ms  "
        f"median {ttfb_stats['median']:.1f} ms  "
        f"min {ttfb_stats['min']:.1f}  "
        f"max {ttfb_stats['max']:.1f}"
    )
    print(
        f"  Ttlb: "
        f"avg {ttlb_stats['avg']:.1f} ms  "
        f"median {ttlb_stats['median']:.1f} ms  "
        f"min {ttlb_stats['min']:.1f}  "
        f"max {ttlb_stats['max']:.1f}"
    )
    print(
        f"  Time Until Results: "
        f"avg {timing_breakdown_stats['time_until_results_ms']['avg']:.1f} ms  "
        f"median {timing_breakdown_stats['time_until_results_ms']['median']:.1f} ms  "
        f"min {timing_breakdown_stats['time_until_results_ms']['min']:.1f}  "
        f"max {timing_breakdown_stats['time_until_results_ms']['max']:.1f}"
    )

    summary = {"ttfb": ttfb_stats, "ttlb": ttlb_stats}
    _write_results(metrics, summary, timing_breakdown_stats, client_init_ms, args, telemetry_enabled=False)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Deepgram TTS via SDK v5 synchronous mode with telemetry disabled and granular timing instrumentation (Run 3)."
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

