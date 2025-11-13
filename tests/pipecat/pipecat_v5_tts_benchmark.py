"""
PipeCat + Deepgram SDK v5.3 TTS Benchmark

Tests PipeCat's DeepgramTTSService modified to use SDK v5.3 with 6 granular timing metrics:
1. Client initialization (one-time setup cost)
2. Time until request (generator creation)
3. Time until response (first frame received after SDK call)
4. TTFB (first audio chunk)
5. TTLB (last audio chunk)
6. Time until results (cleanup/finalization)

Usage:
    PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/pipecat/pipecat_v5_tts_benchmark.py --iterations 25
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

# Import our modified v5.3 service
from deepgram_tts_service_v5 import DeepgramTTSServiceV5


async def _run_once(
    service: DeepgramTTSServiceV5,
    text: str,
) -> Dict:
    """
    Run a single TTS request and measure timing at 6 milestones.

    Returns dict with:
    - success: bool
    - error: Optional[str]
    - ttfb_ms: float (time to first audio byte)
    - ttlb_ms: float (time to last audio byte)
    - total_bytes: int
    - timing_breakdown: Dict with all 6 milestones
    """
    start = time.perf_counter()
    timing_breakdown: Dict[str, float] = {}

    ttfb: Optional[float] = None
    total_bytes = 0
    frame_count = 0
    first_frame_time: Optional[float] = None

    try:
        # Call PipeCat's TTS service (modified for v5.3)
        # This returns an async generator of Frame objects (lazy evaluation - no SDK call yet)
        frames = service.run_tts(text)

        # Milestone 2: Time until request (about to start iteration, which triggers SDK call)
        request_start = time.perf_counter()
        timing_breakdown["time_until_request_ms"] = (request_start - start) * 1000.0

        # Process frames from PipeCat
        # Expected sequence: TTSStartedFrame → TTSAudioRawFrame(s) → TTSStoppedFrame
        # The SDK call happens during the first iteration
        response_received = False

        async for frame in frames:
            # Milestone 3: Time until response (first frame received, SDK call complete)
            if not response_received:
                response_ready = time.perf_counter()
                timing_breakdown["time_until_response_ms"] = (response_ready - start) * 1000.0
                response_received = True

            if isinstance(frame, TTSStartedFrame):
                # TTS has started, but no audio yet
                continue

            elif isinstance(frame, TTSAudioRawFrame):
                frame_count += 1
                audio_data = frame.audio

                # Milestone 4: TTFB - First audio frame
                if ttfb is None:
                    ttfb = (time.perf_counter() - start) * 1000.0
                    first_frame_time = time.perf_counter()

                total_bytes += len(audio_data)

            elif isinstance(frame, TTSStoppedFrame):
                # TTS complete
                break

        # Milestone 6: TTLB - Last audio frame received
        ttlb = (time.perf_counter() - start) * 1000.0

        # Milestone 5: Time until response results (data processed and ready)
        # For PipeCat, this is right after all frames are consumed
        results_ready = time.perf_counter()
        timing_breakdown["time_until_results_ms"] = (results_ready - start) * 1000.0

        if ttfb is None:
            return {
                "success": False,
                "error": "No audio frames received",
                "ttfb_ms": 0,
                "ttlb_ms": 0,
                "total_bytes": 0,
                "timing_breakdown": timing_breakdown,
            }

        return {
            "success": True,
            "error": None,
            "ttfb_ms": ttfb,
            "ttlb_ms": ttlb,
            "total_bytes": total_bytes,
            "timing_breakdown": timing_breakdown,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ttfb_ms": 0,
            "ttlb_ms": 0,
            "total_bytes": 0,
            "timing_breakdown": timing_breakdown,
        }


async def run_benchmark(iterations: int) -> Dict:
    """
    Run TTS benchmark for specified iterations.

    Returns dict with:
    - results: List[Dict] of individual run results
    - client_init_ms: float (one-time initialization cost)
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set")

    text = "Hello, this is a test of the Deepgram text to speech API using PipeCat with SDK v5.3."

    # Milestone 1: Service initialization (one-time cost)
    print("Initializing PipeCat DeepgramTTSServiceV5...")
    init_start = time.perf_counter()
    service = DeepgramTTSServiceV5(
        api_key=api_key,
        voice="aura-2-andromeda-en",
        sample_rate=16000,
        encoding="linear16",
    )
    init_time = (time.perf_counter() - init_start) * 1000.0
    print(f"Service initialization: {init_time:.1f} ms\n")

    results = []
    for i in range(iterations):
        result = await _run_once(service, text)
        results.append(result)

        if result["success"]:
            print(
                f"[{i+1:02d}] TTFB {result['ttfb_ms']:.1f} ms  "
                f"TTLB {result['ttlb_ms']:.1f} ms  bytes {result['total_bytes']}"
            )
        else:
            print(f"[{i+1:02d}] ERROR: {result['error']}")

    return {
        "results": results,
        "client_init_ms": init_time,
    }


def _calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate avg, median, min, max for a list of values."""
    if not values:
        return {"avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    sorted_values = sorted(values)
    n = len(sorted_values)
    median = sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

    return {
        "avg": sum(values) / len(values),
        "median": median,
        "min": min(values),
        "max": max(values),
    }


def _timing_breakdown_summaries(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate timing breakdown stats across all successful runs.

    Returns dict with stats for each timing milestone.
    """
    successful = [r for r in results if r["success"]]
    if not successful:
        return {}

    # Collect values for each metric
    metrics = {
        "time_until_request_ms": [],
        "time_until_response_ms": [],
        "time_until_results_ms": [],
    }

    for result in successful:
        breakdown = result["timing_breakdown"]
        for key in metrics.keys():
            if key in breakdown:
                metrics[key].append(breakdown[key])

    # Calculate stats for each metric
    summaries = {}
    for key, values in metrics.items():
        if values:
            summaries[key] = _calculate_stats(values)

    return summaries


def _write_results(benchmark_data: Dict, output_dir: Path):
    """Write benchmark results to metrics.json and summary.txt."""
    results = benchmark_data["results"]
    client_init_ms = benchmark_data["client_init_ms"]

    successful = [r for r in results if r["success"]]
    failed_count = len(results) - len(successful)

    if not successful:
        print("\n❌ All runs failed; nothing to report")
        return

    # Calculate stats
    ttfb_values = [r["ttfb_ms"] for r in successful]
    ttlb_values = [r["ttlb_ms"] for r in successful]
    bytes_values = [r["total_bytes"] for r in successful]

    ttfb_stats = _calculate_stats(ttfb_values)
    ttlb_stats = _calculate_stats(ttlb_values)
    bytes_stats = _calculate_stats(bytes_values)

    # Get timing breakdown summaries
    timing_summaries = _timing_breakdown_summaries(successful)

    # Prepare metrics.json
    metrics = {
        "test_name": "pipecat_v5_tts",
        "sdk_version": "5.3.0",
        "timestamp": datetime.now().isoformat(),
        "iterations": {
            "total": len(results),
            "successful": len(successful),
            "failed": failed_count,
        },
        "client_init_ms": client_init_ms,
        "ttfb_ms": ttfb_stats,
        "ttlb_ms": ttlb_stats,
        "total_bytes": bytes_stats,
        "timing_breakdown": timing_summaries,
    }

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Prepare summary.txt
    summary_lines = [
        "PipeCat + Deepgram SDK v5.3 TTS Benchmark",
        f"SDK Version: 5.3.0",
        f"Iterations: {len(results)}",
        f"Service initialization (one-time): {client_init_ms:.1f} ms",
        "",
    ]

    # Add individual run details
    for i, result in enumerate(results, 1):
        if result["success"]:
            summary_lines.append(
                f"[{i:02d}] TTFB {result['ttfb_ms']:.1f} ms  "
                f"TTLB {result['ttlb_ms']:.1f} ms  "
                f"bytes {result['total_bytes']}"
            )
        else:
            summary_lines.append(f"[{i:02d}] ERROR: {result['error']}")

    summary_lines.extend([
        "",
        f"Per-Request Timing Breakdown (chronological):",
    ])

    # Add timing breakdown in chronological order with all stats
    timing_order = [
        ("time_until_request_ms", "Time Until Request"),
        ("time_until_response_ms", "Time Until Response"),
    ]

    for key, label in timing_order:
        if key in timing_summaries:
            stats = timing_summaries[key]
            summary_lines.append(
                f"  {label}: "
                f"avg {stats['avg']:.1f} ms  "
                f"median {stats['median']:.1f} ms  "
                f"min {stats['min']:.1f}  "
                f"max {stats['max']:.1f}"
            )

    # Add TTFB and TTLB
    summary_lines.extend([
        f"  Ttfb: "
        f"avg {ttfb_stats['avg']:.1f} ms  "
        f"median {ttfb_stats['median']:.1f} ms  "
        f"min {ttfb_stats['min']:.1f}  "
        f"max {ttfb_stats['max']:.1f}",
        f"  Ttlb: "
        f"avg {ttlb_stats['avg']:.1f} ms  "
        f"median {ttlb_stats['median']:.1f} ms  "
        f"min {ttlb_stats['min']:.1f}  "
        f"max {ttlb_stats['max']:.1f}",
    ])

    # Add time until results
    if "time_until_results_ms" in timing_summaries:
        stats = timing_summaries["time_until_results_ms"]
        summary_lines.append(
            f"  Time Until Results: "
            f"avg {stats['avg']:.1f} ms  "
            f"median {stats['median']:.1f} ms  "
            f"min {stats['min']:.1f}  "
            f"max {stats['max']:.1f}"
        )

    summary_lines.append("")

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nResults written to:")
    print(f"  {metrics_file}")
    print(f"  {summary_file}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PipeCat + Deepgram SDK v5.3 TTS performance"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Number of TTS requests to run (default: 25)",
    )
    args = parser.parse_args()

    # Run benchmark
    benchmark_data = await run_benchmark(args.iterations)

    # Calculate stats for console output
    results = benchmark_data["results"]
    successful = [r for r in results if r["success"]]

    if successful:
        ttfb_values = [r["ttfb_ms"] for r in successful]
        ttlb_values = [r["ttlb_ms"] for r in successful]

        ttfb_stats = _calculate_stats(ttfb_values)
        ttlb_stats = _calculate_stats(ttlb_values)
        timing_breakdown_stats = _timing_breakdown_summaries(successful)

        # Print consolidated timing breakdown (matching v4 format)
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

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("results/pipecat/pipecat_v5_tts") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write results
    _write_results(benchmark_data, output_dir)


if __name__ == "__main__":
    asyncio.run(main())

