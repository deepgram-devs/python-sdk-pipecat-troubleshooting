"""
PipeCat + Deepgram SDK v4.7 TTS Benchmark with Granular Timing

This script benchmarks PipeCat's actual DeepgramTTSService (imported from PipeCat repo)
with SDK v4.7, measuring 6 granular timing milestones to identify performance characteristics.

Uses the exact same timing methodology as run-3 for apples-to-apples comparison.
"""

import argparse
import asyncio
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import actual PipeCat service from the forked repo
sys.path.insert(0, '/Users/johnvajda/Documents/Github/sdk-testing/Python/pipecat/src')
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame


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


async def _run_once(
    service: DeepgramTTSService,
    text: str,
) -> Dict[str, Any]:
    """Execute one TTS request through PipeCat and measure granular timing at 6 milestones.

    PipeCat's DeepgramTTSService uses SDK v4.7 API:
    - speak.asyncrest.v("1").stream_raw()
    - Returns async iterator of audio chunks

    Milestones (chronological):
    1. Start
    2. Time until request (when HTTP request is sent via SDK)
    3. Time until response (when response headers received)
    4. TTFB (first audio frame arrives)
    5. Time until response results (all audio processed)
    6. TTLB (last audio frame received)
    """
    start = time.perf_counter()
    timing_breakdown = {}

    ttfb: Optional[float] = None
    total_bytes = 0
    frame_count = 0
    first_frame_time: Optional[float] = None

    try:
        # Call PipeCat's TTS service
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
        results_ready = time.perf_counter()
        timing_breakdown["time_until_results_ms"] = (results_ready - start) * 1000.0

        return {
            "ttfb_ms": ttfb if ttfb is not None else ttlb,
            "ttlb_ms": ttlb,
            "bytes": float(total_bytes),
            "frame_count": frame_count,
            "timing_breakdown": timing_breakdown,
        }
    except Exception as exc:
        raise RuntimeError(f"PipeCat TTS request failed: {exc}") from exc


def _write_results(
    metrics: List[Dict[str, Any]],
    summary: Dict[str, Any],
    timing_summary: Dict[str, Dict[str, float]],
    service_init_ms: float,
    args: argparse.Namespace,
) -> None:
    """Write metrics and summary to timestamped output directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path(args.output_dir) / "pipecat" / "pipecat_v4_tts" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "benchmark": "pipecat_v4_tts",
        "generated_at": timestamp,
        "iterations": len(metrics),
        "service_init_ms": service_init_ms,
        "sdk_version": "4.7.0",
        "parameters": {
            "text": args.text,
            "voice": args.voice,
            "sample_rate": args.sample_rate,
            "encoding": args.encoding,
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
        handle.write("PipeCat + Deepgram SDK v4.7 TTS Benchmark\n")
        handle.write(f"SDK Version: 4.7.0\n")
        handle.write(f"Iterations: {len(metrics)}\n")
        handle.write(f"Service initialization (one-time): {service_init_ms:.1f} ms\n\n")

        # Individual run results
        for idx, metric in enumerate(metrics, start=1):
            handle.write(
                f"[{idx:02d}] TTFB {metric['ttfb_ms']:.1f} ms  "
                f"TTLB {metric['ttlb_ms']:.1f} ms  bytes {int(metric['bytes'])}  "
                f"frames {metric['frame_count']}\n"
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


async def _run(args: argparse.Namespace) -> None:
    """Execute benchmark suite."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY is required", file=sys.stderr)
        sys.exit(1)

    # Milestone 1: Measure service initialization (one-time cost)
    print("Initializing PipeCat DeepgramTTSService...")
    service_init_start = time.perf_counter()
    service = DeepgramTTSService(
        api_key=api_key,
        voice=args.voice,
        sample_rate=args.sample_rate,
        encoding=args.encoding,
    )
    service_init_ms = (time.perf_counter() - service_init_start) * 1000.0
    print(f"Service initialization: {service_init_ms:.1f} ms\n")

    metrics: List[Dict[str, Any]] = []

    for idx in range(1, args.iterations + 1):
        try:
            result = await _run_once(service, args.text)
            metrics.append(result)
            print(
                f"[{idx:02d}] TTFB {result['ttfb_ms']:.1f} ms  "
                f"TTLB {result['ttlb_ms']:.1f} ms  bytes {int(result['bytes'])}  "
                f"frames {result['frame_count']}"
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
    _write_results(metrics, summary, timing_breakdown_stats, service_init_ms, args)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark PipeCat's DeepgramTTSService with SDK v4.7 and granular timing instrumentation."
    )
    parser.add_argument("--text", default="Hello, this is a test of the Deepgram text to speech API using PipeCat with SDK v4.7.")
    parser.add_argument("--voice", default="aura-2-andromeda-en")
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
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()

