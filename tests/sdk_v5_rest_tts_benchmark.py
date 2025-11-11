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
    values = [m[key] for m in metrics]
    return {
        "min": min(values),
        "max": max(values),
        "avg": statistics.mean(values),
        "median": statistics.median(values),
    }


def _run_once(
    client: DeepgramClient,
    args: argparse.Namespace,
    request_options: Optional[Dict[str, int]],
) -> Dict[str, float]:
    start = time.perf_counter()
    ttfb: float | None = None
    total_bytes = 0

    iterator = client.speak.v1.audio.generate(
        text=args.text,
        model=args.model,
        encoding=args.encoding,
        sample_rate=args.sample_rate,
        container="none",
        request_options=request_options,
    )

    for chunk in iterator:
        if not chunk:
            continue
        if ttfb is None:
            ttfb = (time.perf_counter() - start) * 1000.0
        total_bytes += len(chunk)

    ttlb = (time.perf_counter() - start) * 1000.0
    return {
        "ttfb_ms": ttfb if ttfb is not None else ttlb,
        "ttlb_ms": ttlb,
        "bytes": float(total_bytes),
    }


def _write_results(
    metrics: List[Dict[str, float]],
    summary: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path(args.output_dir) / "sdk_v5_rest_tts" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "benchmark": "sdk_v5_rest_tts",
        "generated_at": timestamp,
        "iterations": len(metrics),
        "parameters": {
            "text": args.text,
            "model": args.model,
            "encoding": args.encoding,
            "sample_rate": args.sample_rate,
            "chunk_size": args.chunk_size,
        },
        "runs": metrics,
        "summary": summary,
    }

    with (base_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with (base_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Deepgram SDK v5 REST benchmark\n")
        handle.write(
            f"Iterations: {len(metrics)}\n"
            f"TTFB avg {summary['ttfb']['avg']:.1f} ms | "
            f"median {summary['ttfb']['median']:.1f} ms | "
            f"min {summary['ttfb']['min']:.1f} ms | "
            f"max {summary['ttfb']['max']:.1f} ms\n"
        )
        handle.write(
            f"TTLB avg {summary['ttlb']['avg']:.1f} ms | "
            f"median {summary['ttlb']['median']:.1f} ms | "
            f"min {summary['ttlb']['min']:.1f} ms | "
            f"max {summary['ttlb']['max']:.1f} ms\n"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Deepgram TTS via SDK v5 REST (non-streaming) API.",
    )
    parser.add_argument("--text", default="Testing Deepgram SDK v5 REST TTS.")
    parser.add_argument("--model", default="aura-2-andromeda-en")
    parser.add_argument("--encoding", default="linear16")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Chunk size in bytes for streaming audio; set 0 to use the httpx default.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where benchmark results will be written.",
    )
    return parser.parse_args()


def main() -> None:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY is required", file=sys.stderr)
        sys.exit(1)

    client = DeepgramClient(api_key=api_key)

    args = _parse_args()
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else None
    request_options: Optional[Dict[str, int]] = None
    if chunk_size is not None:
        request_options = {"chunk_size": chunk_size}

    metrics: List[Dict[str, float]] = []
    for idx in range(1, args.iterations + 1):
        try:
            result = _run_once(client, args, request_options)
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

    ttfb_stats = _summaries(metrics, "ttfb_ms")
    ttlb_stats = _summaries(metrics, "ttlb_ms")

    print("\nSummary (Deepgram SDK v5 REST)")
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

    summary = {"ttfb": ttfb_stats, "ttlb": ttlb_stats}
    _write_results(metrics, summary, args)


if __name__ == "__main__":
    main()
