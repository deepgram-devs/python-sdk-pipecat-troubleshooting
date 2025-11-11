"""
Verification script to validate V5 timing measurements with detailed logging.
Runs a single request with verbose output to confirm timing accuracy.
"""

import os
import sys
import time
from deepgram import DeepgramClient


def main():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY is required", file=sys.stderr)
        sys.exit(1)

    print("Initializing V5 client...")
    init_start = time.perf_counter()
    client = DeepgramClient(api_key=api_key, telemetry_opt_out=False)
    init_end = time.perf_counter()
    print(f"✓ Client initialized in {(init_end - init_start) * 1000:.3f}ms\n")

    text = "Testing Deepgram SDK streaming TTS."
    model = "aura-2-andromeda-en"
    encoding = "linear16"
    sample_rate = 16000

    print("=" * 60)
    print("DETAILED TIMING TRACE")
    print("=" * 60)

    # Start overall timer
    overall_start = time.perf_counter()
    print(f"[T+0.000ms] Starting request...")

    # Phase 1: Iterator creation
    iter_start = time.perf_counter()
    response = client.speak.v1.audio.generate(
        text=text,
        model=model,
        encoding=encoding,
        sample_rate=sample_rate,
        container="none",
    )
    iter_end = time.perf_counter()
    iter_duration = (iter_end - overall_start) * 1000
    print(f"[T+{iter_duration:.3f}ms] Iterator created (no HTTP request yet)")

    # Phase 2: First iteration (triggers HTTP request)
    print(f"[T+{iter_duration:.3f}ms] Starting iteration (HTTP request will execute now)...")

    chunk_count = 0
    total_bytes = 0
    first_chunk_time = None
    last_chunk_time = None

    for chunk in response:
        if not chunk:
            continue

        chunk_count += 1
        total_bytes += len(chunk)
        current_time = (time.perf_counter() - overall_start) * 1000

        if chunk_count == 1:
            first_chunk_time = current_time
            print(f"[T+{current_time:.3f}ms] ✓ FIRST CHUNK received ({len(chunk)} bytes)")
            print(f"    → HTTP request execution took: {current_time - iter_duration:.3f}ms")
            break  # Stop after first chunk to measure separately

    # Phase 3: Remaining chunks
    print(f"[T+{first_chunk_time:.3f}ms] Processing remaining chunks...")
    remaining_start = time.perf_counter()

    for chunk in response:
        if not chunk:
            continue
        chunk_count += 1
        total_bytes += len(chunk)

    overall_end = time.perf_counter()
    last_chunk_time = (overall_end - overall_start) * 1000
    remaining_duration = (overall_end - remaining_start) * 1000

    print(f"[T+{last_chunk_time:.3f}ms] ✓ ALL CHUNKS received ({chunk_count} total)")

    # Summary
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN SUMMARY")
    print("=" * 60)
    print(f"1. Iterator Creation:  {iter_duration:.3f}ms")
    print(f"2. First Chunk:        {first_chunk_time - iter_duration:.3f}ms")
    print(f"3. Remaining Chunks:   {remaining_duration:.3f}ms")
    print(f"   {'─' * 58}")
    print(f"   Total (calculated):  {iter_duration + (first_chunk_time - iter_duration) + remaining_duration:.3f}ms")
    print(f"   Total (measured):    {last_chunk_time:.3f}ms")
    print(f"\n   Difference: {abs(last_chunk_time - (iter_duration + (first_chunk_time - iter_duration) + remaining_duration)):.3f}ms")
    print(f"\nTotal chunks: {chunk_count}")
    print(f"Total bytes: {total_bytes:,}")
    print(f"\nTTFB: {first_chunk_time:.3f}ms")
    print(f"TTLB: {last_chunk_time:.3f}ms")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    sum_of_parts = iter_duration + (first_chunk_time - iter_duration) + remaining_duration
    if abs(last_chunk_time - sum_of_parts) < 0.1:
        print("✅ PASS: Timing components add up correctly")
    else:
        print(f"❌ FAIL: Timing mismatch of {abs(last_chunk_time - sum_of_parts):.3f}ms")

    # Check that iterator creation is minimal
    if iter_duration < 10:
        print("✅ PASS: Iterator creation is minimal (< 10ms)")
    else:
        print(f"⚠️  WARNING: Iterator creation took {iter_duration:.3f}ms (expected < 10ms)")

    # Check that TTFB is reasonable
    if 100 < first_chunk_time < 500:
        print(f"✅ PASS: TTFB is reasonable ({first_chunk_time:.3f}ms)")
    else:
        print(f"⚠️  WARNING: TTFB of {first_chunk_time:.3f}ms is outside expected range")


if __name__ == "__main__":
    main()

