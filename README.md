# python-sdk-pipecat-troubleshooting

To troubleshoot Python SDK TTS performance issues with Pipecat.


## Overview

To Troubleshoot the issue reported by Daily with the Deepgram Python SDK and PipeCat.

### Tests

See [tests](./tests/)

### Results

See [results](./results/)


### Using this Repository

1. **Install tooling**
   - Ensure Python 3.12 and `pipenv` are available.
   - Clone this repo and `cd` into it.

2. **Seed environment variables**
   - Export `DEEPGRAM_API_KEY` (and any other keys you need).
   - Optional: copy `examples/pipecat/sample.env` to `.env` if you prefer dotenv loading.

3. **Set up the v4 baseline environment**
   - Run `pipenv install --skip-lock` to install the dependencies listed in `Pipfile` (`deepgram-sdk==4.7.0`, `aiohttp`, etc.).
   - Verify you’re in the correct virtualenv via `pipenv --venv` or `pipenv shell`.

4. **Set up the v5 comparison environment**
   - Copy `Pipfile` → `Pipfile.v5` and bump the dependency to `deepgram-sdk==5.3.0`.
   - Install into a separate virtualenv:
     `PIPENV_PIPFILE=Pipfile.v5 pipenv install --skip-lock`
   - Confirm with `PIPENV_PIPFILE=Pipfile.v5 pipenv --venv`.


## Deepgram SDK Testing

### Test Run 1

Test Run 1 focuses on providing the following metrics for 10 test iterations.

**Summary**
- TTFB: avg, median, min , max
- TTLB: avg, median, min , max

To run tests:

1. **Run the v4 + HTTP benchmarks**
   - SDK v4 streaming clone of Pipecat’s usage:
     `pipenv run python tests/run-1/sdk_v4_tts_benchmark.py --iterations 10`
   - Direct HTTP baseline (Pipecat-style `aiohttp` client):
     `pipenv run python tests/run-1/http_tts_benchmark.py --iterations 10`

2. **Run the v5 benchmarks**
   - Streaming SDK (async generator using v5 client):
     `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/run-1/sdk_v5_stream_tts_benchmark.py --iterations 10`
   - Non-streaming/REST SDK path:
     `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/run-1/sdk_v5_rest_tts_benchmark.py --iterations 10`

----
### Test Run 2

Test Run 2 focuses on providing the following metrics for 25 test iterations.

**Summary**
- TTFB: avg, median, min , max
- TTLB: avg, median, min , max

**Request Send / SDK Preprocessing**
- Starts: When the method is called
- Ends: When the response object/iterator is created (HTTP headers received)
- Captures: Request preparation + network latency + response headers

**Network TTFB**
- Starts: When response object/iterator is created
- Ends: When first data chunk arrives
- Captures: Server processing time + time to generate first audio chunk

**Chunk Processing**
- Starts: After first chunk received
- Ends: After all chunks processed
- Captures: Iterator overhead + any SDK processing of chunks

To run tests:

1. **HTTP Baseline:**
- Deepgram TTS via direct HTTP requests (no SDK)
  `pipenv run python tests/run-2/http_tts_benchmark.py --iterations 25`

2. **v4.7 SDK Tests:**
- Deepgram SDK v4.7 TTS Benchmark with Telemetry enabled
  `pipenv run python tests/run-2/sdk_v4_tts_benchmark.py --iterations 25`
- Deepgram SDK v4.7 TTS Benchmark with Telemetry Disabled
  `pipenv run python tests/run-2/sdk_v4_tts_telemetry_off_benchmark.py --iterations 25`

3. **v5 SDK Tests**
- Deepgram SDK v5 Synchronous generate() TTS Benchmark with Telemetry enabled
  `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/run-2/sdk_v5_sync_tts_benchmark.py --iterations 25`
- Deepgram SDK v5 Synchronous generate() TTS Benchmark with Telemetry disabled
  `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/run-2/sdk_v5_sync_tts_telemetry_off_benchmark.py --iterations 25`

----
### Test Run 3

Test Run 3 focuses on providing the following metrics for 25 test iterations. In this run we skipped v4.7 SDK tests as
we got what we needed from the previous test runs.

1. SDK/Session Initialization (one-time) - Before iterations
2. Time Until Request - When HTTP request is sent
3. Time Until Response - When response headers received
4. TTFB - First data chunk arrives
5. Time Until Response Results - Data processed and ready for use
6. TTLB - Last data chunk received

To run tests:

1. **Test 1: HTTP Baseline**
   - Deepgram TTS via direct HTTP requests (no SDK)
   `pipenv run python tests/run-3/http_tts_benchmark.py --iterations 25`

2. **Test 2: V5 SDK (Telemetry Enabled)**
   - Deepgram SDK v5 Synchronous generate() TTS Benchmark with Telemetry enabled
   `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/run-3/sdk_v5_sync_tts_benchmark.py --iterations 25`

3. **Test 3: V5 SDK (Telemetry Disabled)**
   - Deepgram SDK v5 Synchronous generate() TTS Benchmark with Telemetry disabled
   `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/run-3/sdk_v5_sync_tts_telemetry_off_benchmark.py --iterations 25`

## Pipe Cat Isolation Testing

Testing Pipecat attempts to reproduce the issue reported by Daily. It also allows us to isolate the issue in
Pipecat and determine a possible fix in their library

To run tests:

1. **Test 1: PipeCat + v4.7 SDK**
    - PipeCat + Deepgram SDK v4.7 TTS Benchmark
    `pipenv run python tests/pipecat/pipecat_v4_tts_benchmark.py --iterations 25`

2. **Test 2: PipeCat + v5.3 SDK**
    - PipeCat + Deepgram SDK v5.3 TTS Benchmark
    `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/pipecat/pipecat_v5_tts_benchmark.py --iterations 25`

3. **Test 3: Debugging PipCat + v4.7 SDK**

```json
{"err_code":"INVALID_QUERY_PARAMETER","err_msg":"Failed to deserialize query parameters: invalid value: integer `0`, expected a nonzero u32","request_id":"3b7f3c04-3b3e-438a-93e4-30b178e0198a"}
```
  - PipeCat + Deepgram SDK v4.7 Error troubleshooting
  `pipenv run python debug/debug_v4_error.py`

  - PipeCat + Deepgram SDK v5.3 Error troubleshooting
  `PIPENV_PIPFILE=Pipfile.v5 pipenv run python debug/debug_v5_error.py`

  - PipeCat TTS Isolation test with SDK v4.7
  `pipenv run python debug/test_pipecat_example_tts.py`

  - PipeCat TTS Isolation test with SDK v5.3
  `PIPENV_PIPFILE=Pipfile.v5 pipenv run python debug/test_pipecat_example_tts.py`


4. **Test 4: Testing PipeCat Attempted Fix with 4.7 SDK**

> Note this was the wrong approach see https://github.com/pipecat-ai/pipecat/pull/3054

To run tests:

  - PipCat + Fix + Deepgram SDK v4.7
  `PIPENV_PIPFILE=Pipfile.pipecat pipenv run python debug/test_pipecat_example_tts.py`

  - PipCat + Fix + Deepgram SDK v4.7 Benchmark
  `PIPENV_PIPFILE=Pipfile.pipecat pipenv run python tests/pipecat/pipecat_v4_tts_benchmark.py --iterations 25`

5. **Test 5: Testing PipeCat Fix with 5.3 SDK**
  - PipCat + Fix + Deepgram SDK (poc) v5.3 Benchmark
  `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/pipecat/pipecat_v5_tts_benchmark.py --iterations 25`

## PipeCat End to End Testing

These tests rely on the benchmarks tests available in John Vajda'a forked version of [Pipecat found here](https://github.com/jpvajda/pipecat)
To run these tests you'll need to run one of these `benchmark examples`.

> To run tests you'll need to be in the forked repo!

- https://github.com/jpvajda/pipecat/blob/main/examples/foundational/07c-interruptible-deepgram-http-benchmarked.py
- https://github.com/jpvajda/pipecat/blob/main/examples/foundational/07c-interruptible-deepgram-benchmarked.py

Tests:

1. **Test 1: Http Bechmark test NO SDK**
  - `uv run python 07c-interruptible-deepgram-http-benchmarked.py`
  - disconnect the client to generate the metrics.

2. **Test 2: Deepgram TTS Test w/ 4.7 SDK**
 - `uv run python 07c-interruptible-deepgram-benchmarked.py`
