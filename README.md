# python-sdk-pipecat-troubleshooting

To troubleshoot Python SDK TTS performance issues with Pipecat.


## Overview

To Troubleshoot the issue reported by Daily with the Deepgram Python SDK and PipeCat.

### Tests

See [tests](./tests/)

### Results

See [results](./results/)

### Examples

See [examples](./examples/)

This directory contains the specific PipeCat examples [1](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07c-interruptible-deepgram-http.py) & [2](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07c-interruptible-deepgram.py) mentioned by the Daily user, but I was unable to get these to run, instead of spending too much time troubleshooting those, I focused on the tests mentioned above.

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


### TEST RUN 2

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


  ### Test Run 3

Test Run 3 focuses on providing the following metrics for 25 test iterations. In this run we skipped v4.7 SDK tests as
we got what we needed from the previous test runs.

1. SDK/Session Initialization (one-time) - Before iterations
2. Time Until Request - When HTTP request is sent
3. Time Until Response - When response headers received
4. TTFB - First data chunk arrives
5. Time Until Response Results - Data processed and ready for use
6. TTLB - Last data chunk received


# Test 1: HTTP Baseline
- Deepgram TTS via direct HTTP requests (no SDK)
`pipenv run python tests/run-3/http_tts_benchmark.py --iterations 25`
- Deepgram SDK v5 Synchronous generate() TTS Benchmark with Telemetry enabled
# Test 2: V5 SDK (Telemetry Enabled)
`PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/run-3/sdk_v5_sync_tts_benchmark.py --iterations 25`
- Deepgram SDK v5 Synchronous generate() TTS Benchmark with Telemetry disabled
# Test 3: V5 SDK (Telemetry Disabled)
`PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/run-3/sdk_v5_sync_tts_telemetry_off_benchmark.py --iterations 25`