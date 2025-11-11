# python-sdk-pipecat-troubleshooting

To troubleshoot v5Python SDK TTS performance issues with Pipecat.


## Overview

To Troubleshoot the issue reported by Daily with the Deepgram Python SDK and PipeCat.

### Tests

See [tests](./tests/)

These benchmark scripts skip Pipecat entirely. They call the SDK’s speak.`asyncrest.v("1").stream_raw()` (mirroring Pipecat’s [TTS implementation](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/deepgram/tts.py)) and the raw HTTP endpoint directly, so the measurements isolate Deepgram v4.7 vs REST without the surrounding Pipecat pipeline.

### Results

See [results](./results/)

The results directory contains the test results run with a summary of TTFB (Time To First Byte) and TTLB (Time To Last Byte).


### Examples

See [examples](./examples/)

This directory contains the specific PipeCat examples [1](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07c-interruptible-deepgram-http.py) & [2](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07c-interruptible-deepgram.py) mentioned by the Daily user, but I was unable to get these to run, instead of spending too much time troubleshooting those, I focused on the tests mentioned above.

### Using this Repository

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

4. **Run the v4 + HTTP benchmarks**
   - SDK v4 streaming clone of Pipecat’s usage:
     `pipenv run python tests/sdk_v4_tts_benchmark.py --iterations 10`
   - Direct HTTP baseline (Pipecat-style `aiohttp` client):
     `pipenv run python tests/http_tts_benchmark.py --iterations 10`
   - Each run writes raw metrics and summaries to `results/sdk_v4_tts/<timestamp>/` and `results/http_tts/<timestamp>/`.

5. **Set up the v5 comparison environment**
   - Copy `Pipfile` → `Pipfile.v5` and bump the dependency to `deepgram-sdk==5.3.0`.
   - Install into a separate virtualenv:
     `PIPENV_PIPFILE=Pipfile.v5 pipenv install --skip-lock`
   - Confirm with `PIPENV_PIPFILE=Pipfile.v5 pipenv --venv`.

6. **Run the v5 benchmarks**
   - Streaming SDK (async generator using v5 client):
     `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/sdk_v5_stream_tts_benchmark.py --iterations 10`
   - Non-streaming/REST SDK path:
     `PIPENV_PIPFILE=Pipfile.v5 pipenv run python tests/sdk_v5_rest_tts_benchmark.py --iterations 10`
   - Outputs land in `results/sdk_v5_stream_tts/<timestamp>/` and `results/sdk_v5_rest_tts/<timestamp>/`.

7. **Compare results**
   - Each summary reports TTFB, TTLB, and byte counts.
   - The four latest summaries in `results/` are what we use to contrast v4 SDK, v5 SDK (stream + REST), and direct HTTP.

8. **Optional follow-ups**
   - Fork Pipecat (example: https://github.com/jpvajda/pipecat) and adapt its Deepgram integration to the v5 streaming SDK.
   - Re-run the above scripts after Pipecat switches to v5 to capture the before/after delta.
   - Use the metrics to populate customer updates or migration guidance.