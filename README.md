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

This directory contains the specific examples mentioned by the Daily user, but I was unable to get these to run, instead of spending too much time troubleshooting those, I focused on the tests mentioned above.

### Using this Repository

TBD