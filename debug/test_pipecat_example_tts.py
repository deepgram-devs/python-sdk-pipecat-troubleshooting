#!/usr/bin/env python3
"""
Simplified test mimicking PipeCat's example setup.
Tests if the TTS service works the way PipeCat examples use it.
"""

import asyncio
import os
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.frames.frames import TTSAudioRawFrame

async def main():
    # Initialize exactly like the example does (line 66)
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-2-andromeda-en",
        # Test different params combinations
        sample_rate=16000,
        encoding="linear16"
    )

    print(f"Service created:")
    print(f"  sample_rate: {tts.sample_rate}")
    print(f"  _init_sample_rate: {tts._init_sample_rate}")
    print(f"  voice: {tts._voice_id}")
    print(f"  encoding: {tts._settings.get('encoding')}")
    print()

    # Try to generate TTS
    frames = tts.run_tts("Testing PipeCat example configuration")

    total_bytes = 0
    frame_count = 0
    async for frame in frames:
        if isinstance(frame, TTSAudioRawFrame):
            frame_count += 1
            total_bytes += len(frame.audio)
            if frame_count == 1:
                print(f"First audio frame: {len(frame.audio)} bytes")
                # Check if it's an error (JSON starts with {{)
                if frame.audio[:1] == b'{':
                    print(f"ERROR: {frame.audio.decode('utf-8', errors='replace')}")
                else:
                    print(f"SUCCESS: Got audio data")

    print(f"\nTotal: {frame_count} frames, {total_bytes} bytes")

if __name__ == '__main__':
    asyncio.run(main())

