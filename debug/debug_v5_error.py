#!/usr/bin/env python3
"""Debug script to test PipeCat with SDK v5.3.
Expected: ImportError showing PipeCat incompatibility with SDK v5.3
"""

import asyncio
import os

async def main():
    try:
        # Attempt to import PipeCat (will fail with v5.3 SDK)
        from pipecat.services.deepgram.tts import DeepgramTTSService
        from pipecat.frames.frames import TTSAudioRawFrame

        print("PipeCat import succeeded - testing with v5.3 SDK")

        # Same parameters as our other tests
        service = DeepgramTTSService(
            api_key=os.getenv('DEEPGRAM_API_KEY'),
            voice='aura-2-andromeda-en',
            sample_rate=16000,  # Omit to test the bug scenario
            encoding='linear16',
        )

        frames = service.run_tts('Testing Deepgram SDK v5.3')

        total_bytes = 0
        async for frame in frames:
            if isinstance(frame, TTSAudioRawFrame):
                total_bytes += len(frame.audio)
                if total_bytes <= 200:
                    print(f"First chunk bytes: {len(frame.audio)}")

        print(f"\nTotal bytes received: {total_bytes}")

    except ImportError as e:
        print(f"ImportError (EXPECTED with v5.3 SDK):")
        print(f"  {e}")
        print(f"\nConclusion: PipeCat 0.0.94 is incompatible with SDK v5.3")

if __name__ == '__main__':
    asyncio.run(main())

