#!/usr/bin/env python3
"""Debug script to capture the full 193-byte error response from v4.7 SDK. with PipeCat"""

import asyncio
import os
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.frames.frames import TTSAudioRawFrame

async def main():
    service = DeepgramTTSService(
        api_key=os.getenv('DEEPGRAM_API_KEY'),
        voice='aura-2-andromeda-en',
        #Testing with sample_rate = 0, or None or not set all reveals similar errors:
        #PipeCat sends sample_rate=0 to API, Deepgram API Returns error:
        #"err_code":"INVALID_QUERY_PARAMETER","err_msg":"Failed to deserialize query parameters: invalid value: integer `0`, expected a nonzero u32"
        sample_rate=16000,
        encoding='linear16',
    )

    frames = service.run_tts('Testing Deepgram SDK streaming TTS.')

    async for frame in frames:
        if isinstance(frame, TTSAudioRawFrame):
            data = frame.audio
            print(f"Audio frame bytes: {len(data)}")
            print(f"Full content:")
            print(data.decode('utf-8', errors='replace'))
            break

if __name__ == '__main__':
    asyncio.run(main())

