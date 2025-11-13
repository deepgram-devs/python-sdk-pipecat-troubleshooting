"""
Modified DeepgramTTSService for SDK v5.3

This is a copy of PipeCat's DeepgramTTSService modified to use Deepgram SDK v5.3 API.
Original: pipecat/src/pipecat/services/deepgram/tts.py

Key changes:
- Uses v5.3 API: client.speak.v1.audio.generate()
- Wraps synchronous generator as async generator
- Maintains same interface and frame types for benchmarking
"""

import asyncio
from typing import AsyncGenerator

from deepgram import DeepgramClient

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class DeepgramTTSServiceV5(TTSService):
    """Modified DeepgramTTSService using SDK v5.3"""

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-2-andromeda-en",
        sample_rate: int = 16000,
        encoding: str = "linear16",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._voice = voice
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._deepgram_client = DeepgramClient(api_key=self._api_key)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate TTS audio frames using Deepgram SDK v5.3

        This wraps the synchronous generator from v5.3's generate() method
        as an async generator to match PipeCat's expected interface.
        """
        try:
            # Emit started frame
            yield TTSStartedFrame()

            # Call SDK v5.3 generate() - returns synchronous generator
            # This is lazy evaluation - HTTP call happens during first iteration
            response = self._deepgram_client.speak.v1.audio.generate(
                text=text,
                model=self._voice,
                encoding=self._encoding,
                sample_rate=self._sample_rate,
                container="none",
            )

            # Iterate through audio chunks
            # The synchronous generator needs to be wrapped for async context
            for chunk in response:
                if chunk:
                    frame = TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                    yield frame
                    # Yield control to event loop
                    await asyncio.sleep(0)

            # Emit stopped frame
            yield TTSStoppedFrame()

        except Exception as e:
            # Log error but don't crash
            print(f"TTS error: {e}")
            yield TTSStoppedFrame()

