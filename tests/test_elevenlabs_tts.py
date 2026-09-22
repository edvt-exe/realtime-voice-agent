import asyncio
from src.tts.elevenlabs_tts import ElevenLabsTTS

async def main():
    tts = ElevenLabsTTS()
    audio_bytes = bytearray()

    async for chunk in tts.stream("Hello, this is a test of the text to speech system."):
        audio_bytes.extend(chunk)

    with open("tests/output_test.mp3", "wb") as f:
        f.write(audio_bytes)

    print(f"Wrote {len(audio_bytes)} bytes to tests/output_test.mp3")

asyncio.run(main())