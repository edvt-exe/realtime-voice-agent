import asyncio
from src.vad.silero_vad import SileroVAD

async def main():
    vad = SileroVAD()
    q: asyncio.Queue[bytes] = asyncio.Queue()

    # Feed 2 seconds of digital silence, should yield no events
    silence_chunk = b"\x00\x00" * 512
    for _ in range(60):
        await q.put(silence_chunk)

    events = []

    async def collect():
        async for event in vad.stream(q):
            events.append(event)

    task = asyncio.create_task(collect())
    await asyncio.sleep(1.0)
    task.cancel()

    print(f"Events on silence: {len(events)}")
    assert len(events) == 0, "Silence should not trigger speech_start"

asyncio.run(main())