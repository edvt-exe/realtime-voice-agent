import asyncio
import websockets

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        print("Connected. Sending dummy audio chunk...")
        await ws.send(b"\x00\x00" * 512)
        response = await asyncio.wait_for(ws.recv(), timeout=5.0)
        print(f"Received {len(response)} bytes back")

asyncio.run(main())