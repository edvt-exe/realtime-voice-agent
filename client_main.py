import asyncio
import argparse
import structlog

from src.client.ws_client import VoiceClient

structlog.configure(processors=[structlog.processors.JSONRenderer()])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Echo voice agent client")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    return parser.parse_args()

async def main() -> None:
    args = parse_args()
    client = VoiceClient(server_url=args.server)
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDisconnected.")