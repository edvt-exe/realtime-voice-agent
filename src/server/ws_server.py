from __future__ import annotations
import asyncio
import uuid
import structlog
import websockets
from websockets.asyncio.server import ServerConnection

from src.config import settings
from src.server.session import build_orchestrator

log = structlog.get_logger()

async def handle_session(ws: ServerConnection) -> None:
    session_id = str(uuid.uuid4())
    log.info("session_started", session_id=session_id)

    orch = build_orchestrator()
    pipeline_task = asyncio.create_task(orch.run(), name=f"pipeline-{session_id}")
    sender_task = asyncio.create_task(_forward_audio_out(orch, ws), name=f"sender-{session_id}")

    try:
        async for message in ws:
            if isinstance(message, bytes):
                await orch.push_audio_chunk(message)
            else:
                log.warning("unexpected_text_message", session_id=session_id, message=message)

    except websockets.ConnectionClosed:
        log.info("session_disconnected", session_id=session_id)

    except Exception as e:
        log.error("session_error", session_id=session_id, error=str(e))

    finally:
        pipeline_task.cancel()
        sender_task.cancel()
        await asyncio.gather(pipeline_task, sender_task, return_exceptions=True)
        log.info("session_cleaned_up", session_id=session_id)


async def _forward_audio_out(orch, ws: ServerConnection) -> None:
    # Continuously forwards synthesised audio chunks back to the client
    while True:
        chunk = await orch.audio_out_q.get()
        if chunk is None:
            continue
        try:
            await ws.send(chunk)
        except websockets.ConnectionClosed:
            break


async def main() -> None:
    async with websockets.serve(handle_session, settings.server_host, settings.server_port, max_size=None,):
        log.info("server_started", host=settings.server_host, port=settings.server_port)
        await asyncio.Future()