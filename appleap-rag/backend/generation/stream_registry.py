"""In-memory registry of active streaming generations.

The HTTP request that calls /query may disconnect at any time (user navigates,
laptop sleeps, network drops). Generation must continue server-side regardless
so the answer is preserved and so other clients (the same user opening the
conversation in another tab, or coming back later) can reconnect to the
live stream.

Single-process, in-memory. If the backend restarts, all in-flight streams
are lost; `lifespan` sweeps any rows still marked status='streaming' to
status='error' on startup so clients see a clear signal instead of waiting
forever.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    """One in-flight assistant-message generation.

    `subscribers` is a list of per-connection queues. Each new subscriber
    gets a fresh queue; on subscribe the producer's accumulated_text is
    flushed to that queue first, then the queue receives any new deltas.

    `status` transitions: 'streaming' -> 'completed' | 'error'. Once
    terminal, the StreamState is left in the registry briefly so any
    in-flight subscribers can drain, then removed.
    """
    message_id: str
    conversation_id: str
    accumulated_text: str = ""
    status: str = "streaming"  # 'streaming' | 'completed' | 'error'
    error_text: Optional[str] = None
    sources: list[dict] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


# Module-level singleton. Keyed by message_id (one in-flight assistant
# message per /query call). A given conversation can technically have
# multiple in-flight messages if the user fires off /query before the
# previous one finished, but in practice the UI gates that.
_active: dict[str, StreamState] = {}


def register(message_id: str, conversation_id: str, sources: list[dict]) -> StreamState:
    """Create + register a new StreamState. Called by /query before generation
    starts. Returns the state so the producer can push deltas to it."""
    state = StreamState(
        message_id=message_id,
        conversation_id=conversation_id,
        sources=sources,
    )
    _active[message_id] = state
    return state


def get(message_id: str) -> Optional[StreamState]:
    return _active.get(message_id)


def get_streaming_for_conversation(conversation_id: str) -> Optional[StreamState]:
    """Return the first in-flight stream for this conversation, or None.
    Used by GET /conversations/{id} to expose `streaming_message_id`."""
    for state in _active.values():
        if state.conversation_id == conversation_id and state.status == "streaming":
            return state
    return None


def push_delta(state: StreamState, delta: str) -> None:
    """Append to accumulated text and fan out to all subscribers."""
    state.accumulated_text += delta
    for q in state.subscribers:
        q.put_nowait({"type": "delta", "text": delta})


def finish(state: StreamState, error: Optional[str] = None) -> None:
    """Mark stream as completed (or errored) and wake all subscribers."""
    if error is not None:
        state.status = "error"
        state.error_text = error
    else:
        state.status = "completed"
    for q in state.subscribers:
        q.put_nowait({"type": "done", "error": error})
    state.done_event.set()


def subscribe(state: StreamState) -> asyncio.Queue:
    """Register a new subscriber queue.

    Always pair with a snapshot of `state.accumulated_text` taken atomically
    via `subscribe_with_snapshot()`; calling this directly skips the snapshot
    and risks losing deltas pushed before subscribe.

    If the state is already terminal (completed/error), a synthetic done
    event is queued so the subscriber doesn't hang waiting for an event
    that already fired."""
    q: asyncio.Queue = asyncio.Queue()
    state.subscribers.append(q)
    if state.status != "streaming":
        q.put_nowait({
            "type": "done",
            "error": state.error_text if state.status == "error" else None,
        })
    return q


def subscribe_with_snapshot(state: StreamState) -> tuple[str, str, asyncio.Queue]:
    """Atomically snapshot accumulated_text + status, then register a
    subscriber queue. Single-threaded asyncio guarantees no push_delta or
    finish can interleave between these two sync calls, so the snapshot
    contains every delta pushed before subscribe and the queue receives
    every delta pushed after.

    Returns: (accumulated_text, status, queue)
    """
    text = state.accumulated_text
    status = state.status
    q = subscribe(state)
    return text, status, q


def unsubscribe(state: StreamState, q: asyncio.Queue) -> None:
    try:
        state.subscribers.remove(q)
    except ValueError:
        pass


def remove(message_id: str) -> None:
    """Drop a terminal stream from the registry. Called after subscribers
    have drained or after a grace period."""
    _active.pop(message_id, None)
