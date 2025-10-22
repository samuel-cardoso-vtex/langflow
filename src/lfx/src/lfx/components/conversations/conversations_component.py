# File: langflow/src/lfx/src/lfx/components/conversations/mem0_chat_memory.py

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import httpx

from lfx.base.memory.model import LCChatMemoryComponent
from lfx.inputs.inputs import (
    BoolInput,
    DictInput,
    IntInput,
    MessageTextInput,
    SecretStrInput,
    StrInput,
)
from lfx.io import Output
from lfx.log.logger import logger
from lfx.schema.data import Data


def _hash_idempotency_key(*, role: Optional[str], content: Optional[str], message_id: Optional[str]) -> str:
    """Generate an idempotency key compatible with the client helper."""
    if message_id:
        return f"mid:{message_id}"
    if role is not None and content is not None:
        basis = f"{role}|{(content)[:256]}"
        h = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]
        return f"hash:{h}"
    raise ValueError("Either message_id or both role and content must be provided to build an idempotency key.")


class _SyncHTTP:
    """Small sync HTTP helper that works with httpx or requests seamlessly."""

    def __init__(self, base_url: str, headers: Dict[str, str], timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def post(self, path: str, json_body: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, Any]]:
        headers = {**self.headers, **(extra_headers or {})}
        with httpx.Client(timeout=self.timeout, headers=headers, http2=True) as c:  # type: ignore
            r = c.post(self._url(path), json=json_body)
            content = r.json() if r.content else {}
            return r.status_code, content

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        with httpx.Client(timeout=self.timeout, headers=self.headers, http2=True) as c:  # type: ignore
                r = c.get(self._url(path), params=params or {})
                content = r.json() if r.content else {}
                return r.status_code, content


class ConversationsComponent(LCChatMemoryComponent):
    """
    Stores and retrieves chat messages using the Conversations service.

    Design notes:
    - This component is synchronous to integrate smoothly with Langflow's execution model.
    - We talk directly to the Conversations HTTP API (via httpx/requests).
    - If 'conversation_id' is not provided and 'create_if_missing' is True, a new conversation is created.
    """

    display_name = "Conversations"
    description = "Retrieves and stores chat messages using the Conversations service."
    name = "conversations_chat_memory"
    icon: str = "Conversations"

    # ----------------------------- Inputs (UI) -----------------------------
    inputs = [
        # Service / auth
        StrInput(
            name="base_url",
            display_name="Conversations Base URL",
            info="Base URL of the Conversations API (e.g., http://localhost:8092). Can also come from env CONVERSATIONS_BASE_URL.",
            required=False,
            value=os.getenv("CONVERSATIONS_BASE_URL", ""),
        ),
        SecretStrInput(
            name="token",
            display_name="Client Token",
            info="X-VTEX-CLIENT-TOKEN to authenticate requests.",
            required=True,
        ),
        StrInput(
            name="account",
            display_name="Account",
            info="X-VTEX-CLIENT-ACCOUNT header value.",
            required=True,
        ),
        StrInput(
            name="agent_id",
            display_name="Agent ID",
            info="X-VTEX-USER-AGENT header value.",
            required=True,
        ),

        # Conversation identity / creation
        StrInput(
            name="conversation_id",
            display_name="Conversation ID",
            info="Existing conversation ID (UUIDv7). If empty and 'Create if Missing' is enabled, a new conversation is created.",
            required=False,
        ),
        BoolInput(
            name="create_if_missing",
            display_name="Create if Missing",
            info="If true and no conversation_id is provided, create a new conversation automatically.",
            value=True,
        ),
        StrInput(
            name="summary",
            display_name="Conversation Summary",
            info="Optional human-friendly title/summary when creating a new conversation.",
            required=False,
        ),
        MessageTextInput(
            name="user_id",
            display_name="User ID",
            info="Optional end-user identifier for the conversation.",
            required=False,
        ),

        # Message ingestion
        MessageTextInput(
            name="ingest_message",
            display_name="Message to Ingest",
            info="Message content to append to the conversation.",
        ),
        StrInput(
            name="role",
            display_name="Role",
            info="Role for the message: user | assistant | tool | system.",
            value="user",
            required=False,
        ),
        StrInput(
            name="message_id",
            display_name="Message ID",
            info="Optional explicit message_id (UUIDv7) for idempotent creation.",
            required=False,
        ),
        StrInput(
            name="parent_message_id",
            display_name="Parent Message ID",
            info="Optional parent message id to link threads.",
            required=False,
        ),
        DictInput(
            name="metadata",
            display_name="Metadata",
            info="Optional extra metadata for the message.",
            required=False,
        ),
        DictInput(
            name="tool_input",
            display_name="Tool Input",
            info="Optional tool input payload (stored and possibly encrypted by the service).",
            required=False,
        ),
        DictInput(
            name="tool_output",
            display_name="Tool Output",
            info="Optional tool output payload (stored and possibly encrypted by the service).",
            required=False,
        ),
        DictInput(
            name="tool_calls",
            display_name="Tool Calls",
            info="Assistant planning metadata (OpenAI-compatible 'tool_calls' array).",
            required=False,
        ),
        BoolInput(
            name="auto_idempotency",
            display_name="Auto Idempotency",
            info="Generate an Idempotency-Key automatically from (message_id) or (role, content).",
            value=True,
        ),
        StrInput(
            name="idempotency_key",
            display_name="Idempotency Key",
            info="Override the Idempotency-Key header (advanced).",
            required=False,
        ),

        # Retrieval controls
        IntInput(
            name="list_limit",
            display_name="List Limit",
            info="Max items per page for messages/timeline retrieval.",
            value=100,
        ),
        StrInput(
            name="list_cursor",
            display_name="List Cursor",
            info="Opaque cursor for pagination when listing messages/timeline.",
            required=False,
        ),
    ]

    # ----------------------------- Outputs -----------------------------
    outputs = [
        Output(
            name="conversation",
            display_name="Conversation",
            method="get_or_create_conversation",
        ),
        Output(
            name="append_result",
            display_name="Append Result",
            method="ingest_data",
        ),
        Output(
            name="messages",
            display_name="Messages",
            method="build_messages",
        ),
        Output(
            name="timeline",
            display_name="Timeline",
            method="build_timeline",
        ),
    ]

    # ----------------------------- Helpers -----------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "X-VTEX-CLIENT-TOKEN": self.token,
            "X-VTEX-CLIENT-ACCOUNT": self.account,
            "X-VTEX-USER-AGENT": self.agent_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _client(self) -> _SyncHTTP:
        base_url = (self.base_url or os.getenv("CONVERSATIONS_BASE_URL") or "").strip()
        if not base_url:
            raise RuntimeError("CONVERSATIONS_BASE_URL is required (either input or env).")
        return _SyncHTTP(base_url=base_url, headers=self._headers(), timeout=15.0)

    # ----------------------------- Core ops -----------------------------

    def get_or_create_conversation(self) -> Data:
        """
        Ensure we have a conversation and return its descriptor:
        { "id": "...", "createdAt": "...", ... } (fields may vary depending on the API).
        """
        if self.conversation_id:
            # Try to fetch it for convenience/validation (non-fatal if not found).
            try:
                status, payload = self._client().get(f"/v1/conversations/{self.conversation_id}")
                if 200 <= status < 300:
                    return payload
                logger.warning("Conversation %s not found (status=%s).", self.conversation_id, status)
            except Exception:
                logger.exception("Failed to fetch conversation %s", self.conversation_id)
            # Return at least the id so downstream nodes can continue.
            return {"id": self.conversation_id}

        if not self.create_if_missing:
            logger.warning("No conversation_id provided and create_if_missing is False.")
            return {}

        body: Dict[str, Any] = {}
        if self.summary:
            body["summary"] = self.summary
        # The service ignores/derives account/agent from headers, but accepting both is harmless
        try:
            status, payload = self._client().post("/v1/conversations", json_body=body)
            if status in (200, 201):
                # Expected: {"id": "...", "createdAt": "..."}
                self.conversation_id = payload.get("id", self.conversation_id)
                return payload
            # 409 means conflict when an explicit id exists; we are not sending explicit ids here.
            logger.error("Create conversation failed: status=%s payload=%s", status, payload)
            raise RuntimeError("Failed to create conversation.")
        except Exception as e:
            logger.exception("Exception while creating conversation: %s", e)
            raise

    def _resolve_conversation_id(self) -> str:
        conv = self.get_or_create_conversation()
        cid = self.conversation_id or conv.get("id")
        if not cid:
            raise RuntimeError("Unable to resolve conversation_id.")
        return cid

    def _compute_idempotency(self, role: Optional[str], content: Optional[str]) -> Optional[str]:
        if self.idempotency_key:
            return self.idempotency_key
        if not self.auto_idempotency:
            return None
        try:
            return _hash_idempotency_key(role=role, content=content, message_id=self.message_id)
        except Exception as e:
            logger.warning("Could not compute idempotency key automatically: %s", e)
            return None

    # ----------------------------- Outputs (methods) -----------------------------

    def ingest_data(self) -> Data:
        """
        Append a message when 'ingest_message' is provided.
        Returns the API response (e.g., {"messageId": "...", "createdAt": "..."}).
        If no message is provided, returns an empty dict.
        """
        if not self.ingest_message:
            logger.info("No 'ingest_message' provided; skipping append. Returning empty result.")
            return {}

        cid = self._resolve_conversation_id()
        role = (self.role or "user").lower().strip()
        if role not in {"user", "assistant", "tool", "system"}:
            raise ValueError("Invalid role. Must be one of: user | assistant | tool | system")

        message: Dict[str, Any] = {
            "role": role,
            "content": self.ingest_message or "",
        }
        if self.message_id:
            message["message_id"] = self.message_id
        if self.parent_message_id:
            message["parent_message_id"] = self.parent_message_id
        if self.metadata:
            message["metadata"] = dict(self.metadata)
        if self.tool_input is not None:
            message["tool_input"] = self.tool_input
        if self.tool_output is not None:
            message["tool_output"] = self.tool_output
        if self.tool_calls:
            # Accept dict or list; if dict, forward as-is; if string, try to parse JSON.
            tc = self.tool_calls
            if isinstance(tc, str):
                try:
                    tc = json.loads(tc)
                except Exception:
                    pass
            message["tool_calls"] = tc

        # Build headers with optional Idempotency-Key
        idemp = self._compute_idempotency(role=role, content=self.ingest_message or "")
        extra_headers = {"Idempotency-Key": idemp} if idemp else {}

        try:
            status, payload = self._client().post(f"/v1/conversations/{cid}/messages", json_body=message, extra_headers=extra_headers)
            if 200 <= status < 300:
                return payload
            logger.error("Append message failed: status=%s payload=%s", status, payload)
            raise RuntimeError("Failed to append message.")
        except Exception as e:
            logger.exception("Exception while appending message: %s", e)
            raise

    def build_messages(self) -> Data:
        """
        Retrieve raw messages for the conversation.
        Returns {"items": [...], "next_cursor": "..."} with snake_case keys.
        """
        cid = self._resolve_conversation_id()
        params: Dict[str, Any] = {}
        if self.list_limit:
            params["limit"] = int(self.list_limit)
        if self.list_cursor:
            params["cursor"] = self.list_cursor

        try:
            status, payload = self._client().get(f"/v1/conversations/{cid}/messages", params=params)
            if 200 <= status < 300:
                # Normalize next_cursor across possible API shapes (defensive)
                if "nextCursor" in payload and "next_cursor" not in payload:
                    payload["next_cursor"] = payload.get("nextCursor")
                return payload
            logger.error("List messages failed: status=%s payload=%s", status, payload)
            raise RuntimeError("Failed to list messages.")
        except Exception as e:
            logger.exception("Exception while listing messages: %s", e)
            raise

    def build_timeline(self) -> Data:
        """
        Retrieve safe UI projection (timeline) for the conversation.
        Returns {"items": [...], "next_cursor": "..."}.
        """
        cid = self._resolve_conversation_id()
        params: Dict[str, Any] = {}
        if self.list_limit:
            params["limit"] = int(self.list_limit)
        if self.list_cursor:
            params["cursor"] = self.list_cursor

        try:
            status, payload = self._client().get(f"/v1/conversations/{cid}/timeline", params=params)
            if 200 <= status < 300:
                # Normalize next_cursor across possible API shapes (defensive)
                if "nextCursor" in payload and "next_cursor" not in payload:
                    payload["next_cursor"] = payload.get("nextCursor")
                return payload
            logger.error("Get timeline failed: status=%s payload=%s", status, payload)
            raise RuntimeError("Failed to get timeline.")
        except Exception as e:
            logger.exception("Exception while getting timeline: %s", e)
            raise
