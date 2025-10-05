"""SQLite-based session implementation for conversation history storage."""

import json
import sqlite3
import aiosqlite
from typing import Optional, List
from pathlib import Path
from datetime import datetime


class SessionABC:
    """Abstract base class for session implementations."""

    async def get_items(self, limit: Optional[int] = None) -> List[dict]:
        """Retrieve conversation history."""
        raise NotImplementedError

    async def add_items(self, items: List[dict]) -> None:
        """Add items to conversation history."""
        raise NotImplementedError

    async def pop_item(self) -> Optional[dict]:
        """Remove and return the most recent item."""
        raise NotImplementedError

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        raise NotImplementedError


class SQLiteSession(SessionABC):
    """SQLite-based implementation of session storage.

    This implementation stores conversation history in a SQLite database.
    By default, uses an in-memory database that is lost when the process ends.
    For persistent storage, provide a file path.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str | Path = ":memory:",
        sessions_table: str = "agent_sessions",
        messages_table: str = "agent_messages",
    ):
        """Initialize the SQLite session.

        Args:
            session_id: Unique identifier for the conversation session
            db_path: Path to the SQLite database file. Defaults to ':memory:'
                (in-memory database)
            sessions_table: Name of the table to store session metadata.
                Defaults to 'agent_sessions'
            messages_table: Name of the table to store message data.
                Defaults to 'agent_messages'
        """
        self.session_id = session_id
        self.db_path = str(db_path)
        self.sessions_table = sessions_table
        self.messages_table = messages_table
        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Ensure database connection is established and initialized."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row

        if not self._initialized:
            await self._initialize_tables()
            self._initialized = True

        return self._connection

    async def _initialize_tables(self):
        """Create tables if they don't exist."""
        conn = await self._ensure_connection()

        # Create sessions table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.sessions_table} (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Create messages table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES {self.sessions_table}(session_id)
            )
        """)

        # Create index on session_id for faster queries
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.messages_table}_session_id
            ON {self.messages_table}(session_id)
        """)

        await conn.commit()

    async def get_items(
        self,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history
        """
        conn = await self._ensure_connection()

        if limit is not None:
            # Get latest N items, ordered by timestamp descending
            query = f"""
                SELECT role, content, timestamp, metadata
                FROM {self.messages_table}
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            cursor = await conn.execute(query, (self.session_id, limit))
        else:
            # Get all items, ordered by timestamp descending
            query = f"""
                SELECT role, content, timestamp, metadata
                FROM {self.messages_table}
                WHERE session_id = ?
                ORDER BY timestamp DESC
            """
            cursor = await conn.execute(query, (self.session_id,))

        rows = await cursor.fetchall()

        # Return in chronological order (reverse the DESC order)
        items = []
        for row in reversed(rows):
            item = {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            if row["metadata"]:
                try:
                    item["metadata"] = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    item["metadata"] = row["metadata"]
            items.append(item)

        return items

    async def add_items(self, items: List[dict]) -> None:
        """Add new items to the conversation history.

        Args:
            items: List of input items to add to the history
        """
        conn = await self._ensure_connection()

        # Ensure session exists
        now = datetime.utcnow().isoformat()
        await conn.execute(
            f"""
            INSERT OR IGNORE INTO {self.sessions_table} (session_id, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (self.session_id, now, now),
        )

        # Add messages
        for item in items:
            role = item.get("role", "user")
            content = item.get("content", "")
            timestamp = item.get("timestamp", now)
            metadata = item.get("metadata")

            if metadata is not None and not isinstance(metadata, str):
                metadata = json.dumps(metadata)

            await conn.execute(
                f"""
                INSERT INTO {self.messages_table} (session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.session_id, role, content, timestamp, metadata),
            )

        # Update session timestamp
        await conn.execute(
            f"""
            UPDATE {self.sessions_table}
            SET updated_at = ?
            WHERE session_id = ?
            """,
            (now, self.session_id),
        )

        await conn.commit()

    async def pop_item(self) -> Optional[dict]:
        """Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty
        """
        conn = await self._ensure_connection()

        # Get the most recent item
        query = f"""
            SELECT id, role, content, timestamp, metadata
            FROM {self.messages_table}
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """
        cursor = await conn.execute(query, (self.session_id,))
        row = await cursor.fetchone()

        if not row:
            return None

        item = {
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
        }
        if row["metadata"]:
            try:
                item["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                item["metadata"] = row["metadata"]

        # Delete the item
        await conn.execute(
            f"DELETE FROM {self.messages_table} WHERE id = ?",
            (row["id"],),
        )
        await conn.commit()

        return item

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        conn = await self._ensure_connection()

        await conn.execute(
            f"DELETE FROM {self.messages_table} WHERE session_id = ?",
            (self.session_id,),
        )
        await conn.execute(
            f"DELETE FROM {self.sessions_table} WHERE session_id = ?",
            (self.session_id,),
        )
        await conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            # Use asyncio to close the connection properly
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the close operation
                    asyncio.create_task(self._connection.close())
                else:
                    # Run the close operation
                    loop.run_until_complete(self._connection.close())
            except RuntimeError:
                # If no event loop, create one
                asyncio.run(self._connection.close())
            finally:
                self._connection = None
                self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False
