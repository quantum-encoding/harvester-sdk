"""SQLAlchemy implementation of Session for persistent conversation history."""

from typing import Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import (
    Table, Column, String, Integer, Text, DateTime, MetaData,
    select, delete, func
)
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


class SQLAlchemySession(SessionABC):
    """SQLAlchemy implementation of Session.

    Provides persistent storage for agent conversation history using SQLAlchemy.
    Supports PostgreSQL, MySQL, SQLite, and other SQLAlchemy-compatible databases.
    """

    def __init__(
        self,
        session_id: str,
        *,
        engine: AsyncEngine,
        create_tables: bool = False,
        sessions_table: str = "agent_sessions",
        messages_table: str = "agent_messages",
    ):
        """Initializes a new SQLAlchemySession.

        Args:
            session_id: Unique identifier for the conversation.
            engine: A pre-configured SQLAlchemy async engine. The engine must be created
                with an async driver (e.g., 'postgresql+asyncpg://', 'mysql+aiomysql://',
                or 'sqlite+aiosqlite://').
            create_tables: Whether to automatically create the required tables and indexes.
                Defaults to False for production use. Set to True for development and
                testing when migrations aren't used.
            sessions_table: Override the default table name for sessions if needed.
            messages_table: Override the default table name for messages if needed.
        """
        self.session_id = session_id
        self.engine = engine
        self.sessions_table_name = sessions_table
        self.messages_table_name = messages_table

        # Define metadata and tables
        self.metadata = MetaData()

        self.sessions_table = Table(
            sessions_table,
            self.metadata,
            Column("session_id", String(255), primary_key=True),
            Column("created_at", DateTime, default=datetime.utcnow),
            Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        )

        self.messages_table = Table(
            messages_table,
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(255), nullable=False, index=True),
            Column("role", String(50), nullable=False),
            Column("content", Text, nullable=False),
            Column("timestamp", DateTime, default=datetime.utcnow),
            Column("metadata", Text, nullable=True),
        )

        if create_tables:
            import asyncio
            asyncio.create_task(self._create_tables())

    async def _create_tables(self):
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(self.metadata.create_all)

    @classmethod
    def from_url(
        cls,
        session_id: str,
        *,
        url: str,
        engine_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "SQLAlchemySession":
        """Create a session from a database URL string.

        Args:
            session_id: Conversation ID.
            url: Any SQLAlchemy async URL, e.g. "postgresql+asyncpg://user:pass@host/db".
            engine_kwargs: Additional keyword arguments forwarded to
                sqlalchemy.ext.asyncio.create_async_engine.
            **kwargs: Additional keyword arguments forwarded to the main constructor
                (e.g., create_tables, custom table names, etc.).

        Returns:
            An instance of SQLAlchemySession connected to the specified database.
        """
        engine_kwargs = engine_kwargs or {}
        engine = create_async_engine(url, **engine_kwargs)
        return cls(session_id, engine=engine, **kwargs)

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
        async with self.engine.connect() as conn:
            query = (
                select(self.messages_table)
                .where(self.messages_table.c.session_id == self.session_id)
                .order_by(self.messages_table.c.timestamp.desc())
            )

            if limit is not None:
                query = query.limit(limit)

            result = await conn.execute(query)
            rows = result.fetchall()

            # Return in chronological order (reverse the DESC order)
            items = []
            for row in reversed(rows):
                items.append({
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "metadata": row.metadata,
                })

            return items

    async def add_items(self, items: List[dict]) -> None:
        """Add new items to the conversation history.

        Args:
            items: List of input items to add to the history
        """
        async with self.engine.begin() as conn:
            # Ensure session exists
            session_exists = await conn.execute(
                select(self.sessions_table).where(
                    self.sessions_table.c.session_id == self.session_id
                )
            )

            if not session_exists.fetchone():
                await conn.execute(
                    self.sessions_table.insert().values(
                        session_id=self.session_id,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                )

            # Add messages
            for item in items:
                await conn.execute(
                    self.messages_table.insert().values(
                        session_id=self.session_id,
                        role=item.get("role", "user"),
                        content=item.get("content", ""),
                        timestamp=datetime.utcnow(),
                        metadata=item.get("metadata"),
                    )
                )

            # Update session timestamp
            await conn.execute(
                self.sessions_table.update()
                .where(self.sessions_table.c.session_id == self.session_id)
                .values(updated_at=datetime.utcnow())
            )

    async def pop_item(self) -> Optional[dict]:
        """Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty
        """
        async with self.engine.begin() as conn:
            # Get the most recent item
            result = await conn.execute(
                select(self.messages_table)
                .where(self.messages_table.c.session_id == self.session_id)
                .order_by(self.messages_table.c.timestamp.desc())
                .limit(1)
            )

            row = result.fetchone()
            if not row:
                return None

            item = {
                "role": row.role,
                "content": row.content,
                "timestamp": row.timestamp,
                "metadata": row.metadata,
            }

            # Delete the item
            await conn.execute(
                delete(self.messages_table).where(
                    self.messages_table.c.id == row.id
                )
            )

            return item

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        async with self.engine.begin() as conn:
            await conn.execute(
                delete(self.messages_table).where(
                    self.messages_table.c.session_id == self.session_id
                )
            )

            await conn.execute(
                delete(self.sessions_table).where(
                    self.sessions_table.c.session_id == self.session_id
                )
            )
