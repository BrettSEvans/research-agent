"""Custom authentication backends for FastAPI-Users.

Includes API key backend for programmatic access via X-API-Key header.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request
from sqlalchemy.orm import Session

from auth.core import get_user_by_api_key
from db import SessionLocal
from models import User


class APIKeyBackend:
    """Custom backend for API key authentication via X-API-Key header or query param.

    API keys are stored in the User.api_key field and allow programmatic access
    without requiring Google OAuth or session cookies.

    Usage:
        - Header: curl -H "X-API-Key: xxx" http://api/endpoint
        - Query param: curl http://api/endpoint?api_key=xxx
    """

    def __init__(self, name: str = "api-key"):
        """Initialize the API key backend.

        Args:
            name: Backend identifier name (used in FastAPI-Users logging/naming)
        """
        self.name = name

    async def authenticate(self, request: Request) -> Optional[User]:
        """Authenticate request by API key.

        Extracts API key from X-API-Key header or api_key query parameter.
        Looks up user in database and returns User object if found.

        Args:
            request: FastAPI request object

        Returns:
            User object if API key is valid, None if not found or not provided
        """
        # Try header first, then query param
        api_key = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
        )
        if not api_key:
            return None

        # Get a database session to look up the user
        db = SessionLocal()
        try:
            user = get_user_by_api_key(db, api_key)
            if user:
                return user
        finally:
            db.close()

        # API key not found; return None (let FastAPI handle 401)
        return None
