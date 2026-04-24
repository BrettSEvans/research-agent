"""FastAPI dependencies for authentication.

Provides `get_current_user()` which supports multiple auth methods:
1. Old HMAC session cookies (backwards compatibility)
2. API key via X-API-Key header or query param
3. (Future) FastAPI-Users JWT/Session

Also provides `get_current_org()` for org-scoped operations.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from auth.core import get_user_by_api_key
from auth.handlers import load_session
from db import get_db
from models import Organization, User

logger = logging.getLogger(__name__)

SESSION_COOKIE = "vc_session"


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """Unified user dependency supporting multiple auth methods.

    Tries authentication in this order:
    1. Legacy HMAC session cookie (backwards compatibility during migration)
    2. API key from X-API-Key header or api_key query param
    3. (Future) FastAPI-Users JWT/Session

    Raises:
        HTTPException 401: If no valid auth method is found

    Returns:
        Authenticated User object
    """
    # 1. Try legacy HMAC token from cookie (backwards compatibility)
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        session = load_session(token)
        if session:
            user_id = session.get("user_id")
            if user_id:
                user = db.get(User, user_id)
                if user:
                    logger.debug(f"Authenticated via legacy HMAC token: {user.email}")
                    return user

    # 2. Try API key from header or query param
    api_key = (
        request.headers.get("X-API-Key")
        or request.query_params.get("api_key")
    )
    if api_key:
        user = get_user_by_api_key(db, api_key)
        if user:
            logger.debug(f"Authenticated via API key: {user.email}")
            return user

    # 3. (Future) FastAPI-Users JWT/Session would go here

    # No valid auth found
    logger.debug("No valid auth found, returning 401")
    raise HTTPException(status_code=401, detail="Not authenticated")


def get_current_org(user: User = Depends(get_current_user)) -> Organization:
    """Get the organization for the current user.

    Helper dependency that extracts the org from the authenticated user.
    Use in routes that need org-scoped access control.

    Args:
        user: Current authenticated user (from get_current_user)

    Returns:
        The user's organization object

    Raises:
        HTTPException 500: If user's organization doesn't exist (database inconsistency)
    """
    if not user.organization_id:
        logger.error(f"User {user.id} has no organization_id")
        raise HTTPException(status_code=500, detail="User organization not found")
    return user.organization


def get_optional_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Optional user dependency (doesn't raise 401 if not authenticated).

    For routes that work with or without authentication.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        User object if authenticated, None otherwise
    """
    # Try legacy HMAC token
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        session = load_session(token)
        if session:
            user_id = session.get("user_id")
            if user_id:
                return db.get(User, user_id)

    # Try API key
    api_key = (
        request.headers.get("X-API-Key")
        or request.query_params.get("api_key")
    )
    if api_key:
        return get_user_by_api_key(db, api_key)

    return None
