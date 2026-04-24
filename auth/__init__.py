"""Authentication package for compliance-agent.

Provides centralized auth logic:
- Password hashing (PBKDF2-HMAC-SHA256)
- Session management (HMAC-signed tokens)
- API key management
- Whitelist checking
- User/org management
- FastAPI dependencies
"""

from auth.core import (
    PBKDF2PasswordHasher,
    create_api_key,
    get_user_by_api_key,
    hash_password,
    verify_password,
)
from auth.dependencies import get_current_org, get_current_user, get_optional_user
from auth.handlers import find_or_create_user_org, is_email_allowed, load_session, sign_session

__all__ = [
    # Core
    "hash_password",
    "verify_password",
    "create_api_key",
    "get_user_by_api_key",
    "PBKDF2PasswordHasher",
    # Handlers
    "sign_session",
    "load_session",
    "is_email_allowed",
    "find_or_create_user_org",
    # Dependencies
    "get_current_user",
    "get_current_org",
    "get_optional_user",
]
