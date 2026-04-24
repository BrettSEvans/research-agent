"""Core authentication utilities: password hashing, API key management."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from models import User

PASSWORD_ITERATIONS = 200_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256.

    Args:
        password: Plain text password
        salt: Optional salt bytes; generates random 16-byte salt if None

    Returns:
        Hashed password in format: {salt_hex}${derived_hex}
    """
    salt_bytes = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PASSWORD_ITERATIONS)
    return salt_bytes.hex() + "$" + derived.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash using constant-time comparison.

    Args:
        password: Plain text password to verify
        stored_hash: Hash in format {salt_hex}${derived_hex}

    Returns:
        True if password matches hash, False otherwise
    """
    try:
        salt_hex, derived_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    salt_bytes = bytes.fromhex(salt_hex)
    expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PASSWORD_ITERATIONS)
    return hmac.compare_digest(expected.hex(), derived_hex)


def create_api_key() -> str:
    """Generate a random URL-safe API key (32 bytes).

    Returns:
        Random API key string
    """
    return secrets.token_urlsafe(32)


def get_user_by_api_key(db: Session, api_key: str) -> Optional[User]:
    """Look up a user by API key.

    Args:
        db: SQLAlchemy session
        api_key: API key to look up (will be stripped of whitespace)

    Returns:
        User object if found, None otherwise
    """
    return db.query(User).filter(User.api_key == api_key.strip()).first()


class PBKDF2PasswordHasher:
    """FastAPI-Users compatible password hasher using PBKDF2-HMAC-SHA256.

    Wraps the existing hash_password() / verify_password() functions
    to provide the interface expected by FastAPI-Users.
    """

    def hash(self, password: str) -> str:
        """Hash a password (synchronous).

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        return hash_password(password)

    def verify(self, password: str, hash: str) -> bool:
        """Verify a password against a hash (synchronous).

        Args:
            password: Plain text password to verify
            hash: Stored password hash

        Returns:
            True if password matches, False otherwise
        """
        return verify_password(password, hash)

    def verify_and_update(self, password: str, hash: str) -> Tuple[bool, str]:
        """Verify password and optionally update hash (FastAPI-Users interface).

        Args:
            password: Plain text password to verify
            hash: Current stored hash

        Returns:
            Tuple of (valid, updated_hash). Hash is not updated for PBKDF2.
        """
        valid = verify_password(password, hash)
        # PBKDF2 doesn't need rehashing with current params; return as-is
        return valid, hash

    async def hash_async(self, password: str) -> str:
        """Async version of hash() for FastAPI-Users.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        return self.hash(password)

    async def verify_and_update_async(self, password: str, hash: str) -> Tuple[bool, str]:
        """Async version of verify_and_update() for FastAPI-Users.

        Args:
            password: Plain text password to verify
            hash: Current stored hash

        Returns:
            Tuple of (valid, updated_hash)
        """
        return self.verify_and_update(password, hash)
