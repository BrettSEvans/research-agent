"""Phase 1 tests: Verify auth package works before Phase 2 refactor.

Tests that:
1. Old HMAC tokens still work (backwards compat)
2. New API key auth works
3. Both auth methods can coexist
"""

import os
import secrets
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from auth import (
    create_api_key,
    find_or_create_user_org,
    get_user_by_api_key,
    hash_password,
    is_email_allowed,
    load_session,
    sign_session,
    verify_password,
)
from models import Base, Organization, User, Whitelist


@pytest.fixture
def db():
    """Create a fresh test database for each test."""
    # Use in-memory SQLite for each test
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()


class TestPasswordHashing:
    """Test PBKDF2 password hashing."""

    def test_hash_and_verify_password(self):
        """Password should hash and verify correctly."""
        password = "test123secure"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed)

    def test_verify_wrong_password(self):
        """Wrong password should not verify."""
        password = "correct"
        hashed = hash_password(password)
        assert not verify_password("wrong", hashed)

    def test_hash_is_deterministic_with_salt(self):
        """Same password + salt should produce same hash."""
        password = "test"
        salt = secrets.token_bytes(16)
        hash1 = hash_password(password, salt)
        hash2 = hash_password(password, salt)
        assert hash1 == hash2


class TestSessionSigning:
    """Test HMAC session token creation and verification."""

    def test_sign_and_load_session(self):
        """Session should sign and load correctly."""
        payload = {"user_id": 1, "email": "user@example.com"}
        token = sign_session(payload)
        loaded = load_session(token)
        assert loaded is not None
        assert loaded["user_id"] == 1
        assert loaded["email"] == "user@example.com"

    def test_load_corrupted_session(self):
        """Corrupted token should not load."""
        token = "invalid.signature"
        assert load_session(token) is None

    def test_load_expired_session(self):
        """Expired session should not load."""
        # Create session with negative expiry
        import time
        import json
        import base64
        import hashlib
        import hmac

        SECRET_KEY = os.environ.get("SECRET_KEY", "dev")
        payload = {"user_id": 1, "exp": int(time.time()) - 3600}  # 1 hour ago
        data = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        sig = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        token = f"{data}.{sig}"

        assert load_session(token) is None

    def test_tampered_signature_rejected(self):
        """Token with tampered signature should not load."""
        payload = {"user_id": 1}
        token = sign_session(payload)
        # Tamper with signature
        data, sig = token.rsplit(".", 1)
        tampered = f"{data}.badbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbad"
        assert load_session(tampered) is None


class TestAPIKey:
    """Test API key creation and lookup."""

    def test_create_api_key(self):
        """API key should be random and URL-safe."""
        key1 = create_api_key()
        key2 = create_api_key()
        assert key1 != key2
        assert len(key1) > 20
        # Should not contain problematic characters
        assert "/" not in key1 or "_" in key1 or "-" in key1

    def test_get_user_by_api_key(self, db: Session):
        """Should look up user by API key."""
        org = Organization(name="test.com")
        db.add(org)
        db.commit()

        user = User(
            email="test@test.com",
            display_name="Test User",
            hashed_password="oauth",
            api_key="test_key_12345",
            organization_id=org.id,
        )
        db.add(user)
        db.commit()

        found = get_user_by_api_key(db, "test_key_12345")
        assert found is not None
        assert found.email == "test@test.com"

    def test_get_user_by_invalid_api_key(self, db: Session):
        """Should return None for invalid API key."""
        assert get_user_by_api_key(db, "invalid_key") is None

    def test_api_key_lookup_strips_whitespace(self, db: Session):
        """API key lookup should strip whitespace."""
        org = Organization(name="test.com")
        db.add(org)
        db.commit()

        user = User(
            email="test@test.com",
            display_name="Test User",
            hashed_password="oauth",
            api_key="test_key_12345",
            organization_id=org.id,
        )
        db.add(user)
        db.commit()

        found = get_user_by_api_key(db, "  test_key_12345  ")
        assert found is not None
        assert found.email == "test@test.com"


class TestWhitelistChecking:
    """Test email whitelist logic."""

    def test_no_restrictions_allows_all(self, db: Session):
        """With no restrictions, all emails should be allowed."""
        assert is_email_allowed("anyone@example.com", db)
        assert is_email_allowed("someone@corp.io", db)

    def test_db_whitelist_allows_email(self, db: Session):
        """Email in DB whitelist should be allowed."""
        db.add(Whitelist(value="user@example.com", type="email"))
        db.commit()

        assert is_email_allowed("user@example.com", db)
        assert not is_email_allowed("other@example.com", db)

    def test_db_whitelist_allows_domain(self, db: Session):
        """Domain in DB whitelist should allow all emails from that domain."""
        db.add(Whitelist(value="example.com", type="domain"))
        db.commit()

        assert is_email_allowed("user1@example.com", db)
        assert is_email_allowed("user2@example.com", db)
        assert not is_email_allowed("user@other.com", db)

    def test_email_normalized_to_lowercase(self, db: Session):
        """Email should be normalized to lowercase when checking."""
        db.add(Whitelist(value="user@example.com", type="email"))
        db.commit()

        assert is_email_allowed("USER@EXAMPLE.COM", db)
        assert is_email_allowed("User@Example.Com", db)


class TestUserOrgCreation:
    """Test user and org creation logic."""

    def test_create_user_and_org_on_first_login(self, db: Session):
        """First login should create user and org."""
        user, org = find_or_create_user_org(
            db,
            email="john@acme.com",
            display_name="John Doe",
        )
        assert user.email == "john@acme.com"
        assert user.display_name == "John Doe"
        assert user.organization_id == org.id
        assert org.name == "acme.com"

    def test_org_derived_from_email_domain(self, db: Session):
        """Organization should be named after email domain."""
        user, org = find_or_create_user_org(
            db,
            email="alice@mycompany.io",
            display_name="Alice",
        )
        assert org.name == "mycompany.io"

    def test_subsequent_login_reuses_user_and_org(self, db: Session):
        """Second login should not create new user/org."""
        user1, org1 = find_or_create_user_org(db, "user@example.com", "User One")
        user2, org2 = find_or_create_user_org(db, "user@example.com", "User Two")

        assert user1.id == user2.id
        assert org1.id == org2.id
        # Display name should be updated
        db.refresh(user2)
        assert user2.display_name == "User Two"

    def test_users_from_same_domain_share_org(self, db: Session):
        """Multiple users from same domain should share org."""
        user1, org1 = find_or_create_user_org(db, "alice@acme.com")
        user2, org2 = find_or_create_user_org(db, "bob@acme.com")

        assert org1.id == org2.id
        assert org1.name == "acme.com"

    def test_users_get_api_keys(self, db: Session):
        """Each user should get a random API key."""
        user1, _ = find_or_create_user_org(db, "user1@example.com")
        user2, _ = find_or_create_user_org(db, "user2@example.com")

        assert user1.api_key is not None
        assert user2.api_key is not None
        assert user1.api_key != user2.api_key


class TestBackwardsCompatibility:
    """Test that old and new auth methods work together."""

    def test_hmac_and_api_key_both_work(self, db: Session):
        """Both HMAC tokens and API keys should authenticate."""
        # Create user
        org = Organization(name="test.com")
        db.add(org)
        db.commit()

        user = User(
            email="test@test.com",
            display_name="Test",
            hashed_password="oauth",
            api_key="secret_key_123",
            organization_id=org.id,
        )
        db.add(user)
        db.commit()

        # Method 1: HMAC token
        token = sign_session({"user_id": user.id, "email": user.email})
        session = load_session(token)
        assert session is not None
        assert session["user_id"] == user.id

        # Method 2: API key
        found = get_user_by_api_key(db, "secret_key_123")
        assert found is not None
        assert found.email == "test@test.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
