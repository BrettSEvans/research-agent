"""
Tests for Epic 0, Story 0.4: Whitelist Check Utility

Tests cover:
- _is_email_allowed() function logic
- Env var whitelisting (email + domain)
- DB whitelisting (email + domain)
- Open access when no restrictions configured
- Case insensitivity
"""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Whitelist
from web import _is_email_allowed


@pytest.fixture
def db_session():
    """In-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestWhitelistByEnvEmail:
    """Tests for email whitelisting via ALLOWED_EMAILS env var."""

    def test_email_in_allowed_emails_env(self, db_session):
        """Email in _ALLOWED_EMAILS env var is allowed."""
        with patch("web._ALLOWED_EMAILS", {"user@example.com"}), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("user@example.com", db_session) is True

    def test_email_in_allowed_emails_case_insensitive(self, db_session):
        """Email whitelist check is case-insensitive."""
        with patch("web._ALLOWED_EMAILS", {"user@example.com"}), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("USER@EXAMPLE.COM", db_session) is True
            assert _is_email_allowed("User@Example.Com", db_session) is True

    def test_email_not_in_allowed_emails_denied(self, db_session):
        """Email not in whitelist is denied when restrictions exist."""
        with patch("web._ALLOWED_EMAILS", {"user@example.com"}), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("other@example.com", db_session) is False


class TestWhitelistByEnvDomain:
    """Tests for domain whitelisting via ALLOWED_EMAIL_DOMAINS env var."""

    def test_domain_in_allowed_domains_env(self, db_session):
        """Email with domain in _ALLOWED_DOMAINS is allowed."""
        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", {"example.com"}):
            assert _is_email_allowed("user@example.com", db_session) is True
            assert _is_email_allowed("admin@example.com", db_session) is True

    def test_domain_not_in_allowed_domains_denied(self, db_session):
        """Email with domain not in whitelist is denied."""
        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", {"example.com"}):
            assert _is_email_allowed("user@other.com", db_session) is False

    def test_domain_case_insensitive(self, db_session):
        """Domain whitelist check is case-insensitive."""
        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", {"example.com"}):
            assert _is_email_allowed("user@EXAMPLE.COM", db_session) is True
            assert _is_email_allowed("user@Example.Com", db_session) is True


class TestWhitelistByDBEmail:
    """Tests for email whitelisting via Whitelist DB table."""

    def test_email_in_db_whitelist_allowed(self, db_session):
        """Email in Whitelist table with type='email' is allowed."""
        whitelist = Whitelist(
            value="user@example.com",
            type="email",
        )
        db_session.add(whitelist)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            # At least one restriction exists (the DB entry), so deny by default
            # But the email is in DB, so it should be allowed
            assert _is_email_allowed("user@example.com", db_session) is True

    def test_email_in_db_case_insensitive(self, db_session):
        """DB email lookup is case-insensitive."""
        whitelist = Whitelist(
            value="user@example.com",
            type="email",
        )
        db_session.add(whitelist)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("USER@EXAMPLE.COM", db_session) is True

    def test_email_not_in_db_denied(self, db_session):
        """Email not in DB whitelist is denied when restrictions exist."""
        whitelist = Whitelist(
            value="user@example.com",
            type="email",
        )
        db_session.add(whitelist)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("other@example.com", db_session) is False


class TestWhitelistByDBDomain:
    """Tests for domain whitelisting via Whitelist DB table."""

    def test_domain_in_db_whitelist_allowed(self, db_session):
        """Domain in Whitelist table with type='domain' is allowed."""
        whitelist = Whitelist(
            value="example.com",
            type="domain",
        )
        db_session.add(whitelist)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("user@example.com", db_session) is True
            assert _is_email_allowed("admin@example.com", db_session) is True

    def test_domain_in_db_case_insensitive(self, db_session):
        """DB domain lookup is case-insensitive."""
        whitelist = Whitelist(
            value="example.com",
            type="domain",
        )
        db_session.add(whitelist)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("user@EXAMPLE.COM", db_session) is True

    def test_domain_not_in_db_denied(self, db_session):
        """Domain not in DB whitelist is denied when restrictions exist."""
        whitelist = Whitelist(
            value="example.com",
            type="domain",
        )
        db_session.add(whitelist)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("user@other.com", db_session) is False


class TestWhitelistOpenAccess:
    """Tests for open access (no restrictions)."""

    def test_open_access_when_no_restrictions(self, db_session):
        """When nothing is configured, everyone is allowed (open access)."""
        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            # No restrictions and DB is empty
            assert _is_email_allowed("anyone@anywhere.com", db_session) is True
            assert _is_email_allowed("user@example.com", db_session) is True

    def test_restrictions_apply_when_any_exist(self, db_session):
        """If any restriction exists (env or DB), deny by default."""
        # Add one entry to DB
        whitelist = Whitelist(
            value="user@example.com",
            type="email",
        )
        db_session.add(whitelist)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            # Now that DB has an entry, restrictions apply
            assert _is_email_allowed("user@example.com", db_session) is True  # whitelisted
            assert _is_email_allowed("other@example.com", db_session) is False  # not whitelisted


class TestWhitelistMixedScenarios:
    """Tests for complex scenarios with multiple restriction types."""

    def test_env_email_overrides_domain_restriction(self, db_session):
        """Email in env whitelist is allowed even if domain is restricted."""
        with patch("web._ALLOWED_EMAILS", {"user@other.com"}), \
             patch("web._ALLOWED_DOMAINS", {"example.com"}):
            # Both env email and env domain are configured
            assert _is_email_allowed("user@other.com", db_session) is True  # in email whitelist
            assert _is_email_allowed("admin@example.com", db_session) is True  # in domain whitelist
            assert _is_email_allowed("user@blocked.com", db_session) is False  # not in either

    def test_env_and_db_restrictions_combined(self, db_session):
        """Email allowed if in env OR in DB whitelist."""
        # Add domain to DB
        whitelist = Whitelist(
            value="db-example.com",
            type="domain",
        )
        db_session.add(whitelist)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", {"env-user@env.com"}), \
             patch("web._ALLOWED_DOMAINS", set()):
            # Allowed by env email
            assert _is_email_allowed("env-user@env.com", db_session) is True
            # Allowed by DB domain
            assert _is_email_allowed("user@db-example.com", db_session) is True
            # Denied (not in any whitelist)
            assert _is_email_allowed("user@blocked.com", db_session) is False

    def test_multiple_db_entries(self, db_session):
        """Multiple DB whitelist entries all work correctly."""
        whitelist_entries = [
            Whitelist(value="user1@example.com", type="email"),
            Whitelist(value="user2@example.com", type="email"),
            Whitelist(value="approved.com", type="domain"),
            Whitelist(value="trusted.org", type="domain"),
        ]
        db_session.add_all(whitelist_entries)
        db_session.commit()

        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            # Both email entries allowed
            assert _is_email_allowed("user1@example.com", db_session) is True
            assert _is_email_allowed("user2@example.com", db_session) is True
            # Both domain entries allowed
            assert _is_email_allowed("anyone@approved.com", db_session) is True
            assert _is_email_allowed("anyone@trusted.org", db_session) is True
            # Other email/domain denied
            assert _is_email_allowed("user3@example.com", db_session) is False
            assert _is_email_allowed("anyone@blocked.com", db_session) is False

    def test_email_extraction_from_various_domains(self, db_session):
        """Domain is correctly extracted from email."""
        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", {"example.com"}):
            # All should extract "example.com" correctly
            assert _is_email_allowed("simple@example.com", db_session) is True
            assert _is_email_allowed("with+tag@example.com", db_session) is True
            assert _is_email_allowed("dots.in.name@example.com", db_session) is True
            assert _is_email_allowed("underscore_name@example.com", db_session) is True


class TestWhitelistEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_email_without_at_sign_handled(self, db_session):
        """Email without @ is handled gracefully (uses last segment as domain)."""
        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", set()):
            # This is an invalid email but should not crash
            # Domain extraction: "invalid".split("@")[-1] = "invalid"
            result = _is_email_allowed("invalid", db_session)
            assert isinstance(result, bool)

    def test_email_with_multiple_at_signs(self, db_session):
        """Email with multiple @ signs uses last segment as domain."""
        with patch("web._ALLOWED_EMAILS", set()), \
             patch("web._ALLOWED_DOMAINS", {"example.com"}):
            # Domain extraction: "user@fake@example.com".split("@")[-1] = "example.com"
            assert _is_email_allowed("user@fake@example.com", db_session) is True

    def test_empty_email_string(self, db_session):
        """Empty email string is handled (no match)."""
        with patch("web._ALLOWED_EMAILS", {"user@example.com"}), \
             patch("web._ALLOWED_DOMAINS", set()):
            assert _is_email_allowed("", db_session) is False

    def test_whitespace_in_email_preserved(self, db_session):
        """Whitespace is lowercased but not stripped."""
        # The function does .lower() but may not strip
        with patch("web._ALLOWED_EMAILS", {"user@example.com"}), \
             patch("web._ALLOWED_DOMAINS", set()):
            # " user@example.com" has leading space, should not match
            assert _is_email_allowed(" user@example.com", db_session) is False
