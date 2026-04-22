"""
Tests for notification API endpoints.

Tests cover:
- GET /notifications — list undismissed for current user
- POST /notifications/{id}/dismiss — mark dismissed
- Authentication and authorization
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import get_db
from models import Base, User, Organization, Notification, utc_now
from web import app


@pytest.fixture
def db_session():
    """In-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create organization
    org = Organization(name="Test Org")
    session.add(org)
    session.commit()

    # Create two users
    user1 = User(
        email="user1@example.com",
        display_name="User One",
        hashed_password="fake",
        api_key="key1",
        organization_id=org.id,
    )
    user2 = User(
        email="user2@example.com",
        display_name="User Two",
        hashed_password="fake",
        api_key="key2",
        organization_id=org.id,
    )
    session.add_all([user1, user2])
    session.commit()

    yield session
    session.close()


@pytest.fixture
def client(db_session):
    """FastAPI test client with mocked get_db."""

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def make_session_token(user_id: int, email: str, db_session):
    """Create a valid session token for testing."""
    from web import _sign_session
    payload = {
        "email": email,
        "name": email.split("@")[0],
        "picture": "",
        "user_id": user_id,
        "org_id": 1,  # test org
    }
    return _sign_session(payload)


class TestGetNotifications:
    """Tests for GET /notifications."""

    def test_get_notifications_requires_auth(self, client):
        """GET /notifications without session returns 401."""
        resp = client.get("/notifications")
        assert resp.status_code == 401

    def test_get_notifications_returns_undismissed_only(self, client, db_session):
        """GET /notifications returns only undismissed notifications for user."""
        user = db_session.query(User).filter_by(email="user1@example.com").first()

        # Create two notifications: one undismissed, one dismissed
        notif_active = Notification(
            user_id=user.id,
            module="eu_sfdr_csrd",
            source_name="SFDR Level 1",
            title="Update: SFDR Level 1",
            body="This regulation was updated.",
            dismissed_at=None,
        )
        notif_dismissed = Notification(
            user_id=user.id,
            module="ca_sb54",
            source_name="SB 54",
            title="Update: SB 54",
            body="This regulation was updated.",
            dismissed_at=utc_now(),
        )
        db_session.add_all([notif_active, notif_dismissed])
        db_session.commit()

        token = make_session_token(user.id, user.email, db_session)
        client.cookies["vc_session"] = token

        resp = client.get("/notifications")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["source_name"] == "SFDR Level 1"

    def test_get_notifications_returns_only_user_notifications(self, client, db_session):
        """User can only see their own notifications."""
        user1 = db_session.query(User).filter_by(email="user1@example.com").first()
        user2 = db_session.query(User).filter_by(email="user2@example.com").first()

        # User1 notification
        notif1 = Notification(
            user_id=user1.id,
            module="eu_sfdr_csrd",
            source_name="SFDR Level 1",
            title="Update 1",
            body="...",
        )
        # User2 notification
        notif2 = Notification(
            user_id=user2.id,
            module="ca_sb54",
            source_name="SB 54",
            title="Update 2",
            body="...",
        )
        db_session.add_all([notif1, notif2])
        db_session.commit()

        token = make_session_token(user1.id, user1.email, db_session)
        client.cookies["vc_session"] = token

        resp = client.get("/notifications")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["source_name"] == "SFDR Level 1"

    def test_get_notifications_response_shape(self, client, db_session):
        """GET /notifications returns correct JSON shape."""
        user = db_session.query(User).filter_by(email="user1@example.com").first()

        notif = Notification(
            user_id=user.id,
            module="eu_sfdr_csrd",
            source_name="SFDR Level 1 (EU 2019/2088)",
            title="Regulatory update: SFDR Level 1",
            body="This was updated on 2026-04-22...",
        )
        db_session.add(notif)
        db_session.commit()

        token = make_session_token(user.id, user.email, db_session)
        client.cookies["vc_session"] = token

        resp = client.get("/notifications")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert "id" in data[0]
        assert "module" in data[0]
        assert "source_name" in data[0]
        assert "title" in data[0]
        assert "body" in data[0]
        assert "created_at" in data[0]

    def test_get_notifications_empty_list_when_none(self, client, db_session):
        """GET /notifications returns empty list if user has no undismissed notifications."""
        user = db_session.query(User).filter_by(email="user1@example.com").first()

        token = make_session_token(user.id, user.email, db_session)
        client.cookies["vc_session"] = token

        resp = client.get("/notifications")

        assert resp.status_code == 200
        data = resp.json()
        assert data == []


class TestDismissNotification:
    """Tests for POST /notifications/{id}/dismiss."""

    def test_dismiss_notification_requires_auth(self, client):
        """POST /notifications/{id}/dismiss without session returns 401."""
        resp = client.post("/notifications/1/dismiss")
        assert resp.status_code == 401

    def test_dismiss_notification_404_if_not_found(self, client, db_session):
        """Dismiss non-existent notification returns 404."""
        user = db_session.query(User).filter_by(email="user1@example.com").first()

        token = make_session_token(user.id, user.email, db_session)
        client.cookies["vc_session"] = token

        resp = client.post("/notifications/9999/dismiss")

        assert resp.status_code == 404

    def test_dismiss_notification_403_if_not_owner(self, client, db_session):
        """User cannot dismiss another user's notification."""
        user1 = db_session.query(User).filter_by(email="user1@example.com").first()
        user2 = db_session.query(User).filter_by(email="user2@example.com").first()

        notif = Notification(
            user_id=user2.id,
            module="eu_sfdr_csrd",
            source_name="SFDR Level 1",
            title="Update",
            body="...",
        )
        db_session.add(notif)
        db_session.commit()

        token = make_session_token(user1.id, user1.email, db_session)
        client.cookies["vc_session"] = token

        resp = client.post(f"/notifications/{notif.id}/dismiss")

        assert resp.status_code == 403

    def test_dismiss_notification_sets_dismissed_at(self, client, db_session):
        """POST /notifications/{id}/dismiss sets dismissed_at timestamp."""
        user = db_session.query(User).filter_by(email="user1@example.com").first()

        notif = Notification(
            user_id=user.id,
            module="eu_sfdr_csrd",
            source_name="SFDR Level 1",
            title="Update",
            body="...",
            dismissed_at=None,
        )
        db_session.add(notif)
        db_session.commit()

        token = make_session_token(user.id, user.email, db_session)
        client.cookies["vc_session"] = token

        resp = client.post(f"/notifications/{notif.id}/dismiss")

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # Verify dismissed_at is set
        updated = db_session.query(Notification).filter_by(id=notif.id).first()
        assert updated.dismissed_at is not None

    def test_dismissed_notification_excluded_from_get(self, client, db_session):
        """After dismissal, GET /notifications no longer returns it."""
        user = db_session.query(User).filter_by(email="user1@example.com").first()

        notif = Notification(
            user_id=user.id,
            module="eu_sfdr_csrd",
            source_name="SFDR Level 1",
            title="Update",
            body="...",
        )
        db_session.add(notif)
        db_session.commit()
        notif_id = notif.id

        token = make_session_token(user.id, user.email, db_session)
        client.cookies["vc_session"] = token

        # Before dismiss: notification appears in GET
        resp = client.get("/notifications")
        assert len(resp.json()) == 1

        # Dismiss it
        resp = client.post(f"/notifications/{notif_id}/dismiss")
        assert resp.status_code == 200

        # After dismiss: notification does not appear in GET
        resp = client.get("/notifications")
        assert len(resp.json()) == 0
