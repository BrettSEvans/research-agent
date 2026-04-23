from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Optional

from sqlalchemy.orm import Session

from models import Organization, Project, User

PASSWORD_ITERATIONS = 200_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt_bytes = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PASSWORD_ITERATIONS)
    return salt_bytes.hex() + "$" + derived.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, derived_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    salt_bytes = bytes.fromhex(salt_hex)
    expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PASSWORD_ITERATIONS)
    return hmac.compare_digest(expected.hex(), derived_hex)


def create_api_key() -> str:
    return secrets.token_urlsafe(32)


def get_user_by_api_key(db: Session, api_key: str) -> Optional[User]:
    return db.query(User).filter(User.api_key == api_key.strip()).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.email == email.strip().lower()).first()
    if user is None:
        return None
    if verify_password(password, user.hashed_password):
        return user
    return None


def create_user(
    db: Session,
    email: str,
    password: str,
    organization: Organization,
    display_name: str | None = None,
    is_admin: bool = False,
    project: Project | None = None,
) -> User:
    normalized_email = email.strip().lower()
    existing = db.query(User).filter(User.email == normalized_email).first()
    if existing:
        return existing

    user = User(
        email=normalized_email,
        display_name=display_name or normalized_email.split("@")[0],
        hashed_password=hash_password(password),
        api_key=os.environ.get("DEFAULT_API_KEY") or create_api_key(),
        organization_id=organization.id,
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    if project and user.id and project.owner_id is None:
        project.owner_id = user.id
        db.add(project)
        db.commit()
        db.refresh(project)

    return user


def ensure_default_organization_and_user(db: Session) -> tuple[Organization, User, Project]:
    default_org_name = os.environ.get("DEFAULT_ORG_NAME", "brettevanssf")
    default_user_email = os.environ.get("DEFAULT_ADMIN_EMAIL", "brettevanssf@gmail.com")
    default_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "change-me")
    default_project_name = os.environ.get("DEFAULT_PROJECT_NAME", "default")

    organization = db.query(Organization).filter_by(name=default_org_name).first()
    if not organization:
        organization = Organization(name=default_org_name)
        db.add(organization)
        db.commit()
        db.refresh(organization)

    user = db.query(User).filter_by(email=default_user_email).first()
    if not user:
        user = create_user(
            db,
            email=default_user_email,
            password=default_password,
            organization=organization,
            display_name="Brett Evans",
            is_admin=True,
            project=None,  # Create project after user exists
        )
    elif os.environ.get("DEFAULT_API_KEY") and user.api_key != os.environ.get("DEFAULT_API_KEY"):
        user.api_key = os.environ["DEFAULT_API_KEY"]
        db.add(user)
        db.commit()
        db.refresh(user)

    # Create default project after user exists (so owner_id is valid)
    project = db.query(Project).filter_by(name=default_project_name, organization_id=organization.id).first()
    if not project:
        project = Project(
            name=default_project_name,
            description="Default organization project.",
            organization_id=organization.id,
            owner_id=user.id,
        )
        db.add(project)
        db.commit()
        db.refresh(project)

    if project.owner_id == 0:
        project.owner_id = user.id
        db.add(project)
        db.commit()
        db.refresh(project)

    return organization, user, project
