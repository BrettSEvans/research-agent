from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    users = relationship("User", back_populates="organization")
    projects = relationship("Project", back_populates="organization")
    deck_contexts = relationship("DeckContext", back_populates="organization")
    saved_extractions = relationship("SavedExtraction", back_populates="organization")
    reports = relationship("Report", back_populates="organization")
    uploads = relationship("Upload", back_populates="organization")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(256), unique=True, nullable=False, index=True)
    display_name = Column(String(128), nullable=True)
    hashed_password = Column(String(256), nullable=False)
    api_key = Column(String(128), unique=True, nullable=False, index=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    organization = relationship("Organization", back_populates="users")
    projects = relationship("Project", back_populates="owner")
    deck_contexts = relationship("DeckContext", back_populates="owner")
    saved_extractions = relationship("SavedExtraction", back_populates="owner")
    reports = relationship("Report", back_populates="owner")
    uploads = relationship("Upload", back_populates="owner")


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("name", "organization_id", name="uix_project_org"),)

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    organization = relationship("Organization", back_populates="projects")
    owner = relationship("User", back_populates="projects")
    deck_contexts = relationship("DeckContext", back_populates="project")
    saved_extractions = relationship("SavedExtraction", back_populates="project")
    reports = relationship("Report", back_populates="project")
    uploads = relationship("Upload", back_populates="project")


class DeckContext(Base):
    __tablename__ = "deck_contexts"

    id = Column(Integer, primary_key=True, index=True)
    context_id = Column(String(64), unique=True, nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    original_filename = Column(String(256), nullable=True)
    extractor_model = Column(String(128), nullable=True)
    extractor_version = Column(String(64), nullable=True)
    context_path = Column(String(512), nullable=True)
    extraction_json = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="uploaded")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    owner = relationship("User", back_populates="deck_contexts")
    organization = relationship("Organization", back_populates="deck_contexts")
    project = relationship("Project", back_populates="deck_contexts")


class SavedExtraction(Base):
    __tablename__ = "saved_extractions"

    id = Column(Integer, primary_key=True, index=True)
    save_id = Column(String(64), unique=True, nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    company_name = Column(String(256), nullable=True)
    original_filename = Column(String(256), nullable=True)
    extractor_model = Column(String(128), nullable=True)
    extractor_version = Column(String(64), nullable=True)
    meta_json = Column(Text, nullable=True)
    extraction_json = Column(Text, nullable=False)
    saved_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    share_token = Column(String(64), unique=True, nullable=True, index=True)
    is_public = Column(Boolean, default=False, nullable=False)

    owner = relationship("User", back_populates="saved_extractions")
    organization = relationship("Organization", back_populates="saved_extractions")
    project = relationship("Project", back_populates="saved_extractions")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(64), unique=True, nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    company_name = Column(String(256), nullable=True)
    cik = Column(String(32), nullable=True)
    extractor_model = Column(String(128), nullable=True)
    analyzer_model = Column(String(128), nullable=True)
    report_json = Column(Text, nullable=False)
    generated_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    log_path = Column(String(512), nullable=True)
    share_token = Column(String(64), unique=True, nullable=True, index=True)
    is_public = Column(Boolean, default=False, nullable=False)

    owner = relationship("User", back_populates="reports")
    organization = relationship("Organization", back_populates="reports")
    project = relationship("Project", back_populates="reports")


class Whitelist(Base):
    __tablename__ = "whitelist"

    id = Column(Integer, primary_key=True, index=True)
    value = Column(String(256), unique=True, nullable=False, index=True)  # email or domain
    type = Column(String(16), nullable=False)   # "email" or "domain"
    added_by = Column(String(256), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    upload_token = Column(String(64), unique=True, nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    original_filename = Column(String(256), nullable=True)
    stored_path = Column(String(512), nullable=False)
    status = Column(String(32), nullable=False, default="uploaded")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    owner = relationship("User", back_populates="uploads")
    organization = relationship("Organization", back_populates="uploads")
    project = relationship("Project", back_populates="uploads")
