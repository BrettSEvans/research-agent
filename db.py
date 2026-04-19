from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, DeckContext, Organization, Project, Report, SavedExtraction, Upload, User

BASE = Path(__file__).parent
DATABASE_URL = os.environ.get("DATABASE_URL") or f"sqlite:///{BASE / 'compliance_agent.db'}"

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[sessionmaker, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_existing_data(default_user_email: str, default_org_name: str, default_project_name: str) -> None:
    db = SessionLocal()
    try:
        organization = db.query(Organization).filter_by(name=default_org_name).first()
        if not organization:
            organization = Organization(name=default_org_name)
            db.add(organization)
            db.commit()
            db.refresh(organization)

        user = db.query(User).filter_by(email=default_user_email).first()
        if not user:
            return

        project = db.query(Project).filter_by(name=default_project_name, organization_id=organization.id).first()
        if not project:
            project = Project(name=default_project_name, description="Default project for legacy data.", organization_id=organization.id, owner_id=user.id)
            db.add(project)
            db.commit()
            db.refresh(project)

        # Migrate saved extractions to the database, preserving any existing metadata.
        saved_dir = BASE / "saved_extractions"
        if saved_dir.exists():
            for path in saved_dir.glob("*.json"):
                save_id = path.stem
                if db.query(SavedExtraction).filter_by(save_id=save_id).first():
                    continue
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    continue
                extraction_data = data.get("extraction") or {}
                meta_data = data.get("meta") or {}
                saved = SavedExtraction(
                    save_id=save_id,
                    owner_id=user.id,
                    organization_id=organization.id,
                    project_id=project.id,
                    company_name=meta_data.get("company_name"),
                    original_filename=meta_data.get("original_filename"),
                    extractor_model=meta_data.get("extractor_model"),
                    extractor_version=meta_data.get("extractor_version"),
                    meta_json=json.dumps(meta_data),
                    extraction_json=json.dumps(extraction_data),
                )
                db.add(saved)

        # Migrate saved reports to the database.
        reports_dir = BASE / "saved_reports"
        if reports_dir.exists():
            for path in reports_dir.glob("report_*.json"):
                report_id = path.stem.replace("report_", "")
                if db.query(Report).filter_by(report_id=report_id).first():
                    continue
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    continue
                report = Report(
                    report_id=report_id,
                    owner_id=user.id,
                    organization_id=organization.id,
                    project_id=project.id,
                    company_name=data.get("company_name"),
                    cik=data.get("cik"),
                    extractor_model=data.get("extractor_model"),
                    analyzer_model=data.get("analyzer_model"),
                    report_json=json.dumps(data),
                    log_path=str(path),
                )
                db.add(report)

        # Migrate persisted deck contexts.
        contexts_dir = BASE / "deck_contexts"
        if contexts_dir.exists():
            for path in contexts_dir.glob("deck_*.json"):
                context_id = path.stem.replace("deck_", "")
                if db.query(DeckContext).filter_by(context_id=context_id).first():
                    continue
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    continue
                context = DeckContext(
                    context_id=context_id,
                    owner_id=user.id,
                    organization_id=organization.id,
                    project_id=project.id,
                    context_path=str(path),
                    extraction_json=json.dumps(data),
                    status="saved",
                )
                db.add(context)

        # Migrate orphan uploads for legacy files.
        uploads_dir = BASE / "uploads"
        if uploads_dir.exists():
            for path in uploads_dir.glob("*.pdf"):
                token = path.name.split("_")[0]
                if db.query(Upload).filter_by(upload_token=token).first():
                    continue
                upload = Upload(
                    upload_token=token,
                    owner_id=user.id,
                    organization_id=organization.id,
                    project_id=project.id,
                    original_filename=path.name,
                    stored_path=str(path),
                    status="migrated",
                )
                db.add(upload)

        db.commit()
    finally:
        db.close()
