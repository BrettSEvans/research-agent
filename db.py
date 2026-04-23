from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from models import Base, DeckContext, Organization, Project, Report, SavedExtraction, Upload, User, Whitelist, RegulationSource

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
DATABASE_URL = os.environ.get("DATABASE_URL") or f"sqlite:///{BASE / 'compliance_agent.db'}"

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    migrate_schema()


def migrate_schema() -> None:
    """Add new columns to existing tables without dropping data.

    SQLite does not support IF NOT EXISTS on ALTER TABLE, so each ADD COLUMN
    is wrapped in a try/except. Safe to run on every startup.
    """
    with engine.connect() as conn:
        new_columns = [
            ("reports",           "share_token", "TEXT"),
            ("reports",           "is_public",   "INTEGER NOT NULL DEFAULT 0"),
            ("saved_extractions", "share_token", "TEXT"),
            ("saved_extractions", "is_public",   "INTEGER NOT NULL DEFAULT 0"),
        ]
        for table, col, coltype in new_columns:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"
                    )
                )
                conn.commit()
            except Exception:
                pass  # Column already exists — safe to ignore


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


def seed_eu_regulatory_sources(db: Session) -> None:
    """Seed EU regulatory sources on first startup.

    Idempotent: checks if sources exist before creating and ingesting.
    Immediately ingests all 5 sources to populate the knowledge base.

    This function is called during startup (in web.py) before the scheduler begins,
    ensuring the regulatory KB is available for the first user query.
    """
    # Check if EU sources already exist (idempotent)
    existing = db.query(RegulationSource).filter_by(module="eu_sfdr_csrd").first()
    if existing:
        logger.debug("EU regulatory sources already seeded, skipping")
        return

    # Import here to avoid circular imports
    from regulatory_kb import fetch_if_changed, ingest_source
    from models import utc_now

    # Define the 5 EU regulatory sources to seed
    sources_data = [
        {
            "module": "eu_sfdr_csrd",
            "name": "SFDR Level 1 (EU 2019/2088)",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R2088",
            "version_label": "ELI:32019R2088",
        },
        {
            "module": "eu_sfdr_csrd",
            "name": "SFDR RTS Delegated Regulation (EU 2022/1288)",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R1288",
            "version_label": "ELI:32022R1288",
        },
        {
            "module": "eu_sfdr_csrd",
            "name": "CSRD Directive (EU 2022/2464)",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022L2464",
            "version_label": "ELI:32022L2464",
        },
        {
            "module": "eu_sfdr_csrd",
            "name": "EU AI Act High-Risk Annex III (EU 2024/1689)",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
            "version_label": "ELI:32024R1689",
        },
        {
            "module": "eu_sfdr_csrd",
            "name": "ESRS Set 1 — CSRD Technical Standards (EU 2023/2772)",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023R2772",
            "version_label": "ELI:32023R2772",
        },
    ]

    try:
        # Create RegulationSource rows for each EU source
        sources = []
        for data in sources_data:
            source = RegulationSource(**data)
            db.add(source)
            sources.append(source)
        db.commit()
        logger.info(f"Created {len(sources)} EU regulatory sources in DB")

        # Immediately fetch and ingest each source to populate the KB
        for source in sources:
            try:
                # Fetch the source and check if it changed (always True on first run)
                changed, raw_text = fetch_if_changed(source)
                if changed and raw_text:
                    # Ingest: chunk, embed, and write to disk
                    chunk_count = ingest_source(source, raw_text, db)
                    logger.info(
                        f"Ingested {source.name}: {chunk_count} chunks, "
                        f"last_fetched={source.last_fetched}, last_changed={source.last_changed}"
                    )
                else:
                    logger.warning(f"No content received for {source.name}")
            except Exception as e:
                logger.error(f"Failed to ingest {source.name}: {e}")
                # Continue to next source instead of failing the entire seed

        logger.info("EU regulatory source seeding complete")
    except Exception as e:
        logger.error(f"Error seeding EU regulatory sources: {e}")
        db.rollback()
        raise



def seed_ca_regulatory_sources(db: Session) -> None:
    """Seed California regulatory sources on first startup.

    Idempotent: checks if sources exist before creating and ingesting.
    Immediately ingests all CA sources to populate the knowledge base.

    This function is called during startup (in web.py) before the scheduler begins,
    ensuring the CA regulatory KB is available for the first user query.
    """
    # Check if CA sources already exist (idempotent)
    existing = db.query(RegulationSource).filter_by(module="ca_sb54").first()
    if existing:
        logger.debug("CA regulatory sources already seeded, skipping")
        return

    # Import here to avoid circular imports
    from regulatory_kb import fetch_if_changed, ingest_source
    from models import utc_now

    # Define the CA regulatory sources to seed
    sources_data = [
        {
            "module": "ca_sb54",
            "name": "California SB 54 — Nonprofit Integrity Act",
            "url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=201320140SB54",
            "version_label": "CA SB 54 (2013)",
        },
        {
            "module": "ca_sb54",
            "name": "California SB 164 — Board Diversity Requirements",
            "url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=201820190SB164",
            "version_label": "CA SB 164 (2018)",
        },
        {
            "module": "ca_sb54",
            "name": "California Department of Fair Employment and Housing (DFEH) Guidelines",
            "url": "https://dfeh.ca.gov/wp-content/uploads/sites/32/2020/07/DFEH_Investigations_and_Complaints_Process.pdf",
            "version_label": "DFEH Guidelines (2023)",
        },
    ]

    try:
        # Create RegulationSource rows for each CA source
        sources = []
        for data in sources_data:
            source = RegulationSource(**data)
            db.add(source)
            sources.append(source)
        db.commit()
        logger.info(f"Created {len(sources)} CA regulatory sources in DB")

        # Immediately fetch and ingest each source to populate the KB
        for source in sources:
            try:
                # Fetch the source and check if it changed (always True on first run)
                changed, raw_text = fetch_if_changed(source)
                if changed and raw_text:
                    # Ingest: chunk, embed, and write to disk
                    chunk_count = ingest_source(source, raw_text, db)
                    logger.info(
                        f"Ingested {source.name}: {chunk_count} chunks, "
                        f"last_fetched={source.last_fetched}, last_changed={source.last_changed}"
                    )
                else:
                    logger.warning(f"No content received for {source.name}")
            except Exception as e:
                logger.error(f"Failed to ingest {source.name}: {e}")
                # Continue to next source instead of failing the entire seed

        logger.info("CA regulatory source seeding complete")
    except Exception as e:
        logger.error(f"Error seeding CA regulatory sources: {e}")
        db.rollback()
        raise
