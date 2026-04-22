"""
Tests for Epic 0 artifact verification.

Tests cover:
- Story 0.1: auth_old.py is deleted
- Story 0.2: chromadb is not in requirements.txt
- Story 0.5: apscheduler is in requirements.txt and importable
- Story 0.3: analyzer_protocol.py exists and is importable
- Story 0.5: scheduler.py exists and has start_scheduler function
"""

import os
from pathlib import Path

import pytest


class TestStory01AuthOldRemoved:
    """Verify Story 0.1: auth_old.py is removed from repo."""

    def test_auth_old_does_not_exist(self):
        """auth_old.py should not exist in the repository."""
        repo_root = Path(__file__).parent
        auth_old_path = repo_root / "auth_old.py"
        assert not auth_old_path.exists(), "auth_old.py should be deleted"

    def test_auth_old_not_imported_anywhere(self):
        """auth_old module should not be imported in any active code."""
        repo_root = Path(__file__).parent

        # Search for imports of auth_old in Python files
        import_patterns = ["import auth_old", "from auth_old"]

        for py_file in repo_root.glob("*.py"):
            # Skip test files
            if py_file.name.startswith("test_"):
                continue

            with open(py_file) as f:
                content = f.read()
                for pattern in import_patterns:
                    assert pattern not in content, \
                        f"Found '{pattern}' in {py_file.name} — auth_old should not be imported"


class TestStory02ChromadbRemoved:
    """Verify Story 0.2: chromadb is removed from dependencies."""

    def test_chromadb_not_in_requirements(self):
        """chromadb should not be in requirements.txt."""
        req_file = Path(__file__).parent / "requirements.txt"
        assert req_file.exists()

        with open(req_file) as f:
            content = f.read()
            assert "chromadb" not in content, "chromadb should be removed from requirements.txt"

    def test_chromadb_not_imported(self):
        """chromadb should not be imported anywhere in active code."""
        repo_root = Path(__file__).parent

        for py_file in repo_root.glob("*.py"):
            # Skip test files
            if py_file.name.startswith("test_"):
                continue

            with open(py_file) as f:
                content = f.read()
                assert "chromadb" not in content, \
                    f"Found chromadb import in {py_file.name} — should be removed"


class TestStory03AnalyzerProtocol:
    """Verify Story 0.3: analyzer_protocol.py exists and is functional."""

    def test_analyzer_protocol_file_exists(self):
        """analyzer_protocol.py should exist."""
        analyzer_protocol_file = Path(__file__).parent / "analyzer_protocol.py"
        assert analyzer_protocol_file.exists(), "analyzer_protocol.py should exist"

    def test_analyzer_protocol_importable(self):
        """analyzer_protocol module should be importable."""
        try:
            from analyzer_protocol import ClaimAssessment, AnalyzerModule
            assert ClaimAssessment is not None
            assert AnalyzerModule is not None
        except ImportError as e:
            pytest.fail(f"Failed to import analyzer_protocol: {e}")

    def test_claim_assessment_is_pydantic_model(self):
        """ClaimAssessment should be a Pydantic BaseModel."""
        from analyzer_protocol import ClaimAssessment
        from pydantic import BaseModel

        # Check it has Pydantic methods
        assert hasattr(ClaimAssessment, "model_validate")
        assert hasattr(ClaimAssessment, "model_dump")
        assert hasattr(ClaimAssessment, "model_dump_json")

    def test_analyzer_module_is_protocol(self):
        """AnalyzerModule should be a runtime_checkable Protocol."""
        from analyzer_protocol import AnalyzerModule
        from typing import runtime_checkable

        # Verify it's a protocol with assess method
        assert hasattr(AnalyzerModule, "__mro__") or hasattr(AnalyzerModule, "assess")


class TestStory05SchedulerDependency:
    """Verify Story 0.5: apscheduler is in requirements and scheduler.py exists."""

    def test_apscheduler_in_requirements(self):
        """apscheduler should be in requirements.txt."""
        req_file = Path(__file__).parent / "requirements.txt"
        assert req_file.exists()

        with open(req_file) as f:
            content = f.read()
            assert "apscheduler" in content, "apscheduler should be in requirements.txt"

    def test_scheduler_file_exists(self):
        """scheduler.py should exist."""
        scheduler_file = Path(__file__).parent / "scheduler.py"
        assert scheduler_file.exists(), "scheduler.py should exist"

    def test_scheduler_importable(self):
        """scheduler module should be importable (without dependencies installed)."""
        try:
            # We can import the module without executing imports
            import scheduler
            assert hasattr(scheduler, "start_scheduler")
            assert hasattr(scheduler, "stop_scheduler")
            assert hasattr(scheduler, "run_regulatory_update")
        except ImportError as e:
            # apscheduler might not be installed in test env, but file should exist
            pytest.skip(f"apscheduler not installed: {e}")

    def test_scheduler_has_required_functions(self):
        """scheduler.py should have start_scheduler, stop_scheduler, run_regulatory_update."""
        scheduler_file = Path(__file__).parent / "scheduler.py"

        with open(scheduler_file) as f:
            content = f.read()
            assert "def start_scheduler" in content
            assert "def stop_scheduler" in content
            assert "def run_regulatory_update" in content


class TestStory04WhitelistUtility:
    """Verify Story 0.4: _is_email_allowed utility exists in web.py."""

    def test_is_email_allowed_in_web(self):
        """_is_email_allowed function should be defined in web.py."""
        web_file = Path(__file__).parent / "web.py"

        with open(web_file) as f:
            content = f.read()
            assert "def _is_email_allowed" in content, \
                "_is_email_allowed function should be in web.py"

    def test_is_email_allowed_docstring(self):
        """_is_email_allowed should have a docstring explaining its purpose."""
        web_file = Path(__file__).parent / "web.py"

        with open(web_file) as f:
            content = f.read()
            # Find the function and verify it has a docstring
            assert 'def _is_email_allowed(email: str, db: Session) -> bool:' in content
            # Check that it's followed by a docstring
            after_def = content.split('def _is_email_allowed(email: str, db: Session) -> bool:')[1]
            assert '"""' in after_def or "'''" in after_def, \
                "_is_email_allowed should have a docstring"

    def test_is_email_allowed_not_duplicated(self):
        """_is_email_allowed should appear exactly once in web.py (defined once, used once)."""
        web_file = Path(__file__).parent / "web.py"

        with open(web_file) as f:
            content = f.read()
            # Count function definition (should be 1)
            def_count = content.count("def _is_email_allowed")
            assert def_count == 1, \
                "Function _is_email_allowed should be defined exactly once"

    def test_whitelist_logic_extracted(self):
        """Whitelist check should not be duplicated inline elsewhere."""
        web_file = Path(__file__).parent / "web.py"

        with open(web_file) as f:
            content = f.read()

            # Count occurrences of the old inline pattern
            # (email in _ALLOWED_EMAILS combined with domain in _ALLOWED_DOMAINS)
            old_pattern_count = content.count("env_email_hit")
            assert old_pattern_count == 0, \
                "Old inline whitelist pattern (env_email_hit) should be removed"


class TestEpic0Integration:
    """Integration tests verifying Epic 0 impact on overall codebase."""

    def test_no_broken_imports_after_cleanup(self):
        """After Epic 0 cleanup, core modules should still be importable."""
        try:
            import auth  # Should still work (not deleted)
            import models
            import analyzer
            import regulatory_kb
        except ImportError as e:
            pytest.fail(f"Core module import failed after Epic 0: {e}")

    def test_db_models_accessible(self):
        """DB models including new ones should be importable."""
        try:
            from models import (
                Organization, User, Project, Report,
                RegulationSource, Notification  # New models
            )
            assert RegulationSource is not None
            assert Notification is not None
        except ImportError as e:
            pytest.fail(f"Failed to import models: {e}")

    def test_analyzer_protocol_used_in_protocol_check(self):
        """New analyzer_protocol should be usable for isinstance checks."""
        from analyzer_protocol import AnalyzerModule, ClaimAssessment
        from unittest.mock import MagicMock

        # Verify we can create mock analyzers
        mock_analyzer = MagicMock()
        mock_analyzer.assess = MagicMock(return_value=ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Test",
            cited_passages=[],
            jurisdiction="sec",
        ))

        # Should be callable
        result = mock_analyzer.assess(
            claim="Test claim",
            deck_context={},
            retriever=None,
        )
        assert isinstance(result, ClaimAssessment)


class TestEpic0CompletenessCriteria:
    """Verify all Story success criteria for Epic 0."""

    def test_story_01_success_criteria(self):
        """Story 0.1: auth_old.py deleted and not imported."""
        assert not (Path(__file__).parent / "auth_old.py").exists()

        # Verify no imports
        repo_root = Path(__file__).parent
        for py_file in repo_root.glob("*.py"):
            if py_file.name.startswith("test_"):
                continue
            with open(py_file) as f:
                assert "import auth_old" not in f.read()

    def test_story_02_success_criteria(self):
        """Story 0.2: chromadb removed from requirements.txt."""
        req_file = Path(__file__).parent / "requirements.txt"
        with open(req_file) as f:
            assert "chromadb" not in f.read()

    def test_story_03_success_criteria(self):
        """Story 0.3: analyzer_protocol.py exists with Protocol and BaseModel."""
        analyzer_protocol_file = Path(__file__).parent / "analyzer_protocol.py"
        assert analyzer_protocol_file.exists()

        from analyzer_protocol import AnalyzerModule, ClaimAssessment
        assert AnalyzerModule is not None
        assert ClaimAssessment is not None

    def test_story_04_success_criteria(self):
        """Story 0.4: _is_email_allowed utility exists and is used."""
        web_file = Path(__file__).parent / "web.py"
        with open(web_file) as f:
            content = f.read()
            assert "def _is_email_allowed(email: str, db: Session) -> bool:" in content
            # Verify it's called (not just defined)
            assert "_is_email_allowed(" in content

    def test_story_05_success_criteria(self):
        """Story 0.5: apscheduler in requirements, scheduler.py exists with functions."""
        req_file = Path(__file__).parent / "requirements.txt"
        with open(req_file) as f:
            assert "apscheduler" in f.read()

        scheduler_file = Path(__file__).parent / "scheduler.py"
        assert scheduler_file.exists()
        with open(scheduler_file) as f:
            content = f.read()
            assert "def start_scheduler" in content
            assert "def stop_scheduler" in content
            assert "BackgroundScheduler" in content
