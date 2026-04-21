import pytest
import json
import os
from pathlib import Path
from jinja2 import Template


class TestDataFiles:
    """Test loading and validation of JSON data files."""

    def test_deck_contexts_loading(self):
        """Test that deck context JSON files can be loaded and have valid structure."""
        deck_contexts_dir = Path(__file__).parent.parent / "deck_contexts"

        # Find all .json files (not directories)
        json_files = []
        for item in deck_contexts_dir.iterdir():
            if item.is_file() and item.suffix == '.json':
                json_files.append(item)

        assert len(json_files) > 0, "No deck context JSON files found"

        for json_file in json_files[:3]:  # Test first 3 files to keep test fast
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Validate basic structure
            assert 'company' in data, f"Missing 'company' key in {json_file.name}"
            assert 'claims' in data, f"Missing 'claims' key in {json_file.name}"

            # Validate company structure
            company = data['company']
            assert 'name' in company, f"Missing company name in {json_file.name}"
            assert isinstance(company['name'], str), f"Company name should be string in {json_file.name}"

            # Validate claims structure
            claims = data['claims']
            assert isinstance(claims, list), f"Claims should be list in {json_file.name}"

            for i, claim in enumerate(claims):
                required_keys = ['text', 'verbatim', 'slide', 'category', 'likely_forward_looking']
                for key in required_keys:
                    assert key in claim, f"Missing '{key}' in claim {i} of {json_file.name}"

    def test_sample_claims_loading(self):
        """Test that sample claims JSON can be loaded."""
        sample_file = Path(__file__).parent.parent / "examples" / "sample_claims.json"

        assert sample_file.exists(), "sample_claims.json not found"

        with open(sample_file, 'r') as f:
            data = json.load(f)

        # Basic validation - should be a list or dict
        assert isinstance(data, (list, dict)), "sample_claims.json should contain list or dict"

    def test_saved_extractions_loading(self):
        """Test that saved extraction JSON files can be loaded."""
        saved_dir = Path(__file__).parent.parent / "saved_extractions"

        json_files = list(saved_dir.glob("*.json"))
        if json_files:  # Only test if files exist
            for json_file in json_files[:2]:  # Test first 2 files
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Should be valid JSON (basic test)
                assert isinstance(data, dict), f"Saved extraction should be dict in {json_file.name}"


class TestTemplates:
    """Test HTML template rendering."""

    def test_index_template_syntax(self):
        """Test that index.html template exists and is readable."""
        template_file = Path(__file__).parent.parent / "templates" / "index.html"

        assert template_file.exists(), "index.html template not found"

        with open(template_file, 'r') as f:
            template_content = f.read()

        # This appears to be a static HTML file with JavaScript, not Jinja2
        # Just check that it contains expected HTML structure
        assert '<!DOCTYPE html>' in template_content or '<html' in template_content
        assert '<head>' in template_content
        assert '<body>' in template_content

    def test_login_template_syntax(self):
        """Test that login.html template has valid Jinja2 syntax."""
        template_file = Path(__file__).parent.parent / "templates" / "login.html"

        assert template_file.exists(), "login.html template not found"

        with open(template_file, 'r') as f:
            template_content = f.read()

        # Try to create template - will raise exception if syntax is invalid
        try:
            template = Template(template_content)
        except Exception as e:
            pytest.fail(f"Template syntax error in login.html: {e}")

    def test_index_template_rendering(self):
        """Test that index.html contains expected content."""
        template_file = Path(__file__).parent.parent / "templates" / "index.html"

        with open(template_file, 'r') as f:
            template_content = f.read()

        # Since this is static HTML, just check for key elements
        assert 'VC Compliance Pipeline' in template_content
        assert '<title>' in template_content
        assert 'class="container"' in template_content or 'id="app"' in template_content


class TestConfiguration:
    """Test configuration file loading."""

    def test_pytest_ini_exists(self):
        """Test that pytest.ini configuration file exists."""
        config_file = Path(__file__).parent.parent / "pytest.ini"
        assert config_file.exists(), "pytest.ini not found"

        # Should be readable
        with open(config_file, 'r') as f:
            content = f.read()
            assert len(content) > 0
            assert '[tool:pytest]' in content

    def test_requirements_txt_exists(self):
        """Test that requirements.txt exists and is parseable."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        assert req_file.exists(), "requirements.txt not found"

        with open(req_file, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            assert len(lines) > 0

            # Should contain some common packages
            package_lines = [line for line in lines if not line.startswith('#') and line.strip()]
            assert len(package_lines) > 0, "requirements.txt appears to be empty"


class TestDockerFiles:
    """Test Docker-related files exist and are valid."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"

        with open(dockerfile, 'r') as f:
            content = f.read()
            assert len(content) > 0
            # Basic check for FROM instruction
            assert 'FROM' in content.upper()

    def test_docker_entrypoint_exists(self):
        """Test that docker-entrypoint.sh exists."""
        entrypoint = Path(__file__).parent.parent / "docker-entrypoint.sh"
        assert entrypoint.exists(), "docker-entrypoint.sh not found"

        with open(entrypoint, 'r') as f:
            content = f.read()
            assert len(content) > 0
            # Should be executable (basic check) - accept various shebang formats
            assert '#!/bin/bash' in content or '#!/bin/sh' in content or '#!/usr/bin/env bash' in content

    def test_procfile_exists(self):
        """Test that Procfile exists for deployment."""
        procfile = Path(__file__).parent.parent / "Procfile"
        assert procfile.exists(), "Procfile not found"

        with open(procfile, 'r') as f:
            content = f.read()
            assert len(content) > 0
            # Should contain process definitions
            assert ':' in content  # Basic check for process: command format