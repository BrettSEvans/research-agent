"""Authentication package for compliance-agent.

Provides centralized auth logic:
- Password hashing (PBKDF2-HMAC-SHA256)
- Session management (HMAC-signed tokens)
- API key management
- Whitelist checking
- User/org management
- FastAPI dependencies
"""

from auth.core import (
    PBKDF2PasswordHasher,
    create_api_key,
    get_user_by_api_key,
    hash_password,
    verify_password,
)
from auth.dependencies import get_current_org, get_current_user, get_optional_user
from auth.handlers import find_or_create_user_org, is_email_allowed, load_session, sign_session

# Import legacy functions from the parent module-level auth.py file
# This is a workaround since we have both auth.py module and auth/ package
# We need to load the module directly to avoid circular imports
import sys
from pathlib import Path

_auth_py_path = Path(__file__).parent.parent / "auth.py"
if _auth_py_path.exists():
    # Temporarily remove the package from sys.modules to avoid shadowing
    _this_package = sys.modules.pop("auth", None)
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location("_auth_module", _auth_py_path)
        _auth_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_auth_module)
        ensure_default_organization_and_user = _auth_module.ensure_default_organization_and_user
        authenticate_user = _auth_module.authenticate_user
        create_user = _auth_module.create_user
    finally:
        # Restore the package
        sys.modules["auth"] = _this_package

__all__ = [
    # Core
    "hash_password",
    "verify_password",
    "create_api_key",
    "get_user_by_api_key",
    "PBKDF2PasswordHasher",
    # Handlers
    "sign_session",
    "load_session",
    "is_email_allowed",
    "find_or_create_user_org",
    # Dependencies
    "get_current_user",
    "get_current_org",
    "get_optional_user",
    # Legacy functions
    "ensure_default_organization_and_user",
    "authenticate_user",
    "create_user",
]
