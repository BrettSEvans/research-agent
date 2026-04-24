"""Pydantic schemas for FastAPI-Users integration."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserRead(BaseModel):
    """User response schema for API endpoints.

    Used when returning user info via API (e.g., /auth/me, /users/{id}).
    Does NOT include sensitive fields like hashed_password.
    """

    id: int = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email address")
    display_name: Optional[str] = Field(None, description="User display name")
    organization_id: int = Field(..., description="Organization ID this user belongs to")
    is_admin: bool = Field(False, description="Whether user has admin privileges")
    api_key: Optional[str] = Field(None, description="API key for programmatic access")
    created_at: datetime = Field(..., description="Account creation timestamp")

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    """User creation schema for programmatic user creation.

    Used by admins to create new users via API.
    Requires email, password, and organization assignment.
    """

    email: EmailStr = Field(..., description="User email address (must be unique)")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    display_name: Optional[str] = Field(None, description="User display name")
    organization_id: int = Field(..., description="Organization ID to assign user to")
    is_admin: bool = Field(False, description="Whether to grant admin privileges")


class UserUpdate(BaseModel):
    """User update schema for profile changes.

    Users can update display_name and password.
    Email and organization cannot be changed after creation.
    """

    display_name: Optional[str] = Field(None, description="New display name")
    password: Optional[str] = Field(None, min_length=8, description="New password (optional)")

    model_config = {"from_attributes": True}
