"""
Authentication Models
=====================
Pydantic models for user authentication
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class UserRole(str, Enum):
    """User roles for access control."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    is_active: bool = True


class UserCreate(UserBase):
    """Model for creating a new user."""
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """Model for user login."""
    username: str  # Can be username or email
    password: str


class UserUpdate(BaseModel):
    """Model for updating user info."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserPasswordChange(BaseModel):
    """Model for changing password."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class User(UserBase):
    """User model returned from database."""
    id: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserInDB(User):
    """User model with hashed password (internal use only)."""
    hashed_password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[UserRole] = None
    exp: Optional[datetime] = None


class RefreshToken(BaseModel):
    """Refresh token request."""
    refresh_token: str
