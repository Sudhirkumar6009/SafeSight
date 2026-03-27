"""
SafeSight Authentication Module
===============================
MongoDB-based user authentication with JWT tokens
"""

from .mongodb import mongodb_client, get_mongodb, init_mongodb, close_mongodb
from .models import User, UserCreate, UserLogin, UserUpdate, Token, TokenData, UserRole
from .service import AuthService, get_current_user, get_current_active_user, require_role
from .routes import router as auth_router

__all__ = [
    # MongoDB
    "mongodb_client",
    "get_mongodb", 
    "init_mongodb",
    "close_mongodb",
    # Models
    "User",
    "UserCreate",
    "UserLogin",
    "UserUpdate",
    "Token",
    "TokenData",
    "UserRole",
    # Service
    "AuthService",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    # Router
    "auth_router",
]
