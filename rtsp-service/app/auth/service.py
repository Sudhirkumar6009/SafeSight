"""
Authentication Service
======================
JWT-based authentication with MongoDB user storage
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from bson import ObjectId

from app.config import settings
from .mongodb import get_mongodb
from .models import (
    User, UserCreate, UserLogin, UserInDB, UserUpdate,
    Token, TokenData, UserRole
)

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token
security = HTTPBearer()


class AuthService:
    """Authentication service for user management."""
    
    def __init__(self):
        self.db = None
    
    def _get_db(self):
        """Get database connection."""
        if self.db is None:
            self.db = get_mongodb()
        return self.db
    
    @property
    def users_collection(self):
        """Get users collection."""
        return self._get_db().users
    
    # Password utilities
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    # Token utilities
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(
            to_encode,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm
        )
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(
            to_encode,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm
        )
    
    @staticmethod
    def decode_token(token: str) -> Optional[TokenData]:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            
            user_id = payload.get("sub")
            username = payload.get("username")
            role = payload.get("role")
            exp = payload.get("exp")
            
            if user_id is None:
                return None
            
            return TokenData(
                user_id=user_id,
                username=username,
                role=UserRole(role) if role else None,
                exp=datetime.fromtimestamp(exp) if exp else None
            )
        except JWTError as e:
            logger.warning(f"Token decode error: {e}")
            return None
    
    # User CRUD operations
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if email or username already exists
        existing = await self.users_collection.find_one({
            "$or": [
                {"email": user_data.email},
                {"username": user_data.username}
            ]
        })
        
        if existing:
            if existing.get("email") == user_data.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Create user document
        now = datetime.utcnow()
        user_doc = {
            "email": user_data.email,
            "username": user_data.username,
            "full_name": user_data.full_name,
            "role": user_data.role.value,
            "is_active": user_data.is_active,
            "hashed_password": self.hash_password(user_data.password),
            "created_at": now,
            "updated_at": now,
            "last_login": None
        }
        
        result = await self.users_collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        
        logger.info(f"Created user: {user_data.username}")
        
        return self._doc_to_user(user_doc)
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            doc = await self.users_collection.find_one({"_id": ObjectId(user_id)})
            return self._doc_to_user(doc) if doc else None
        except Exception:
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username (includes password hash)."""
        doc = await self.users_collection.find_one({"username": username})
        return self._doc_to_user_in_db(doc) if doc else None
    
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email (includes password hash)."""
        doc = await self.users_collection.find_one({"email": email})
        return self._doc_to_user_in_db(doc) if doc else None
    
    async def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """Authenticate a user by username/email and password."""
        # Try username first, then email
        user = await self.get_user_by_username(login_data.username)
        if not user:
            user = await self.get_user_by_email(login_data.username)
        
        if not user:
            return None
        
        if not self.verify_password(login_data.password, user.hashed_password):
            return None
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        # Update last login
        await self.users_collection.update_one(
            {"_id": ObjectId(user.id)},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        return User(**user.model_dump(exclude={"hashed_password"}))
    
    async def login(self, login_data: UserLogin) -> Token:
        """Login and return tokens."""
        user = await self.authenticate_user(login_data)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create tokens
        token_data = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value
        }
        
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)
        
        logger.info(f"User logged in: {user.username}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_access_token_expire_minutes * 60
        )
    
    async def refresh_access_token(self, refresh_token: str) -> Token:
        """Refresh access token using refresh token."""
        token_data = self.decode_token(refresh_token)
        
        if not token_data or not token_data.user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Verify user still exists and is active
        user = await self.get_user_by_id(token_data.user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        new_token_data = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value
        }
        
        new_access_token = self.create_access_token(new_token_data)
        new_refresh_token = self.create_refresh_token(new_token_data)
        
        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.jwt_access_token_expire_minutes * 60
        )
    
    async def update_user(self, user_id: str, update_data: UserUpdate) -> Optional[User]:
        """Update user information."""
        update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
        
        if not update_dict:
            return await self.get_user_by_id(user_id)
        
        if "role" in update_dict:
            update_dict["role"] = update_dict["role"].value
        
        update_dict["updated_at"] = datetime.utcnow()
        
        result = await self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_dict}
        )
        
        if result.modified_count == 0:
            return None
        
        return await self.get_user_by_id(user_id)
    
    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password."""
        doc = await self.users_collection.find_one({"_id": ObjectId(user_id)})
        if not doc:
            return False
        
        if not self.verify_password(current_password, doc["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        await self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "hashed_password": self.hash_password(new_password),
                "updated_at": datetime.utcnow()
            }}
        )
        
        return True
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        result = await self.users_collection.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0
    
    async def list_users(self, skip: int = 0, limit: int = 100) -> list[User]:
        """List all users."""
        cursor = self.users_collection.find().skip(skip).limit(limit)
        users = []
        async for doc in cursor:
            users.append(self._doc_to_user(doc))
        return users
    
    # Helper methods
    @staticmethod
    def _doc_to_user(doc: dict) -> User:
        """Convert MongoDB document to User model."""
        return User(
            id=str(doc["_id"]),
            email=doc["email"],
            username=doc["username"],
            full_name=doc.get("full_name"),
            role=UserRole(doc["role"]),
            is_active=doc["is_active"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            last_login=doc.get("last_login")
        )
    
    @staticmethod
    def _doc_to_user_in_db(doc: dict) -> UserInDB:
        """Convert MongoDB document to UserInDB model."""
        return UserInDB(
            id=str(doc["_id"]),
            email=doc["email"],
            username=doc["username"],
            full_name=doc.get("full_name"),
            role=UserRole(doc["role"]),
            is_active=doc["is_active"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            last_login=doc.get("last_login"),
            hashed_password=doc["hashed_password"]
        )


# Global auth service instance
auth_service = AuthService()


# Dependency functions for FastAPI
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    token_data = AuthService.decode_token(token)
    
    if not token_data or not token_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user = await auth_service.get_user_by_id(token_data.user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    return current_user


def require_role(required_roles: list[UserRole]):
    """Dependency to require specific user roles."""
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker
