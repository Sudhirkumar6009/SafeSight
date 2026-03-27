"""
SafeSight Authentication API Routes
===================================
REST API endpoints for user authentication using MongoDB
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

from app.auth import (
    AuthService, 
    User, UserCreate, UserLogin, UserUpdate, Token, TokenData, UserRole,
    get_current_user, get_current_active_user, require_role
)
from app.auth.service import auth_service

router = APIRouter(prefix="/auth", tags=["authentication"])


# ============== Request/Response Models ==============

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: Optional[str] = Field(None, description="Full name")


class LoginRequest(BaseModel):
    """User login request."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="Refresh token")


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")


class UpdateProfileRequest(BaseModel):
    """Update user profile request."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class UserResponse(BaseModel):
    """User information response."""
    id: str
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str]


# ============== Public Routes (No Auth Required) ==============

@router.post("/register", response_model=dict)
async def register(request: RegisterRequest):
    """
    Register a new user account.
    
    - **email**: Valid email address
    - **username**: Unique username (3-50 characters)
    - **password**: Password (minimum 8 characters)
    - **full_name**: Optional full name
    """
    try:
        user_data = UserCreate(
            email=request.email,
            username=request.username,
            password=request.password,
            full_name=request.full_name,
            role=UserRole.USER,
            is_active=True
        )
        
        user = await auth_service.create_user(user_data)
        
        return {
            "success": True,
            "message": "User registered successfully",
            "data": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=dict)
async def login(request: LoginRequest):
    """
    Login and get access + refresh tokens.
    
    - **username**: Username or email
    - **password**: Password
    
    Returns JWT access token (short-lived) and refresh token (long-lived).
    """
    try:
        login_data = UserLogin(
            username=request.username,
            password=request.password
        )
        
        token = await auth_service.login(login_data)
        
        return {
            "success": True,
            "message": "Login successful",
            "data": {
                "access_token": token.access_token,
                "refresh_token": token.refresh_token,
                "token_type": token.token_type,
                "expires_in": token.expires_in
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/refresh", response_model=dict)
async def refresh_token(request: RefreshRequest):
    """
    Refresh access token using refresh token.
    
    Use this when your access token expires to get a new one
    without requiring the user to login again.
    """
    try:
        token = await auth_service.refresh_access_token(request.refresh_token)
        
        return {
            "success": True,
            "message": "Token refreshed",
            "data": {
                "access_token": token.access_token,
                "refresh_token": token.refresh_token,
                "token_type": token.token_type,
                "expires_in": token.expires_in
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )


# ============== Protected Routes (Auth Required) ==============

@router.get("/me", response_model=dict)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user's information.
    
    Requires Bearer token in Authorization header.
    """
    return {
        "success": True,
        "data": {
            "id": current_user.id,
            "email": current_user.email,
            "username": current_user.username,
            "full_name": current_user.full_name,
            "role": current_user.role.value,
            "is_active": current_user.is_active,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "last_login": current_user.last_login.isoformat() if current_user.last_login else None
        }
    }


@router.patch("/me", response_model=dict)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Update current user's profile.
    
    Only updates fields that are provided.
    """
    try:
        update_data = UserUpdate(
            full_name=request.full_name,
            email=request.email
        )
        
        updated_user = await auth_service.update_user(current_user.id, update_data)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "message": "Profile updated",
            "data": {
                "id": updated_user.id,
                "email": updated_user.email,
                "username": updated_user.username,
                "full_name": updated_user.full_name
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {str(e)}"
        )


@router.post("/change-password", response_model=dict)
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Change current user's password.
    
    Requires current password for verification.
    """
    try:
        success = await auth_service.change_password(
            user_id=current_user.id,
            current_password=request.current_password,
            new_password=request.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password change failed"
            )
        
        return {
            "success": True,
            "message": "Password changed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )


# ============== Admin Routes (Admin Role Required) ==============

@router.get("/users", response_model=dict)
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    List all users (Admin only).
    """
    try:
        users = await auth_service.list_users(skip=skip, limit=limit)
        
        return {
            "success": True,
            "data": [
                {
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "full_name": user.full_name,
                    "role": user.role.value,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login": user.last_login.isoformat() if user.last_login else None
                }
                for user in users
            ],
            "pagination": {
                "skip": skip,
                "limit": limit,
                "count": len(users)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )


@router.get("/users/{user_id}", response_model=dict)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Get user by ID (Admin only).
    """
    try:
        user = await auth_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "data": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role.value,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}"
        )


class AdminUserUpdate(BaseModel):
    """Admin user update request."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


@router.patch("/users/{user_id}", response_model=dict)
async def update_user_admin(
    user_id: str,
    request: AdminUserUpdate,
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Update any user (Admin only).
    
    Can change role and active status.
    """
    try:
        role = UserRole(request.role) if request.role else None
        
        update_data = UserUpdate(
            full_name=request.full_name,
            email=request.email,
            role=role,
            is_active=request.is_active
        )
        
        updated_user = await auth_service.update_user(user_id, update_data)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "message": "User updated",
            "data": {
                "id": updated_user.id,
                "email": updated_user.email,
                "username": updated_user.username,
                "role": updated_user.role.value,
                "is_active": updated_user.is_active
            }
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {str(e)}"
        )


@router.delete("/users/{user_id}", response_model=dict)
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Delete a user (Admin only).
    """
    try:
        # Prevent self-deletion
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        success = await auth_service.delete_user(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "message": "User deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )
