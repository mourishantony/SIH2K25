
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
import bcrypt
from bson import ObjectId

from database import get_users_collection

router = APIRouter()
security = HTTPBearer()


JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))



class UserRole(str, Enum):
    ADMIN = "admin"
    EHR_USER = "ehr_user"
    OFFICER = "officer"



ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        "dashboard", "registered_persons", "unknown_persons",
        "alerts", "user_management", "monitoring"
    ],
    UserRole.EHR_USER: [
        "dashboard", "registered_persons", "mdr_management", "alerts"
    ],
    UserRole.OFFICER: [
        "dashboard", "register_person", "registered_persons", "unknown_persons"
    ],
}


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: UserRole


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: dict


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    permissions: List[str]
    created_at: datetime


class UserUpdate(BaseModel):
    email: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None



def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def get_user_permissions(role: str) -> List[str]:
    """Get permissions for a user role."""
    try:
        user_role = UserRole(role)
        return ROLE_PERMISSIONS.get(user_role, [])
    except ValueError:
        return []


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency to get current authenticated user."""
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    # Get user from database
    users = get_users_collection()
    user = users.find_one({"username": username})
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
        )
    
    role = user.get("role", UserRole.OFFICER.value)
    permissions = get_user_permissions(role)
    
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "role": role,
        "permissions": permissions,
        "is_admin": role == UserRole.ADMIN.value
    }


def require_permission(permission: str):
    """Dependency factory to check if user has a specific permission."""
    async def check_permission(current_user: dict = Depends(get_current_user)) -> dict:
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission} access required"
            )
        return current_user
    return check_permission


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency to require admin role."""
    if current_user.get("role") != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# Routes
@router.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    """Login and get access token."""
    users = get_users_collection()
    
    # Find user
    user = users.find_one({"username": user_data.username})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )
    
    # Verify password
    if not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Update last login
    users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    # Get role and permissions
    role = user.get("role", UserRole.OFFICER.value)
    permissions = get_user_permissions(role)
    
    # Create token
    access_token = create_access_token(data={"sub": user["username"]})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"],
            "role": role,
            "permissions": permissions,
            "is_admin": role == UserRole.ADMIN.value
        }
    )


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    return current_user


@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: dict = Depends(get_current_user)
):
    """Change user password."""
    users = get_users_collection()
    
    # Get user with password hash
    user = users.find_one({"_id": ObjectId(current_user["id"])})
    
    # Verify old password
    if not verify_password(old_password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    users.update_one(
        {"_id": ObjectId(current_user["id"])},
        {
            "$set": {
                "password_hash": hash_password(new_password),
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    return {"message": "Password changed successfully"}


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout (client should discard the token)."""
    return {"message": "Logged out successfully"}


# ============================================
# Admin-only user management routes
# ============================================

@router.post("/users", response_model=dict)
async def create_user(user_data: UserCreate, admin: dict = Depends(require_admin)):
    """Create a new user (Admin only)."""
    users = get_users_collection()
    
    # Check if username exists
    if users.find_one({"username": user_data.username}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    if users.find_one({"email": user_data.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    new_user = {
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": hash_password(user_data.password),
        "role": user_data.role.value,
        "is_active": True,
        "created_by": admin["username"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = users.insert_one(new_user)
    
    return {
        "message": f"User '{user_data.username}' created successfully",
        "user": {
            "id": str(result.inserted_id),
            "username": user_data.username,
            "email": user_data.email,
            "role": user_data.role.value,
            "permissions": get_user_permissions(user_data.role.value)
        }
    }


@router.get("/users", response_model=List[dict])
async def list_users(admin: dict = Depends(require_admin)):
    """List all users (Admin only)."""
    users = get_users_collection()
    
    user_list = []
    for user in users.find().sort("created_at", -1):
        role = user.get("role", UserRole.OFFICER.value)
        user_list.append({
            "id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"],
            "role": role,
            "permissions": get_user_permissions(role),
            "is_active": user.get("is_active", True),
            "created_at": user.get("created_at"),
            "last_login": user.get("last_login"),
            "created_by": user.get("created_by")
        })
    
    return user_list


@router.get("/users/{user_id}", response_model=dict)
async def get_user(user_id: str, admin: dict = Depends(require_admin)):
    """Get a specific user (Admin only)."""
    users = get_users_collection()
    
    try:
        user = users.find_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID"
        )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    role = user.get("role", UserRole.OFFICER.value)
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "role": role,
        "permissions": get_user_permissions(role),
        "is_active": user.get("is_active", True),
        "created_at": user.get("created_at"),
        "last_login": user.get("last_login"),
        "created_by": user.get("created_by")
    }


@router.put("/users/{user_id}", response_model=dict)
async def update_user(user_id: str, user_data: UserUpdate, admin: dict = Depends(require_admin)):
    """Update a user (Admin only)."""
    users = get_users_collection()
    
    try:
        user = users.find_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID"
        )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent admin from disabling themselves
    if user_id == admin["id"] and user_data.is_active == False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disable your own account"
        )
    
    # Build update
    update_fields = {"updated_at": datetime.utcnow()}
    
    if user_data.email is not None:
        # Check if email is already used by another user
        existing = users.find_one({"email": user_data.email, "_id": {"$ne": ObjectId(user_id)}})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        update_fields["email"] = user_data.email
    
    if user_data.role is not None:
        update_fields["role"] = user_data.role.value
    
    if user_data.is_active is not None:
        update_fields["is_active"] = user_data.is_active
    
    users.update_one({"_id": ObjectId(user_id)}, {"$set": update_fields})
    
    # Fetch updated user
    updated_user = users.find_one({"_id": ObjectId(user_id)})
    role = updated_user.get("role", UserRole.OFFICER.value)
    
    return {
        "message": "User updated successfully",
        "user": {
            "id": str(updated_user["_id"]),
            "username": updated_user["username"],
            "email": updated_user["email"],
            "role": role,
            "permissions": get_user_permissions(role),
            "is_active": updated_user.get("is_active", True)
        }
    }


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(require_admin)):
    """Delete a user (Admin only)."""
    users = get_users_collection()
    
    # Prevent admin from deleting themselves
    if user_id == admin["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    try:
        result = users.delete_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID"
        )
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User deleted successfully"}


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(user_id: str, new_password: str, admin: dict = Depends(require_admin)):
    """Reset a user's password (Admin only)."""
    users = get_users_collection()
    
    try:
        user = users.find_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID"
        )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    users.update_one(
        {"_id": ObjectId(user_id)},
        {
            "$set": {
                "password_hash": hash_password(new_password),
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    return {"message": f"Password reset successfully for user '{user['username']}'"}


@router.get("/roles")
async def get_roles(current_user: dict = Depends(get_current_user)):
    """Get available roles and their permissions."""
    return {
        "roles": [
            {
                "value": UserRole.ADMIN.value,
                "label": "Administrator",
                "description": "Dashboard, Registered Persons, Unknown Persons, Alerts, User Management, AI Monitoring",
                "permissions": ROLE_PERMISSIONS[UserRole.ADMIN]
            },
            {
                "value": UserRole.EHR_USER.value,
                "label": "EHR System User",
                "description": "Dashboard, Registered Persons, MDR Management, Alerts",
                "permissions": ROLE_PERMISSIONS[UserRole.EHR_USER]
            },
            {
                "value": UserRole.OFFICER.value,
                "label": "Officer",
                "description": "Dashboard, Register Person, Registered Persons, Unknown Persons",
                "permissions": ROLE_PERMISSIONS[UserRole.OFFICER]
            }
        ]
    }
