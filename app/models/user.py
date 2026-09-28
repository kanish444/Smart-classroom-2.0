from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class UserRole(str, Enum):
    HOD = "hod"
    CLASS_ADVISOR = "class_advisor"


class UserStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"


class UserModel(BaseModel):
    """Internal user model stored in MongoDB/DB repository."""
    user_id: str = Field(..., description="Unique User ID (e.g. HOD001, ADV001)")
    name: str = Field(..., description="Full Name of the user")
    role: UserRole = Field(..., description="Role: hod or class_advisor")
    password_hash: str = Field(..., description="Bcrypt hashed password")
    email: Optional[str] = Field(None, description="Contact email")
    phone: Optional[str] = Field(None, description="Contact phone")
    department: Optional[str] = Field(None, description="Assigned Department (e.g. AI&DS)")
    year: Optional[str] = Field(None, description="Assigned Year (e.g. 3rd Year)")
    section: Optional[str] = Field(None, description="Assigned Section (e.g. B)")
    assigned_classroom: Optional[str] = Field(None, description="Assigned Classroom ID (e.g. AIDS-B)")
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="Account status: active or disabled")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class UserPublicProfile(BaseModel):
    """Sanitized user profile safe for API responses (never exposes password_hash)."""
    user_id: str
    name: str
    role: str
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    assigned_classroom: Optional[str] = None
    status: str
    created_at: Optional[str] = None


class LoginRequest(BaseModel):
    user_id: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublicProfile
    redirect_url: str


class CreateAdvisorRequest(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=50, description="Unique Advisor User ID")
    name: str = Field(..., min_length=2, max_length=100, description="Advisor Full Name")
    password: str = Field(..., min_length=6, description="Initial Password")
    confirm_password: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: str = Field(..., description="Department, e.g. AI&DS")
    year: str = Field(..., description="Class Year, e.g. 3rd Year")
    section: str = Field(..., description="Section, e.g. B")
    assigned_classroom: Optional[str] = Field(None, description="Assigned Classroom, e.g. AIDS-B")
    status: Optional[str] = "active"


class UpdateAdvisorRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    assigned_classroom: Optional[str] = None
    status: Optional[str] = None


class ResetAdvisorPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)
    confirm_new_password: Optional[str] = None
