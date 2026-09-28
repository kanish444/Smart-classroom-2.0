import os
from typing import Optional, List, Dict, Any
from fastapi import Header, Query, Cookie, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.core.security import decode_access_token
from app.models.user import UserModel, UserRole, UserStatus
from app.services.user_service import UserService

# Backward-compatibility operator token support
OPERATOR_TOKEN = os.getenv("OPERATOR_TOKEN", "smartclass_operator_2026")

bearer_scheme = HTTPBearer(auto_error=False)

_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service


class AuthContext(BaseModel):
    """Context object describing authenticated caller."""
    user_id: str
    name: str
    role: str
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    assigned_classroom: Optional[str] = None
    is_hod: bool = False
    is_advisor: bool = False
    is_operator: bool = False


def get_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token"),
    x_operator_token: Optional[str] = Header(None, alias="X-Operator-Token"),
    operator_token: Optional[str] = Query(None, alias="operator_token"),
    user_service: UserService = Depends(get_user_service)
) -> UserModel:
    """
    Extracts and authenticates the user from JWT Bearer Header, Cookie, or Operator Token.
    Validates token expiration, signature, and account active status.
    """
    token = None
    if auth_header and auth_header.credentials:
        token = auth_header.credentials
    elif access_token_cookie:
        token = access_token_cookie

    if token:
        payload = decode_access_token(token)
        if not payload or "sub" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication credentials.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        user_id = payload["sub"]
        user = user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account not found.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        if user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled. Contact system administrator."
            )
        return user

    # Fallback legacy operator token support
    op_tok = x_operator_token or operator_token
    if op_tok and op_tok == OPERATOR_TOKEN:
        # Generate virtual operator HOD user
        return UserModel(
            user_id="OPERATOR",
            name="System Operator",
            role=UserRole.HOD,
            password_hash="",
            status=UserStatus.ACTIVE
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide a valid Bearer token.",
        headers={"WWW-Authenticate": "Bearer"}
    )


def require_hod(user: UserModel = Depends(get_current_user)) -> UserModel:
    """Restricts endpoint access strictly to HOD (Full System Control)."""
    if user.role != UserRole.HOD:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: HOD role required."
        )
    return user


def require_class_advisor(user: UserModel = Depends(get_current_user)) -> UserModel:
    """Restricts endpoint access to Class Advisor."""
    if user.role != UserRole.CLASS_ADVISOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Class Advisor role required."
        )
    return user


def require_any_authenticated_user(user: UserModel = Depends(get_current_user)) -> UserModel:
    """Allows any active authenticated user (HOD or Class Advisor)."""
    return user


def get_current_role(
    x_operator_token: Optional[str] = Header(None, alias="X-Operator-Token"),
    operator_token: Optional[str] = Query(None, alias="operator_token"),
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token"),
    user_service: UserService = Depends(get_user_service)
) -> AuthContext:
    """Legacy compatibility helper."""
    token = None
    if auth_header and auth_header.credentials:
        token = auth_header.credentials
    elif access_token_cookie:
        token = access_token_cookie

    if token:
        payload = decode_access_token(token)
        if payload and "sub" in payload:
            user = user_service.get_user_by_id(payload["sub"])
            if user and user.status == UserStatus.ACTIVE:
                is_hod = (user.role == UserRole.HOD)
                return AuthContext(
                    user_id=user.user_id,
                    name=user.name,
                    role=user.role.value,
                    department=user.department,
                    year=user.year,
                    section=user.section,
                    assigned_classroom=user.assigned_classroom,
                    is_hod=is_hod,
                    is_advisor=(user.role == UserRole.CLASS_ADVISOR),
                    is_operator=is_hod
                )

    op_tok = x_operator_token or operator_token
    if op_tok and op_tok == OPERATOR_TOKEN:
        return AuthContext(user_id="OPERATOR", name="Operator", role="hod", is_hod=True, is_operator=True)

    return AuthContext(user_id="ANONYMOUS", name="Viewer", role="viewer", is_hod=False, is_operator=False)


def require_operator(
    user: UserModel = Depends(get_current_user)
) -> AuthContext:
    """Legacy compatibility wrapper enforcing operator / HOD privileges."""
    if user.role != UserRole.HOD:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator authorization required for this operation."
        )
    return AuthContext(
        user_id=user.user_id,
        name=user.name,
        role=user.role.value,
        is_hod=True,
        is_operator=True
    )
