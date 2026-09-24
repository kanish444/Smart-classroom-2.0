import os
from typing import Optional
from fastapi import Header, Query, HTTPException, status
from pydantic import BaseModel


class UserRole:
    VIEWER = "VIEWER"
    OPERATOR = "OPERATOR"


class AuthContext(BaseModel):
    role: str
    is_operator: bool = False


# Default operator token; can be overridden via OPERATOR_TOKEN environment variable
OPERATOR_TOKEN = os.getenv("OPERATOR_TOKEN", "smartclass_operator_2026")


def get_current_role(
    x_operator_token: Optional[str] = Header(None, alias="X-Operator-Token"),
    operator_token: Optional[str] = Query(None, alias="operator_token")
) -> AuthContext:
    """
    Identifies the client role based on operator token header or query parameter.
    Does not reject viewers; simply assigns UserRole.VIEWER if token is missing or invalid.
    """
    token = x_operator_token or operator_token
    if token and token == OPERATOR_TOKEN:
        return AuthContext(role=UserRole.OPERATOR, is_operator=True)
    return AuthContext(role=UserRole.VIEWER, is_operator=False)


def require_operator(
    x_operator_token: Optional[str] = Header(None, alias="X-Operator-Token"),
    operator_token: Optional[str] = Query(None, alias="operator_token")
) -> AuthContext:
    """
    Enforces that the client possesses valid operator credentials.
    Raises HTTP 403 Forbidden if invalid or missing.
    """
    auth = get_current_role(x_operator_token=x_operator_token, operator_token=operator_token)
    if not auth.is_operator:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator authorization required for this operation. Provide valid 'X-Operator-Token'."
        )
    return auth
