import os
import datetime
from typing import Optional, Dict, Any
import bcrypt
import jwt
from loguru import logger

# Configuration from environment or defaults
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "smartclass-super-secret-key-change-in-production-2026")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))  # 12 hours


def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt with salt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as e:
        logger.warning(f"Password verification error: {e}")
        return False


def create_access_token(
    user_id: str,
    role: str,
    name: str,
    department: Optional[str] = None,
    year: Optional[str] = None,
    section: Optional[str] = None,
    assigned_classroom: Optional[str] = None,
    expires_delta: Optional[datetime.timedelta] = None
) -> str:
    """Generates a signed JWT access token containing identity and role claims."""
    if expires_delta:
        expire = datetime.datetime.now(datetime.timezone.utc) + expires_delta
    else:
        expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode: Dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "name": name,
        "department": department or "",
        "year": year or "",
        "section": section or "",
        "assigned_classroom": assigned_classroom or "",
        "exp": expire,
        "iat": datetime.datetime.now(datetime.timezone.utc)
    }

    token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT token. Returns payload dict or None if invalid/expired."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("Token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token: {e}")
        return None
