import os
from datetime import datetime, timedelta, timezone

try:
    import jwt
except ImportError:
    class _JwtFallback:
        def encode(self, *args, **kwargs):
            raise ImportError("PyJWT is required to create tokens")

        def decode(self, *args, **kwargs):
            raise ImportError("PyJWT is required to decode tokens")

    jwt = _JwtFallback()

SECRET_KEY = os.getenv("SECRET_KEY") or "dev-secret-key"
ALGORITHM = os.getenv("ALGORITHM") or "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None
