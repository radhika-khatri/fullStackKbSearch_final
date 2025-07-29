from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from models import schemas
from utils.auth_utils import SECRET_KEY, ALGORITHM
from typing import List

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Utility to extract scopes from JWT token
def get_current_user_scopes(token: str = Depends(oauth2_scheme)) -> List[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        scopes = payload.get("scopes", [])
        return scopes
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency to require a scope for a route
def require_scope(required_scope: str):
    def scope_dependency(scopes: List[str] = Depends(get_current_user_scopes)):
        if required_scope not in scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient scope: {required_scope} required",
            )
    return scope_dependency
