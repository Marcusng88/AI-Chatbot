from pydantic import BaseModel
from typing import Optional


class UserBase(BaseModel):
    email: str
    username: str


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[str] = None
    username: Optional[str] = None


class User(UserBase):
    id: int
    
    class Config:
        from_attributes = True

