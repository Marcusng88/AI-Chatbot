from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def read_users():
    return []


@router.get("/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

