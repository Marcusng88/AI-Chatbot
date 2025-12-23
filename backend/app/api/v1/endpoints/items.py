from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def read_items():
    return []


@router.get("/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

