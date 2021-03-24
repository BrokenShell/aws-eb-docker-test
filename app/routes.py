from fastapi import APIRouter

router = APIRouter()


@router.post('/test')
async def test():
    return "Test"
