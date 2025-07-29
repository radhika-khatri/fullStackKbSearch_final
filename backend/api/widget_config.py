from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database.database import get_db
from models.db_models import UserWidgetConfig
from models import schemas

router = APIRouter()

@router.get("/widget-config/{user_id}", response_model=schemas.WidgetConfig)
async def get_widget_config(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserWidgetConfig).where(UserWidgetConfig.user_id == user_id))
    config = result.scalars().first()
    if not config:
        raise HTTPException(status_code=404, detail="Widget config not found")
    return config.config

@router.post("/widget-config/{user_id}", response_model=schemas.WidgetConfig)
async def save_widget_config(user_id: int, config: schemas.WidgetConfig, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserWidgetConfig).where(UserWidgetConfig.user_id == user_id))
    user_config = result.scalars().first()
    if user_config:
        user_config.config = config.dict()
    else:
        user_config = UserWidgetConfig(user_id=user_id, config=config.dict())
        db.add(user_config)
    await db.commit()
    return config
