from fastapi import APIRouter
from sqlalchemy import select

from src.db.models import Game, OddsSnapshot
from src.db.session import async_session_factory

router = APIRouter(prefix="/odds", tags=["odds"])


@router.get("/latest")
async def latest_odds():
    """Return latest cached odds snapshots for upcoming games."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(OddsSnapshot)
            .join(Game)
            .where(Game.status == "NS")
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(500)
        )
        snapshots = result.scalars().all()
        return {
            "odds": [
                {
                    "game_id": s.game_id,
                    "bookmaker": s.bookmaker,
                    "market": s.market,
                    "outcome": s.outcome_name,
                    "price": s.price,
                    "point": s.point,
                    "captured_at": s.captured_at.isoformat(),
                }
                for s in snapshots
            ],
            "count": len(snapshots),
        }
