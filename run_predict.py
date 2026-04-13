import asyncio, sys, os, logging
sys.path.insert(0, '.')
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

async def run():
    from src.db.session import async_session_factory
    from src.models.predictor import Predictor
    from sqlalchemy.orm import selectinload
    from src.db.models import Game

    predictor = Predictor()
    if not predictor.is_ready:
        print('ERROR: Models not loaded.')
        return

    print('Models loaded. Running predictions...\n')
    async with async_session_factory() as db:
        predictions = await predictor.predict_upcoming(db)
        # Reload with team names
        game_ids = [p.game_id for p in predictions]
        from sqlalchemy import select
        result = await db.execute(
            select(Game)
            .options(selectinload(Game.home_team), selectinload(Game.away_team))
            .where(Game.id.in_(game_ids))
        )
        games_map = {g.id: g for g in result.scalars().all()}

    print(f'=== {len(predictions)} PREDICTIONS FOR {__import__("datetime").date.today()} ===\n')
    print(f'{"Matchup":<45} {"Spread":>7} {"Total":>7} {"ML Prob":>8} {"1H Spr":>7} {"1H Tot":>7}')
    print('-' * 90)
    for p in predictions:
        g = games_map.get(p.game_id)
        if g:
            away = getattr(g.away_team, 'name', '?')
            home = getattr(g.home_team, 'name', '?')
            matchup = f'{away} @ {home}'
        else:
            matchup = f'Game #{p.game_id}'
        spread = f'{p.fg_spread:+.1f}' if p.fg_spread else 'N/A'
        total = f'{p.fg_total:.1f}' if p.fg_total else 'N/A'
        ml = f'{p.fg_home_ml_prob:.1%}' if p.fg_home_ml_prob else 'N/A'
        h1s = f'{p.h1_spread:+.1f}' if p.h1_spread else 'N/A'
        h1t = f'{p.h1_total:.1f}' if p.h1_total else 'N/A'
        print(f'{matchup:<45} {spread:>7} {total:>7} {ml:>8} {h1s:>7} {h1t:>7}')

asyncio.run(run())
