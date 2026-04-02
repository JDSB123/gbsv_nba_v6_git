"""Player-level and quarter scoring feature builders."""

from collections import defaultdict
from typing import Any, cast

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Game, Player, PlayerGameStats
from src.models.features._helpers import NaN, _as_float


async def _player_and_quarter_features(
    db: AsyncSession, home_id: int, away_id: int, game: Any,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        recent_game_ids_result = await db.execute(
            select(Game.id)
            .where(
                Game.status == "FT",
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time < game.commence_time,
            )
            .order_by(Game.commence_time.desc())
            .limit(5)
        )
        recent_game_ids = [r[0] for r in recent_game_ids_result.fetchall()]

        if recent_game_ids:
            pgs_result = await db.execute(
                select(PlayerGameStats)
                .join(Player, PlayerGameStats.player_id == Player.id)
                .where(
                    PlayerGameStats.game_id.in_(recent_game_ids),
                    Player.team_id == team_id,
                )
            )
            pgs_rows = pgs_result.scalars().all()
        else:
            pgs_rows = []

        if pgs_rows:
            all_pts = [_as_float(cast(Any, p.points)) for p in pgs_rows]
            all_ast = [_as_float(cast(Any, p.assists)) for p in pgs_rows]
            all_reb = [_as_float(cast(Any, p.rebounds)) for p in pgs_rows]
            all_fg = [_as_float(cast(Any, p.fg_pct)) for p in pgs_rows if p.fg_pct is not None]
            all_3pt = [
                _as_float(cast(Any, p.three_pct)) for p in pgs_rows if p.three_pct is not None
            ]
            all_min = [_as_float(cast(Any, p.minutes)) for p in pgs_rows]

            by_game: dict[int, list[Any]] = defaultdict(list)
            for p in pgs_rows:
                by_game[int(cast(Any, p.game_id))].append(p)

            starter_pts_list = []
            bench_pts_list = []
            for _gid, players in by_game.items():
                sorted_p = sorted(
                    players, key=lambda x: _as_float(cast(Any, x.minutes)), reverse=True
                )
                starters = sorted_p[:5]
                bench = sorted_p[5:]
                starter_pts_list.append(sum(_as_float(cast(Any, s.points)) for s in starters))
                bench_pts_list.append(sum(_as_float(cast(Any, b.points)) for b in bench))

            features[f"{prefix}_player_pts_avg"] = float(np.mean(all_pts))
            features[f"{prefix}_player_ast_avg"] = float(np.mean(all_ast))
            features[f"{prefix}_player_reb_avg"] = float(np.mean(all_reb))
            features[f"{prefix}_player_fg_pct"] = float(np.mean(all_fg)) if all_fg else NaN
            features[f"{prefix}_player_3pt_pct"] = float(np.mean(all_3pt)) if all_3pt else NaN
            features[f"{prefix}_starter_pts_avg"] = (
                float(np.mean(starter_pts_list)) if starter_pts_list else NaN
            )
            features[f"{prefix}_bench_pts_avg"] = (
                float(np.mean(bench_pts_list)) if bench_pts_list else NaN
            )
            features[f"{prefix}_bench_ratio"] = features[f"{prefix}_bench_pts_avg"] / max(
                features[f"{prefix}_starter_pts_avg"] + features[f"{prefix}_bench_pts_avg"],
                1.0,
            )
            features[f"{prefix}_min_std"] = float(np.std(all_min)) if len(all_min) > 1 else NaN
        else:
            for k in [
                "player_pts_avg", "player_ast_avg", "player_reb_avg",
                "player_fg_pct", "player_3pt_pct", "starter_pts_avg",
                "bench_pts_avg", "bench_ratio", "min_std",
            ]:
                features[f"{prefix}_{k}"] = NaN

    # Quarter scoring tendencies
    for prefix, team_id in [("home", home_id), ("away", away_id)]:
        res_2 = await db.execute(
            select(Game)
            .where(
                Game.status == "FT",
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                Game.commence_time < game.commence_time,
                Game.home_q1.is_not(None),
            )
            .order_by(Game.commence_time.desc())
            .limit(10)
        )
        qtr_games = res_2.scalars().all()
        if qtr_games:
            q1_scored = []
            q3_scored = []
            for g in qtr_games:
                if int(cast(Any, g.home_team_id)) == team_id:
                    q1_scored.append(_as_float(cast(Any, g.home_q1)))
                    q3_scored.append(_as_float(cast(Any, g.home_q3)))
                else:
                    q1_scored.append(_as_float(cast(Any, g.away_q1)))
                    q3_scored.append(_as_float(cast(Any, g.away_q3)))
            features[f"{prefix}_q1_avg"] = float(np.mean(q1_scored))
            features[f"{prefix}_q3_avg"] = float(np.mean(q3_scored))
        else:
            features[f"{prefix}_q1_avg"] = NaN
            features[f"{prefix}_q3_avg"] = NaN
    return features
