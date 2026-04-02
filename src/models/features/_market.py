"""Market odds, prop consensus, and sharp/square feature builders."""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import OddsSnapshot
from src.models.features._helpers import (
    SHARP_BOOKS,
    SQUARE_BOOKS,
    NaN,
    _as_float,
    _as_str,
    _home_spreads,
)


async def _prop_consensus_features(
    db: AsyncSession, game: Any, odds_snapshots: list | None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    prop_markets = [
        "player_points", "player_rebounds", "player_assists",
        "player_threes", "player_blocks", "player_steals",
        "player_turnovers", "player_points_rebounds_assists",
        "player_points_rebounds", "player_points_assists",
        "player_rebounds_assists", "player_double_double",
        "player_triple_double",
    ]
    if odds_snapshots is None:
        ranked_props = (
            select(
                OddsSnapshot.id.label("snapshot_id"),
                func.row_number()
                .over(
                    partition_by=(
                        OddsSnapshot.bookmaker,
                        OddsSnapshot.market,
                        OddsSnapshot.description,
                        OddsSnapshot.outcome_name,
                    ),
                    order_by=OddsSnapshot.captured_at.desc(),
                )
                .label("row_num"),
            )
            .where(
                OddsSnapshot.game_id == game.id,
                OddsSnapshot.market.in_(prop_markets),
            )
            .subquery()
        )
        prop_snap_result = await db.execute(
            select(OddsSnapshot)
            .join(ranked_props, OddsSnapshot.id == ranked_props.c.snapshot_id)
            .where(ranked_props.c.row_num == 1)
        )
        prop_snaps = prop_snap_result.scalars().all()
    else:
        prop_snaps = [s for s in odds_snapshots if _as_str(s.market) in prop_markets]

    if prop_snaps:
        _prop_best: dict[tuple[str, str, str, str], Any] = {}
        for s in prop_snaps:
            prop_key = (
                _as_str(s.bookmaker),
                _as_str(s.market),
                _as_str(getattr(s, "description", "")),
                _as_str(s.outcome_name),
            )
            existing = _prop_best.get(prop_key)
            if existing is None or cast(Any, s.captured_at) > cast(Any, existing.captured_at):
                _prop_best[prop_key] = s
        deduped_props = list(_prop_best.values())

        def _over_lines(market: str) -> list[float]:
            return [
                _as_float(s.point)
                for s in deduped_props
                if _as_str(s.market) == market
                and _as_str(s.outcome_name) == "Over"
                and s.point is not None
            ]

        pts_over_lines = _over_lines("player_points")
        features["prop_pts_lines_count"] = float(len(pts_over_lines))
        features["prop_pts_avg_line"] = float(np.mean(pts_over_lines)) if pts_over_lines else NaN

        ast_over_lines = _over_lines("player_assists")
        features["prop_ast_avg_line"] = float(np.mean(ast_over_lines)) if ast_over_lines else NaN

        reb_over_lines = _over_lines("player_rebounds")
        features["prop_reb_avg_line"] = float(np.mean(reb_over_lines)) if reb_over_lines else NaN

        threes_over = _over_lines("player_threes")
        features["prop_threes_avg_line"] = float(np.mean(threes_over)) if threes_over else NaN

        blk_over = _over_lines("player_blocks")
        features["prop_blk_avg_line"] = float(np.mean(blk_over)) if blk_over else NaN

        stl_over = _over_lines("player_steals")
        features["prop_stl_avg_line"] = float(np.mean(stl_over)) if stl_over else NaN

        tov_over = _over_lines("player_turnovers")
        features["prop_tov_avg_line"] = float(np.mean(tov_over)) if tov_over else NaN

        pra_over = _over_lines("player_points_rebounds_assists")
        features["prop_pra_avg_line"] = float(np.mean(pra_over)) if pra_over else NaN

        dd_yes = [
            s for s in deduped_props
            if _as_str(s.market) == "player_double_double" and _as_str(s.outcome_name) == "Yes"
        ]
        features["prop_dd_count"] = float(len(dd_yes))

        td_yes = [
            s for s in deduped_props
            if _as_str(s.market) == "player_triple_double" and _as_str(s.outcome_name) == "Yes"
        ]
        features["prop_td_count"] = float(len(td_yes))

        sharp_pts = [
            _as_float(s.point)
            for s in deduped_props
            if _as_str(s.market) == "player_points"
            and _as_str(s.outcome_name) == "Over"
            and s.point is not None
            and _as_str(s.bookmaker).lower() in SHARP_BOOKS
        ]
        square_pts = [
            _as_float(s.point)
            for s in deduped_props
            if _as_str(s.market) == "player_points"
            and _as_str(s.outcome_name) == "Over"
            and s.point is not None
            and _as_str(s.bookmaker).lower() in SQUARE_BOOKS
        ]
        sharp_avg = float(np.mean(sharp_pts)) if sharp_pts else features["prop_pts_avg_line"]
        square_avg = float(np.mean(square_pts)) if square_pts else features["prop_pts_avg_line"]
        features["prop_sharp_square_diff"] = sharp_avg - square_avg
    else:
        features["prop_pts_lines_count"] = 0.0
        features["prop_pts_avg_line"] = NaN
        features["prop_ast_avg_line"] = NaN
        features["prop_reb_avg_line"] = NaN
        features["prop_threes_avg_line"] = NaN
        features["prop_blk_avg_line"] = NaN
        features["prop_stl_avg_line"] = NaN
        features["prop_tov_avg_line"] = NaN
        features["prop_pra_avg_line"] = NaN
        features["prop_dd_count"] = 0.0
        features["prop_td_count"] = 0.0
        features["prop_sharp_square_diff"] = NaN
    return features


async def _market_features(
    db: AsyncSession, game: Any, odds_snapshots: list | None, home_name: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    if odds_snapshots is None:
        res_6 = await db.execute(
            select(OddsSnapshot)
            .where(OddsSnapshot.game_id == game.id)
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(500)
        )
        snapshots = res_6.scalars().all()
    else:
        snapshots = odds_snapshots

    if snapshots:
        spreads = _home_spreads(snapshots, home_name)
        totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals" and s.point is not None
        ]
        h1_spreads = _home_spreads(snapshots, home_name, "spreads_h1")
        h1_totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals_h1" and s.point is not None
        ]

        features["mkt_spread_avg"] = float(np.mean(spreads)) if spreads else NaN
        features["mkt_spread_std"] = float(np.std(spreads)) if len(spreads) > 1 else NaN
        features["mkt_total_avg"] = float(np.mean(totals)) if totals else NaN
        features["mkt_total_std"] = float(np.std(totals)) if len(totals) > 1 else NaN
        features["mkt_1h_spread_avg"] = float(np.mean(h1_spreads)) if h1_spreads else NaN
        features["mkt_1h_total_avg"] = float(np.mean(h1_totals)) if h1_totals else NaN

        # 1st-half moneyline implied probability
        h2h_h1 = [
            s
            for s in snapshots
            if _as_str(s.market) == "h2h_h1"
            and ("home" in _as_str(s.outcome_name).lower() or _as_str(s.outcome_name) == home_name)
        ]
        if h2h_h1:
            h1_prices = [_as_float(s.price) for s in h2h_h1]
            avg_h1_price = np.mean(h1_prices)
            if avg_h1_price < 0:
                features["mkt_1h_home_ml_prob"] = float(
                    abs(avg_h1_price) / (abs(avg_h1_price) + 100)
                )
            else:
                features["mkt_1h_home_ml_prob"] = float(100 / (avg_h1_price + 100))
        else:
            features["mkt_1h_home_ml_prob"] = 0.5

        # Moneyline implied probability (overall)
        h2h = [s for s in snapshots if _as_str(s.market) == "h2h"]
        if h2h:
            home_prices = [
                _as_float(s.price)
                for s in h2h
                if "home" in _as_str(s.outcome_name).lower() or _as_str(s.outcome_name) == home_name
            ]
            if home_prices:
                avg_price = np.mean(home_prices)
                if avg_price < 0:
                    features["mkt_home_ml_prob"] = float(abs(avg_price) / (abs(avg_price) + 100))
                else:
                    features["mkt_home_ml_prob"] = float(100 / (avg_price + 100))
            else:
                features["mkt_home_ml_prob"] = 0.5
        else:
            features["mkt_home_ml_prob"] = 0.5

        # Sharp vs. Square book analysis
        def _split_by_book_type(snaps: Sequence[Any], mkt: str, field: str = "point"):
            sharp_vals, square_vals = [], []
            for s in snaps:
                if _as_str(s.market) != mkt:
                    continue
                val = getattr(s, field)
                if val is None:
                    continue
                bk = _as_str(s.bookmaker).lower()
                if bk in SHARP_BOOKS:
                    sharp_vals.append(_as_float(val))
                elif bk in SQUARE_BOOKS:
                    square_vals.append(_as_float(val))
            return sharp_vals, square_vals

        def _price_to_implied(price: Any) -> float:
            price_f = _as_float(price)
            if price_f < 0:
                return abs(price_f) / (abs(price_f) + 100)
            return 100 / (price_f + 100)

        sharp_spr = _home_spreads(snapshots, home_name, books=SHARP_BOOKS)
        square_spr = _home_spreads(snapshots, home_name, books=SQUARE_BOOKS)
        features["sharp_spread"] = (
            float(np.mean(sharp_spr)) if sharp_spr else features["mkt_spread_avg"]
        )
        features["square_spread"] = (
            float(np.mean(square_spr)) if square_spr else features["mkt_spread_avg"]
        )
        features["sharp_square_spread_diff"] = features["sharp_spread"] - features["square_spread"]

        sharp_tot, square_tot = _split_by_book_type(snapshots, "totals")
        features["sharp_total"] = (
            float(np.mean(sharp_tot)) if sharp_tot else features["mkt_total_avg"]
        )
        features["square_total"] = (
            float(np.mean(square_tot)) if square_tot else features["mkt_total_avg"]
        )
        features["sharp_square_total_diff"] = features["sharp_total"] - features["square_total"]

        sharp_h2h = [
            s
            for s in snapshots
            if _as_str(s.market) == "h2h"
            and _as_str(s.bookmaker).lower() in SHARP_BOOKS
            and ("home" in _as_str(s.outcome_name).lower() or _as_str(s.outcome_name) == home_name)
        ]
        square_h2h = [
            s
            for s in snapshots
            if _as_str(s.market) == "h2h"
            and _as_str(s.bookmaker).lower() in SQUARE_BOOKS
            and ("home" in _as_str(s.outcome_name).lower() or _as_str(s.outcome_name) == home_name)
        ]
        if sharp_h2h:
            features["sharp_ml_prob"] = float(
                np.mean([_price_to_implied(s.price) for s in sharp_h2h])
            )
        else:
            features["sharp_ml_prob"] = features["mkt_home_ml_prob"]
        if square_h2h:
            features["square_ml_prob"] = float(
                np.mean([_price_to_implied(s.price) for s in square_h2h])
            )
        else:
            features["square_ml_prob"] = features["mkt_home_ml_prob"]
        features["sharp_square_ml_diff"] = float(
            features["sharp_ml_prob"] - features["square_ml_prob"]
        )

        # Line movement (opening -> current)
        spread_timestamps = [
            cast(Any, s.captured_at)
            for s in snapshots
            if _as_str(s.market) == "spreads"
            and s.point is not None
            and _as_str(s.outcome_name) == home_name
        ]
        total_timestamps = [
            cast(Any, s.captured_at)
            for s in snapshots
            if _as_str(s.market) == "totals" and s.point is not None
        ]
        oldest_spread_ts = min(spread_timestamps) if spread_timestamps else None
        latest_spread_ts = max(spread_timestamps) if spread_timestamps else None
        oldest_total_ts = min(total_timestamps) if total_timestamps else None
        latest_total_ts = max(total_timestamps) if total_timestamps else None
        opening_spreads = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "spreads"
            and cast(Any, s.captured_at) == oldest_spread_ts
            and s.point is not None
            and _as_str(s.outcome_name) == home_name
        ]
        current_spreads = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "spreads"
            and cast(Any, s.captured_at) == latest_spread_ts
            and s.point is not None
            and _as_str(s.outcome_name) == home_name
        ]
        opening_totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals"
            and cast(Any, s.captured_at) == oldest_total_ts
            and s.point is not None
        ]
        current_totals = [
            _as_float(s.point)
            for s in snapshots
            if _as_str(s.market) == "totals"
            and cast(Any, s.captured_at) == latest_total_ts
            and s.point is not None
        ]

        open_spr = float(np.mean(opening_spreads)) if opening_spreads else NaN
        curr_spr = float(np.mean(current_spreads)) if current_spreads else NaN
        open_tot = float(np.mean(opening_totals)) if opening_totals else NaN
        curr_tot = float(np.mean(current_totals)) if current_totals else NaN

        features["spread_move"] = curr_spr - open_spr
        features["total_move"] = curr_tot - open_tot

        # Reverse line movement (RLM) indicator
        spread_moved_toward_sharp = (
            features["sharp_square_spread_diff"] > 0 and features["spread_move"] > 0
        ) or (features["sharp_square_spread_diff"] < 0 and features["spread_move"] < 0)
        features["rlm_flag"] = 1.0 if spread_moved_toward_sharp else 0.0
    else:
        for k in [
            "mkt_spread_avg", "mkt_spread_std", "mkt_total_avg", "mkt_total_std",
            "mkt_1h_spread_avg", "mkt_1h_total_avg", "mkt_1h_home_ml_prob",
            "mkt_home_ml_prob", "sharp_spread", "square_spread",
            "sharp_square_spread_diff", "sharp_total", "square_total",
            "sharp_square_total_diff", "sharp_ml_prob", "square_ml_prob",
            "sharp_square_ml_diff", "spread_move", "total_move", "rlm_flag",
        ]:
            features[k] = NaN
    return features
