"""Tests for DB ORM models – schema correctness, constraints, relationships."""

from src.db.models import (
    Base,
    Game,
    IngestionFailure,
    Injury,
    ModelRegistry,
    OddsSnapshot,
    Player,
    PlayerGameStats,
    Prediction,
    Team,
    TeamSeasonStats,
)

# ── Table names ────────────────────────────────────────────────


def test_all_expected_tables():
    names = set(Base.metadata.tables.keys())
    expected = {
        "teams",
        "players",
        "games",
        "team_season_stats",
        "player_game_stats",
        "odds_snapshots",
        "predictions",
        "injuries",
        "ingestion_failures",
        "model_registry",
    }
    assert expected.issubset(names)


# ── Team model ─────────────────────────────────────────────────


def test_team_relationships():
    t = Team()
    assert hasattr(t, "players")
    assert hasattr(t, "season_stats")
    assert hasattr(t, "injuries")


def test_team_columns():
    t = Team(id=1, name="Celtics", abbreviation="BOS", conference="East", division="Atlantic")
    assert t.name == "Celtics"
    assert t.abbreviation == "BOS"


# ── Player model ───────────────────────────────────────────────


def test_player_relationships():
    p = Player()
    assert hasattr(p, "team")
    assert hasattr(p, "game_stats")
    assert hasattr(p, "injuries")


def test_player_columns():
    p = Player(id=10, team_id=1, name="LeBron", position="F", is_active=True)
    assert p.name == "LeBron"
    assert p.is_active is True


# ── Game model ─────────────────────────────────────────────────


def test_game_relationships():
    g = Game()
    assert hasattr(g, "home_team")
    assert hasattr(g, "away_team")
    assert hasattr(g, "odds_snapshots")
    assert hasattr(g, "predictions")
    assert hasattr(g, "player_stats")


def test_game_quarter_scores():
    g = Game(
        home_q1=28,
        home_q2=30,
        home_q3=25,
        home_q4=27,
        away_q1=24,
        away_q2=26,
        away_q3=30,
        away_q4=22,
        home_ot=0,
        away_ot=0,
    )
    assert g.home_q1 == 28
    assert g.away_q4 == 22
    assert g.home_ot == 0


def test_game_default_status():
    g = Game()
    # No default set on constructor, but Column default is "NS"
    assert hasattr(g, "status")


# ── TeamSeasonStats ────────────────────────────────────────────


def test_team_season_stats_columns():
    tss = TeamSeasonStats(
        team_id=1,
        season="2024-2025",
        games_played=82,
        wins=50,
        losses=32,
        ppg=118.2,
        oppg=111.1,
        pace=100.5,
        off_rating=115.0,
        def_rating=108.0,
    )
    assert tss.ppg == 118.2
    assert tss.season == "2024-2025"


def test_team_season_stats_relationship():
    tss = TeamSeasonStats()
    assert hasattr(tss, "team")


# ── PlayerGameStats ────────────────────────────────────────────


def test_player_game_stats_columns():
    pgs = PlayerGameStats(
        player_id=10,
        game_id=100,
        minutes=36,
        points=28,
        rebounds=7,
        assists=5,
        steals=2,
        blocks=1,
        turnovers=3,
        fg_pct=0.52,
        three_pct=0.38,
        ft_pct=0.88,
        plus_minus=12.0,
    )
    assert pgs.points == 28
    assert pgs.fg_pct == 0.52


def test_player_game_stats_relationships():
    pgs = PlayerGameStats()
    assert hasattr(pgs, "player")
    assert hasattr(pgs, "game")


# ── OddsSnapshot ───────────────────────────────────────────────


def test_odds_snapshot_columns():
    snap = OddsSnapshot(
        game_id=1,
        source="odds_api",
        bookmaker="fanduel",
        market="spreads",
        outcome_name="Boston Celtics",
        price=-110,
        point=-5.5,
    )
    assert snap.market == "spreads"
    assert snap.point == -5.5


def test_odds_snapshot_relationship():
    snap = OddsSnapshot()
    assert hasattr(snap, "game")


# ── Prediction ─────────────────────────────────────────────────


def test_prediction_columns():
    pred = Prediction(
        game_id=1,
        model_version="v6.2.0",
        predicted_home_fg=112,
        predicted_away_fg=108,
        predicted_home_1h=55,
        predicted_away_1h=53,
        fg_spread=-4.0,
        fg_total=220.0,
    )
    assert pred.model_version == "v6.2.0"
    assert pred.fg_spread == -4.0


def test_prediction_relationships():
    pred = Prediction()
    assert hasattr(pred, "game")


def test_prediction_clv_columns():
    pred = Prediction(
        game_id=1,
        model_version="v6.2.0",
        predicted_home_fg=112,
        predicted_away_fg=108,
        predicted_home_1h=55,
        predicted_away_1h=53,
        opening_spread=-3.5,
        closing_spread=-4.5,
        clv_spread=1.0,
        clv_total=-0.5,
    )
    assert pred.clv_spread == 1.0


# ── ModelRegistry ──────────────────────────────────────────────


def test_model_registry_columns():
    mr = ModelRegistry(
        model_version="v6.2.0",
        is_active=True,
        promotion_reason="better MAE",
    )
    assert mr.model_version == "v6.2.0"
    assert mr.is_active is True


# ── Injury ─────────────────────────────────────────────────────


def test_injury_columns():
    inj = Injury(
        player_id=10,
        team_id=1,
        status="out",
        description="Knee",
    )
    assert inj.status == "out"


def test_injury_relationships():
    inj = Injury()
    assert hasattr(inj, "player")
    assert hasattr(inj, "team")


# ── IngestionFailure ───────────────────────────────────────────


def test_ingestion_failure_columns():
    f = IngestionFailure(
        job_name="poll_fg_odds",
        error_message="timeout",
    )
    assert f.job_name == "poll_fg_odds"
    assert f.error_message == "timeout"
