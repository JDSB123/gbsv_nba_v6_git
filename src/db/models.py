from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)  # Basketball API team id
    name = Column(String(120), nullable=False)
    abbreviation = Column(String(10), nullable=False)
    conference = Column(String(20))
    division = Column(String(30))

    players = relationship("Player", back_populates="team")
    season_stats = relationship("TeamSeasonStats", back_populates="team")
    injuries = relationship("Injury", back_populates="team")


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)  # Basketball API player id
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    position = Column(String(20))
    is_active = Column(Boolean, default=True)

    team = relationship("Team", back_populates="players")
    game_stats = relationship("PlayerGameStats", back_populates="player")
    injuries = relationship("Injury", back_populates="player")


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)  # Basketball API game id
    odds_api_id = Column(String(64), unique=True, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    commence_time = Column(DateTime, nullable=False, index=True)
    status = Column(String(10), default="NS")  # NS, Q1-Q4, HT, FT, etc.
    season = Column(String(10))  # e.g. "2024-2025"

    # Quarter scores
    home_q1 = Column(Integer)
    home_q2 = Column(Integer)
    home_q3 = Column(Integer)
    home_q4 = Column(Integer)
    home_ot = Column(Integer, default=0)
    away_q1 = Column(Integer)
    away_q2 = Column(Integer)
    away_q3 = Column(Integer)
    away_q4 = Column(Integer)
    away_ot = Column(Integer, default=0)

    # Derived totals
    home_score_1h = Column(Integer)
    away_score_1h = Column(Integer)
    home_score_fg = Column(Integer)
    away_score_fg = Column(Integer)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    odds_snapshots = relationship("OddsSnapshot", back_populates="game")
    predictions = relationship("Prediction", back_populates="game")
    player_stats = relationship("PlayerGameStats", back_populates="game")

    __table_args__ = (Index("ix_games_status_commence", "status", "commence_time"),)


class TeamSeasonStats(Base):
    __tablename__ = "team_season_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    season = Column(String(10), nullable=False)
    games_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    ppg = Column(Float)
    oppg = Column(Float)
    pace = Column(Float)
    off_rating = Column(Float)
    def_rating = Column(Float)

    team = relationship("Team", back_populates="season_stats")

    __table_args__ = (UniqueConstraint("team_id", "season", name="uq_team_season"),)


class PlayerGameStats(Base):
    __tablename__ = "player_game_stats"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    minutes = Column(Integer)
    points = Column(Integer)
    rebounds = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)
    fg_pct = Column(Float)
    three_pct = Column(Float)
    ft_pct = Column(Float)
    plus_minus = Column(Float)

    player = relationship("Player", back_populates="game_stats")
    game = relationship("Game", back_populates="player_stats")

    __table_args__ = (UniqueConstraint("player_id", "game_id", name="uq_player_game"),)


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    source = Column(String(30), nullable=False)  # "odds_api" or "basketball_api"
    bookmaker = Column(String(60), nullable=False)
    market = Column(String(30), nullable=False)  # h2h, spreads, totals, *_1st_half
    outcome_name = Column(String(60), nullable=False)  # team name or "Over"/"Under"
    price = Column(Float, nullable=False)
    point = Column(Float)  # spread or total line
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    game = relationship("Game", back_populates="odds_snapshots")

    __table_args__ = (
        Index("ix_odds_game_market_captured", "game_id", "market", "captured_at"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    predicted_home_fg = Column(Float, nullable=False)
    predicted_away_fg = Column(Float, nullable=False)
    predicted_home_1h = Column(Float, nullable=False)
    predicted_away_1h = Column(Float, nullable=False)
    # Derived market outputs
    fg_spread = Column(Float)
    fg_total = Column(Float)
    fg_home_ml_prob = Column(Float)
    h1_spread = Column(Float)
    h1_total = Column(Float)
    h1_home_ml_prob = Column(Float)
    # Opening-line snapshot for CLV tracking
    opening_spread = Column(Float)
    opening_total = Column(Float)
    # Closing-line value (filled post-game by scheduler)
    closing_spread = Column(Float)
    closing_total = Column(Float)
    clv_spread = Column(Float)
    clv_total = Column(Float)
    predicted_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    game = relationship("Game", back_populates="predictions")

    __table_args__ = (Index("ix_pred_game_version", "game_id", "model_version"),)


class Injury(Base):
    __tablename__ = "injuries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    status = Column(String(20), nullable=False)  # out, doubtful, questionable, probable
    description = Column(String(255))
    reported_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    player = relationship("Player", back_populates="injuries")
    team = relationship("Team", back_populates="injuries")
