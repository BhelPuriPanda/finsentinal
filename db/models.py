from datetime import date, datetime
# pyrefly: ignore [missing-import]
from sqlalchemy import (
    Column,
    String,
    Date,
    DateTime,
    Float,
    Integer,
    Boolean,
    Text,
    UniqueConstraint,
    Index
)
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import DeclarativeBase
# pyrefly: ignore [missing-import]
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class RawPrice(Base):
    __tablename__ = "raw_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adj_close = Column(Float)
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_raw_prices_ticker_date"),
        Index("ix_raw_prices_ticker_date", "ticker", "date"),
    )


class RawNews(Base):
    __tablename__ = "raw_news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    headline = Column(Text, nullable=False)
    source = Column(String(100))
    url = Column(Text)
    vader_compound = Column(Float)
    vader_pos = Column(Float)
    vader_neg = Column(Float)
    vader_neu = Column(Float)
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_raw_news_ticker_date", "ticker", "date"),
    )


class RawMacro(Base):
    __tablename__ = "raw_macro"

    id = Column(Integer, primary_key=True, autoincrement=True)
    series_id = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)
    value = Column(Float)
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("series_id", "date", name="uq_raw_macro_series_date"),
        Index("ix_raw_macro_series_date", "series_id", "date"),
    )


class Features(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    close = Column(Float)
    volume = Column(Float)
    returns_1d = Column(Float)
    returns_5d = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    bb_mid = Column(Float)
    bb_pct = Column(Float)
    atr_14 = Column(Float)
    sentiment_compound_mean = Column(Float)
    sentiment_compound_std = Column(Float)
    sentiment_pos_mean = Column(Float)
    sentiment_neg_mean = Column(Float)
    news_count = Column(Integer)
    cpi = Column(Float)
    fed_funds_rate = Column(Float)
    vix = Column(Float)
    next_day_up = Column(Boolean)
    computed_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_features_ticker_date"),
        Index("ix_features_ticker_date", "ticker", "date"),
    )


class AnomalyAlert(Base):
    __tablename__ = "anomaly_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean)
    top_features = Column(Text)
    detected_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_anomaly_ticker_date"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    prediction_date = Column(Date, nullable=False)
    target_date = Column(Date, nullable=False)
    direction = Column(String(4))
    confidence = Column(Float)
    prob_up = Column(Float)
    prob_down = Column(Float)
    model_version = Column(String(50))
    predicted_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "prediction_date", name="uq_predictions_ticker_date"),
    )