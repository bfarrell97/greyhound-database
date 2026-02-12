"""Core services and components for the Greyhound Racing Analysis System.

This package contains the fundamental building blocks of the application:

Modules:
    config: API keys and configuration constants
    database: SQLite database layer for race and bet data
    live_betting: Real-time bet placement and management
    paper_trading: Simulated trading mode for strategy testing
    predictor: ML model prediction wrapper and feature engineering

Example:
    >>> from src.core.database import GreyhoundDatabase
    >>> from src.core.live_betting import LiveBettingManager
    >>> 
    >>> db = GreyhoundDatabase()
    >>> manager = LiveBettingManager(db)
    >>> manager.place_bet(race_id=123, selection='Fast Freddy', ...)
"""

from src.core.database import GreyhoundDatabase
from src.core.live_betting import LiveBettingManager
from src.core.paper_trading import PaperTradingManager
from src.core.predictor import RacePredictor

__all__ = [
    'GreyhoundDatabase',
    'LiveBettingManager',
    'PaperTradingManager',
    'RacePredictor',
]
