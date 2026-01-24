import pytest
from src.integration.bet_scheduler import BetScheduler


def test_schedule_bet_duplicate():
    # Simple fetcher stub (no network calls during scheduling)
    class DummyFetcher:
        pass

    fetcher = DummyFetcher()
    sched = BetScheduler(fetcher=fetcher)

    bet_id_1 = sched.schedule_bet('MKT1', 101, 'Speedy', 'SomeTrack', 1, '12:00', minutes_before=0)
    bet_id_2 = sched.schedule_bet('MKT1', 101, 'Speedy', 'SomeTrack', 1, '12:00', minutes_before=0)

    # Duplicate scheduling should return the same id when allow_duplicate=False
    assert bet_id_1 == bet_id_2

    # When allow_duplicate=True, a new scheduled bet should be created
    bet_id_3 = sched.schedule_bet('MKT1', 101, 'Speedy', 'SomeTrack', 1, '12:00', minutes_before=0, allow_duplicate=True)
    assert bet_id_3 != bet_id_1
