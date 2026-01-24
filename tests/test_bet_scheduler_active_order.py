from types import SimpleNamespace
from src.integration.bet_scheduler import BetScheduler


def make_order(**kwargs):
    return SimpleNamespace(**kwargs)


class FetcherStub:
    def __init__(self, orders=None, odds=None, place_result=None):
        self._orders = orders or []
        self._odds = odds or {}
        self._place_result = place_result or {'is_success': False}

    def get_current_orders(self, market_id=None):
        return self._orders

    def get_market_odds(self, market_id):
        return self._odds

    def place_back_bet(self, market_id, selection_id, stake, price):
        return self._place_result


def test_has_active_order_various_shapes():
    # Case A: selection_id with size_remaining > 0
    fetcher = FetcherStub(orders=[make_order(selection_id=101, side='BACK', size_remaining=1.0)])
    s = BetScheduler(fetcher=fetcher)
    assert s._has_active_order('MKTX', 101, 'BACK') is True

    # Case B: selectionId and status=EXECUTABLE
    fetcher = FetcherStub(orders=[make_order(selectionId=102, side='BACK', size_unmatched=0.0, status='EXECUTABLE')])
    s = BetScheduler(fetcher=fetcher)
    assert s._has_active_order('MKTX', 102, 'BACK') is True

    # Case C: different side (LAY) - should not count as BACK active order
    fetcher = FetcherStub(orders=[make_order(selection_id=103, side='LAY', size_remaining=1.0)])
    s = BetScheduler(fetcher=fetcher)
    assert s._has_active_order('MKTX', 103, 'BACK') is False

    # Case D: no matching selection
    fetcher = FetcherStub(orders=[make_order(selection_id=104, side='BACK', size_remaining=0.0)])
    s = BetScheduler(fetcher=fetcher)
    assert s._has_active_order('MKTX', 999, 'BACK') is False


def test_place_bet_skips_when_active_order_exists():
    # Active order present -> scheduler should skip placement and mark SKIPPED
    fetcher = FetcherStub(orders=[make_order(selection_id=200, side='BACK', size_remaining=1.0)])
    s = BetScheduler(fetcher=fetcher)

    bet_id = s.schedule_bet('MKT_SKIP', 200, 'SkipDog', 'Track', 1, '12:00', minutes_before=0)
    bet = s.scheduled_bets[bet_id]

    s._place_bet(bet)

    assert bet.status == 'SKIPPED'
    assert 'Existing active order' in bet.result_message


def test_place_bet_success_when_no_active_order():
    # No active orders, market odds present, place_back_bet succeeds -> PLACED
    fetcher = FetcherStub(orders=[], odds={300: 4.5}, place_result={'is_success': True, 'bet_id': 'BET123'})
    s = BetScheduler(fetcher=fetcher)

    bet_id = s.schedule_bet('MKT_GOOD', 300, 'GoDog', 'Track', 1, '12:00', minutes_before=0)
    bet = s.scheduled_bets[bet_id]

    s._place_bet(bet)

    assert bet.status == 'PLACED'
    assert 'PLACED' in bet.result_message or 'Bet ID' in bet.result_message
