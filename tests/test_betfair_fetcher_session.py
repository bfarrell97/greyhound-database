import time
from types import SimpleNamespace
import betfairlightweight
from src.integration.betfair_fetcher import BetfairOddsFetcher


def test_get_greyhound_markets_relogin_success():
    fetcher = BetfairOddsFetcher()
    calls = {'count': 0}

    class FakeBetting:
        def list_market_catalogue(self, filter, market_projection, max_results, sort=None):
            if calls['count'] == 0:
                calls['count'] += 1
                raise betfairlightweight.exceptions.APIError("ANGX-0003: INVALID_SESSION_INFORMATION")
            return [SimpleNamespace(market_id='M1', market_start_time=None)]

    fetcher.trading.betting = FakeBetting()

    login_calls = {'count': 0}

    def fake_login():
        login_calls['count'] += 1
        return True

    fetcher.login = fake_login

    res = fetcher.get_greyhound_markets()
    assert isinstance(res, list)
    assert len(res) == 1
    assert login_calls['count'] == 1
    assert calls['count'] == 1


def test_get_greyhound_markets_relogin_failure():
    fetcher = BetfairOddsFetcher()
    calls = {'count': 0}

    class FakeBetting:
        def list_market_catalogue(self, filter, market_projection, max_results, sort=None):
            # Always fail with invalid session
            raise betfairlightweight.exceptions.APIError("ANGX-0003: INVALID_SESSION_INFORMATION")

    fetcher.trading.betting = FakeBetting()

    def fake_login():
        return False

    fetcher.login = fake_login

    res = fetcher.get_greyhound_markets()
    assert res == []


def test_keepalive_triggers_relogin_if_failed():
    fetcher = BetfairOddsFetcher()

    calls = {'keep': 0, 'login': 0}

    def fake_keep_alive():
        calls['keep'] += 1
        # simulate a failing keep-alive
        return False

    def fake_login():
        calls['login'] += 1
        return True

    fetcher.keep_alive = fake_keep_alive
    fetcher.login = fake_login

    # Start keepalive with short interval and let it run once
    fetcher.start_keepalive(interval_seconds=1)
    time.sleep(1.5)
    fetcher.stop_keepalive()

    assert calls['keep'] >= 1
    assert calls['login'] >= 1
