import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
config.RATE_LIMIT_MAX_REQUESTS = 3
config.RATE_LIMIT_WINDOW_SECONDS = 3600

import rate_limit


def setup_function():
    rate_limit.reset_all()  # each test starts with a clean slate, not polluted by the previous one


def test_allows_requests_under_the_limit():
    ip = "1.1.1.1"
    for _ in range(config.RATE_LIMIT_MAX_REQUESTS):
        assert rate_limit.is_rate_limited(ip) is False


def test_blocks_requests_over_the_limit():
    ip = "2.2.2.2"
    for _ in range(config.RATE_LIMIT_MAX_REQUESTS):
        rate_limit.is_rate_limited(ip)
    assert rate_limit.is_rate_limited(ip) is True


def test_different_ips_have_independent_limits():
    ip_a, ip_b = "3.3.3.3", "4.4.4.4"
    for _ in range(config.RATE_LIMIT_MAX_REQUESTS):
        rate_limit.is_rate_limited(ip_a)
    assert rate_limit.is_rate_limited(ip_a) is True
    assert rate_limit.is_rate_limited(ip_b) is False
