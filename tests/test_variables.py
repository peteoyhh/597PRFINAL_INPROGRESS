from mahjong_sim import variables as v
import numpy as np


def test_quality_range():
    q = v.sample_hand_quality()
    assert 0 <= q <= 1


def test_base_fan_distribution():
    """Test that base fan follows discrete Beijing Mahjong distribution."""
    fans = [v.sample_base_fan() for _ in range(1000)]
    unique_fans = set(fans)
    # Should only contain valid fan values
    valid_fans = {0, 1, 2, 4, 8, 13}
    assert unique_fans.issubset(valid_fans)


def test_kong_events():
    """Test that Kong events are non-negative integers."""
    k = v.sample_kong_events()
    assert isinstance(k, (int, np.integer))
    assert k >= 0


def test_can_complete_hand():
    """Test that can_complete_hand returns boolean."""
    result = v.can_complete_hand(0.5)
    assert isinstance(result, (bool, np.bool_))

