"""Extended tests for mahjong_sim.simulation module."""

import numpy as np
import pytest
from mahjong_sim.simulation import (
    compute_utility,
    run_simulation,
    run_multiple_trials
)
from mahjong_sim.strategies import defensive_strategy, aggressive_strategy


def test_compute_utility_positive_profit():
    """Test utility computation with positive profit."""
    utility = compute_utility(profit=100.0, missed_hu=False, deal_in_as_loser=False)
    assert utility > 0
    assert utility == pytest.approx(np.sqrt(100.0))


def test_compute_utility_negative_profit():
    """Test utility computation with negative profit."""
    utility = compute_utility(profit=-50.0, missed_hu=False, deal_in_as_loser=False)
    assert utility < 0
    assert utility == pytest.approx(-np.sqrt(50.0))


def test_compute_utility_zero_profit():
    """Test utility computation with zero profit."""
    utility = compute_utility(profit=0.0, missed_hu=False, deal_in_as_loser=False)
    assert utility == 0.0


def test_compute_utility_with_missed_hu():
    """Test utility computation with missed Hu penalty."""
    utility_with_penalty = compute_utility(
        profit=100.0, missed_hu=True, deal_in_as_loser=False
    )
    utility_without_penalty = compute_utility(
        profit=100.0, missed_hu=False, deal_in_as_loser=False
    )
    assert utility_with_penalty < utility_without_penalty
    assert utility_with_penalty == pytest.approx(utility_without_penalty - 0.2)


def test_compute_utility_with_deal_in():
    """Test utility computation with deal-in penalty."""
    utility_with_penalty = compute_utility(
        profit=100.0, missed_hu=False, deal_in_as_loser=True
    )
    utility_without_penalty = compute_utility(
        profit=100.0, missed_hu=False, deal_in_as_loser=False
    )
    assert utility_with_penalty < utility_without_penalty
    assert utility_with_penalty == pytest.approx(utility_without_penalty - 0.5)


def test_run_simulation_basic():
    """Test run_simulation returns expected keys."""
    cfg = {
        "base_points": 1,
        "fan_min": 1,
        "t_fan_threshold": 3,
        "alpha": 0.1,
        "penalty_deal_in": 3,
        "rounds_per_trial": 10
    }
    result = run_simulation(lambda f: defensive_strategy(f, 1), cfg)
    
    assert "profit" in result
    assert "utility" in result
    assert "mean_fan" in result
    assert "win_rate" in result
    assert "deal_in_rate" in result
    assert "deal_in_loss_rate" in result
    assert "missed_win_rate" in result
    assert "fan_distribution" in result
    assert 0 <= result["win_rate"] <= 1.0


def test_run_simulation_utility_baseline():
    """Test that run_simulation includes baseline utility."""
    cfg = {
        "base_points": 1,
        "fan_min": 1,
        "t_fan_threshold": 3,
        "alpha": 0.1,
        "penalty_deal_in": 3,
        "rounds_per_trial": 10
    }
    result = run_simulation(lambda f: defensive_strategy(f, 1), cfg, baseline_utility=200)
    
    # Utility should be at least baseline (200) plus any incremental utilities
    assert result["utility"] >= 200


def test_run_multiple_trials():
    """Test run_multiple_trials returns expected structure."""
    cfg = {
        "base_points": 1,
        "fan_min": 1,
        "t_fan_threshold": 3,
        "alpha": 0.1,
        "penalty_deal_in": 3,
        "rounds_per_trial": 10,
        "trials": 5
    }
    results = run_multiple_trials(lambda f: defensive_strategy(f, 1), cfg, num_trials=5)
    
    assert "profits" in results
    assert "utilities" in results
    assert "mean_fans" in results
    assert "win_rates" in results
    assert len(results["profits"]) == 5
    assert len(results["utilities"]) == 5


def test_run_multiple_trials_custom_num():
    """Test run_multiple_trials with custom num_trials."""
    cfg = {
        "base_points": 1,
        "fan_min": 1,
        "t_fan_threshold": 3,
        "alpha": 0.1,
        "penalty_deal_in": 3,
        "rounds_per_trial": 10,
        "trials": 100
    }
    results = run_multiple_trials(lambda f: defensive_strategy(f, 1), cfg, num_trials=3)
    
    assert len(results["profits"]) == 3  # Should use num_trials parameter, not cfg["trials"]

