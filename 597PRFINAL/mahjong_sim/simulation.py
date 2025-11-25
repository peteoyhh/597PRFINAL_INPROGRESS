from typing import Dict, Callable, Union, Any, Optional
import numpy as np
from .variables import (
    sample_hand_quality,
    sample_base_fan,
    sample_kong_events,
    sample_deal_in_risk,
    can_complete_hand
)
from .scoring import (
    compute_score,
    compute_winner_profit,
    compute_loser_cost,
    compute_total_fan
)


def compute_utility(profit: float, missed_hu: bool, deal_in_as_loser: bool, 
                   missed_penalty: float = 0.2, deal_in_penalty: float = 0.5) -> float:
    """
    Compute utility using strong concave reward function with minimal penalties.
    
    U = concave_reward(profit) - small_penalties
    
    Key design principles:
    - Strong concave (non-linear) reward function for positive profits
    - Utility is monotone increasing with profit
    - Penalties are minimal and do not overpower rewards
    - Winners always have strongly positive utility contribution
    
    Args:
        profit: Profit from the hand (can be negative if lost)
        missed_hu: Boolean indicating if player missed a possible Hu
        deal_in_as_loser: Boolean indicating if player dealt in as loser
        missed_penalty: Penalty for missing a Hu (default: 0.2, greatly reduced)
        deal_in_penalty: Penalty for dealing in as loser (default: 0.5, greatly reduced)
    
    Returns:
        Utility value
    """
    # Strong concave reward function for positive profits
    # Using sqrt for stronger concavity than log, ensuring non-linear positive rewards
    if profit > 0:
        # Positive profit: strong concave utility (sqrt is more concave than log)
        # This ensures diminishing returns but always positive and non-linear
        utility = np.sqrt(profit)  # Strong concave, always positive for profit > 0
    elif profit < 0:
        # Negative profit: concave penalty (less severe than linear)
        # Use sqrt of absolute value to make penalty concave (less harsh)
        utility = -np.sqrt(abs(profit))  # Concave penalty, less severe than linear
    else:
        # Zero profit
        utility = 0.0
    
    # Apply minimal penalties only when they occur
    # Penalties are greatly reduced to not overpower the concave rewards
    if missed_hu:
        utility -= missed_penalty
    
    if deal_in_as_loser:
        utility -= deal_in_penalty
    
    return utility


DEALER_READY_THRESHOLD = 0.6


def simulate_round(player_strategy: Callable[[Union[int, float]], bool], 
                  cfg: Dict[str, Any], is_dealer: bool = False) -> Dict[str, Union[float, int, bool]]:
    """
    Simulate a single round of Beijing Mahjong.
    
    Logic flow:
    1. Sample hand quality Q
    2. Check if hand can be completed (probability = Q)
    3. If can complete, sample fan and check strategy
    4. If strategy accepts, determine win/loss and compute scores
    5. Calculate profit and utility
    
    Args:
        player_strategy: Strategy function that takes fan and returns bool
        cfg: Configuration dictionary
    
    Returns:
        Dictionary with round results:
        - profit: Profit from this round
        - utility: Utility from this round
        - fan: Total fan count (0 if didn't win)
        - won: Boolean indicating if player won
        - deal_in_as_winner: Boolean indicating if won via deal-in
        - deal_in_as_loser: Boolean indicating if lost via deal-in
        - missed_hu: Boolean indicating if missed a possible Hu
    """
    # Sample random variables
    Q = sample_hand_quality()
    R = sample_deal_in_risk()
    
    # Check if hand can be completed
    can_win = can_complete_hand(Q)
    
    if not can_win:
        # Hand cannot be completed - no win possible
        return {
            "profit": 0.0,
            "utility": 0.0,
            "fan": 0,
            "won": False,
            "deal_in_as_winner": False,
            "deal_in_as_loser": False,
            "missed_hu": False
        }
    
    # Hand can be completed - sample fan and Kong
    base_fan = sample_base_fan()
    kong_count = sample_kong_events(lam=0.2, max_kongs=2)  # Cap Kongs at 2
    total_fan = compute_total_fan(base_fan, kong_count, max_total_fan=16)  # Cap total fan at 16
    
    # Check if strategy accepts this fan level
    strategy_accepts = player_strategy(total_fan)
    
    if not strategy_accepts:
        # Strategy rejected - missed Hu opportunity
        # Check if opponent wins (deal-in risk)
        deal_in_as_loser = np.random.rand() < R
        
        profit = 0.0
        if deal_in_as_loser:
            # Opponent won, we dealt in
            # Sample opponent's fan (simplified: assume average fan)
            opponent_base_fan = sample_base_fan()
            opponent_kong = sample_kong_events(lam=0.2, max_kongs=2)
            opponent_fan = compute_total_fan(opponent_base_fan, opponent_kong, max_total_fan=16)
            opponent_score = compute_score(opponent_fan, cfg["base_points"])
            profit = compute_loser_cost(
                opponent_score,
                cfg["penalty_deal_in"],
                is_deal_in_loser=True
            )
        
        utility = compute_utility(
            profit,
            missed_hu=True,
            deal_in_as_loser=deal_in_as_loser
        )
        
        return {
            "profit": profit,
            "utility": utility,
            "fan": 0,
            "won": False,
            "deal_in_as_winner": False,
            "deal_in_as_loser": deal_in_as_loser,
            "missed_hu": True
        }
    
    # Strategy accepts - player wins
    score = compute_score(total_fan, cfg["base_points"])
    
    # Determine win type: self-draw or deal-in
    # Deal-in occurs with probability R
    deal_in_occurred = np.random.rand() < R
    
    if deal_in_occurred:
        # Won via deal-in
        profit = compute_winner_profit(
            score,
            is_self_draw=False,
            deal_in_occurred=True
        )
        utility = compute_utility(
            profit,
            missed_hu=False,
            deal_in_as_loser=False
        )
        
        return {
            "profit": profit,
            "utility": utility,
            "fan": total_fan,
            "won": True,
            "deal_in_as_winner": True,
            "deal_in_as_loser": False,
            "missed_hu": False
        }
    else:
        # Won via self-draw
        profit = compute_winner_profit(
            score,
            is_self_draw=True,
            deal_in_occurred=False
        )
        utility = compute_utility(
            profit,
            missed_hu=False,
            deal_in_as_loser=False
        )
        
        return {
            "profit": profit,
            "utility": utility,
            "fan": total_fan,
            "won": True,
            "deal_in_as_winner": False,
            "deal_in_as_loser": False,
            "missed_hu": False
        }


def run_simulation(strategy_fn: Callable[[Union[int, float]], bool], 
                  cfg: Dict[str, Any], baseline_utility: int = 200) -> Dict[str, Any]:
    """
    Run a single trial (multiple rounds) of the simulation.
    
    Args:
        strategy_fn: Strategy function
        cfg: Configuration dictionary
        baseline_utility: Baseline emotional utility at start (default: 200)
    
    Returns:
        Dictionary with aggregated statistics for this trial
    """
    profits = []
    utilities = []
    fans = []
    wins = []
    deal_in_as_winner = []
    deal_in_as_loser = []
    missed_hu = []
    
    for _ in range(cfg["rounds_per_trial"]):
        result = simulate_round(strategy_fn, cfg)
        profits.append(result["profit"])
        utilities.append(result["utility"])  # Incremental utility per round
        fans.append(result["fan"])
        wins.append(result["won"])
        deal_in_as_winner.append(result["deal_in_as_winner"])
        deal_in_as_loser.append(result["deal_in_as_loser"])
        missed_hu.append(result["missed_hu"])
    
    # Incremental utility cannot drive total utility below the baseline.
    # Clamp at zero so baseline_utility acts as a guaranteed floor.
    incremental_utility = np.sum(utilities)
    incremental_utility = max(0.0, incremental_utility)
    total_utility = baseline_utility + incremental_utility
    
    return {
        "profit": np.sum(profits),
        "utility": total_utility,
        "mean_fan": np.mean([f for f in fans if f > 0]) if any(f > 0 for f in fans) else 0.0,
        "win_rate": np.mean(wins),
        "deal_in_rate": np.mean(deal_in_as_winner),  # Deal-in as winner
        "deal_in_loss_rate": np.mean(deal_in_as_loser),  # Deal-in as loser
        "missed_win_rate": np.mean(missed_hu),
        "fan_distribution": fans
    }


def run_multiple_trials(strategy_fn: Callable[[Union[int, float]], bool], 
                       cfg: Dict[str, Any], num_trials: Optional[int] = None) -> Dict[str, Any]:
    """
    Run multiple trials and aggregate results for statistical analysis.
    
    Args:
        strategy_fn: Strategy function
        cfg: Configuration dictionary
        num_trials: Number of trials (defaults to cfg["trials"])
    
    Returns:
        Dictionary with aggregated statistics across all trials
    """
    if num_trials is None:
        num_trials = cfg.get("trials", 2000)
    
    all_profits = []
    all_utilities = []
    all_mean_fans = []
    all_win_rates = []
    all_deal_in_rates = []
    all_deal_in_loss_rates = []
    all_missed_win_rates = []
    all_fan_distributions = []
    
    for _ in range(num_trials):
        result = run_simulation(strategy_fn, cfg)
        all_profits.append(result["profit"])
        all_utilities.append(result["utility"])
        all_mean_fans.append(result["mean_fan"])
        all_win_rates.append(result["win_rate"])
        all_deal_in_rates.append(result["deal_in_rate"])
        all_deal_in_loss_rates.append(result["deal_in_loss_rate"])
        all_missed_win_rates.append(result["missed_win_rate"])
        all_fan_distributions.extend(result["fan_distribution"])
    
    return {
        "profits": np.array(all_profits),
        "utilities": np.array(all_utilities),
        "mean_fans": np.array(all_mean_fans),
        "win_rates": np.array(all_win_rates),
        "deal_in_rates": np.array(all_deal_in_rates),
        "deal_in_loss_rates": np.array(all_deal_in_loss_rates),
        "missed_win_rates": np.array(all_missed_win_rates),
        "fan_distribution": np.array(all_fan_distributions),
        "num_trials": num_trials
    }
