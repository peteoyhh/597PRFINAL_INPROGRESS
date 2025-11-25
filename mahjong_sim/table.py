"""
4-player table composition Monte Carlo simulation.

This module simulates 4 players interacting at a Mahjong table with different
strategy compositions. It extends the single-player simulation to model how
table composition affects individual and average outcomes.

Key design:
- Each player acts independently but interacts through winner/loser selection
- Table composition affects deal-in risk (R) and threat level (T)
- No tile-level simulation - probabilistic interactions only
"""

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
from .simulation import (
    compute_utility,
    DEALER_READY_THRESHOLD
)
from .strategies import defensive_strategy, aggressive_strategy


def adjust_risk_for_composition(base_risk, num_def_players):
    """
    Adjust deal-in risk based on table composition.
    
    More DEF players → lower deal-in risk (they play safer)
    More AGG players → higher deal-in risk (they take more risks)
    
    Args:
        base_risk: Base deal-in risk R
        num_def_players: Number of defensive players at table (0-4)
    
    Returns:
        Adjusted deal-in risk
    """
    # More DEF players reduce risk by up to 30%
    reduction_factor = 1.0 - (num_def_players / 4.0) * 0.3
    return base_risk * reduction_factor


def adjust_fan_growth_for_composition(base_fan, num_def_players):
    """
    Adjust fan growth based on table composition.
    
    More DEF players → lower fan growth (they win earlier)
    More AGG players → higher fan growth (they chase bigger hands)
    
    Args:
        base_fan: Base fan value
        num_def_players: Number of defensive players at table (0-4)
    
    Returns:
        Adjusted fan value (capped appropriately)
    """
    # More AGG players increase fan by up to 20%
    growth_factor = 1.0 + ((4 - num_def_players) / 4.0) * 0.2
    adjusted = int(base_fan * growth_factor)
    # Cap at 16
    return min(adjusted, 16)


def simulate_table_round(players, cfg, dealer_index):
    """
    Simulate a single round with 4 players and track dealer state.
    """
    num_def = sum(1 for p in players if p["strategy_type"] == "DEF")
    theta = num_def / 4.0
    
    player_data = []
    eligible_winners = []
    
    for i, player in enumerate(players):
        is_dealer = i == dealer_index
        Q = sample_hand_quality()
        base_R = sample_deal_in_risk()
        R_adjusted = base_R * (1 - theta * 0.3)
        
        can_win = can_complete_hand(Q)
        
        if not can_win:
            player_data.append({
                "player_idx": i,
                "can_win": False,
                "strategy_accepts": False,
                "fan": 0,
                "R": R_adjusted,
                "Q": Q,
                "player": player,
                "is_dealer": is_dealer
            })
            continue
        
        base_fan = sample_base_fan()
        kong_count = sample_kong_events(lam=0.2, max_kongs=2)
        adjusted_base_fan = adjust_fan_growth_for_composition(base_fan, num_def)
        total_fan = compute_total_fan(adjusted_base_fan, kong_count, max_total_fan=16)
        
        strategy_callable = player["strategy"]
        if hasattr(strategy_callable, "should_hu"):
            strategy_accepts = strategy_callable.should_hu(total_fan, R_adjusted)
        else:
            strategy_accepts = strategy_callable(total_fan)
        
        player_data.append({
            "player_idx": i,
            "can_win": True,
            "strategy_accepts": strategy_accepts,
            "fan": total_fan,
            "R": R_adjusted,
            "Q": Q,
            "player": player,
            "is_dealer": is_dealer
        })
        
        if strategy_accepts:
            eligible_winners.append({
                "player_idx": i,
                "fan": total_fan,
                "player": player,
                "is_dealer": is_dealer
            })
    
    dealer_ready = player_data[dealer_index]["Q"] >= DEALER_READY_THRESHOLD if dealer_index < len(player_data) else False
    
    if len(eligible_winners) == 0:
        results = []
        for i, player in enumerate(players):
            data = player_data[i]
            missed_hu = data.get("can_win", False) and not data.get("strategy_accepts", False)
            results.append({
                "profit": 0.0,
                "utility": compute_utility(0.0, missed_hu=missed_hu, deal_in_as_loser=False),
                "fan": 0,
                "won": False,
                "deal_in_as_winner": False,
                "deal_in_as_loser": False,
                "missed_hu": missed_hu
            })
        return results, {
            "winner_index": None,
            "winner_is_dealer": False,
            "dealer_ready": dealer_ready,
            "dealer_continues": dealer_ready,
            "is_draw": True
        }
    
    if len(eligible_winners) == 1:
        winner = eligible_winners[0]
    else:
        fan_values = [w["fan"] for w in eligible_winners]
        weights = np.array(fan_values, dtype=float)
        total_weight = np.sum(weights)
        if total_weight <= 0:
            weights = np.ones(len(eligible_winners)) / len(eligible_winners)
        else:
            weights = weights / total_weight
        winner_idx = np.random.choice(len(eligible_winners), p=weights)
        winner = eligible_winners[winner_idx]
    
    winner_data = player_data[winner["player_idx"]]
    winner_fan = winner["fan"]
    score = compute_score(winner_fan, cfg["base_points"])
    deal_in_occurred = np.random.rand() < winner_data["R"]
    
    results = []
    
    round_profits = [0.0] * len(players)
    round_utilities = [0.0] * len(players)
    round_win_flags = [False] * len(players)
    round_deal_in_win = [False] * len(players)
    round_deal_in_loss = [False] * len(players)
    round_fans = [0] * len(players)
    round_missed = []
    for data in player_data:
        missed_hu = data.get("can_win", False) and not data.get("strategy_accepts", False)
        round_missed.append(missed_hu)
    
    if deal_in_occurred:
        losers = [p for i, p in enumerate(players) if i != winner["player_idx"]]
        loser_Rs = [player_data[i]["R"] for i in range(len(players)) if i != winner["player_idx"]]
        loser_weights = np.array(loser_Rs)
        loser_weights = loser_weights / loser_weights.sum()
        loser_idx = np.random.choice(len(losers), p=loser_weights)
        loser_player_idx = [i for i in range(len(players)) if i != winner["player_idx"]][loser_idx]
        
        loser_profit = compute_loser_cost(
            score,
            cfg["penalty_deal_in"],
            is_deal_in_loser=True
        )
        round_profits[loser_player_idx] += loser_profit
        round_profits[winner["player_idx"]] -= loser_profit  # loser_profit is negative
        round_win_flags[winner["player_idx"]] = True
        round_deal_in_win[winner["player_idx"]] = True
        round_deal_in_loss[loser_player_idx] = True
        round_fans[winner["player_idx"]] = winner_fan
    else:
        opponent_cost = compute_loser_cost(
            score,
            cfg["penalty_deal_in"],
            is_deal_in_loser=False
        )
        for i in range(len(players)):
            if i == winner["player_idx"]:
                round_profits[i] -= opponent_cost * (len(players) - 1)
                round_win_flags[i] = True
                round_fans[i] = winner_fan
            else:
                round_profits[i] += opponent_cost
    
    for i in range(len(players)):
        round_utilities[i] = compute_utility(
            round_profits[i],
            missed_hu=round_missed[i],
            deal_in_as_loser=round_deal_in_loss[i]
        )
        results.append({
            "profit": round_profits[i],
            "utility": round_utilities[i],
            "fan": round_fans[i],
            "won": round_win_flags[i],
            "deal_in_as_winner": round_deal_in_win[i],
            "deal_in_as_loser": round_deal_in_loss[i],
            "missed_hu": round_missed[i]
        })
    
    return results, {
        "winner_index": winner["player_idx"],
        "winner_is_dealer": winner["player_idx"] == dealer_index,
        "dealer_ready": player_data[dealer_index]["Q"] >= DEALER_READY_THRESHOLD,
        "dealer_continues": winner["player_idx"] == dealer_index,
        "is_draw": False
    }


def _run_table(players, cfg, rounds_per_trial, baseline_utility=200):
    dealer_index = 0

    player_count = len(players)

    all_profits = [[] for _ in range(player_count)]
    all_utilities = [[] for _ in range(player_count)]
    all_fans = [[] for _ in range(player_count)]
    all_wins = [[] for _ in range(player_count)]
    all_deal_in_as_winner = [[] for _ in range(player_count)]
    all_deal_in_as_loser = [[] for _ in range(player_count)]
    all_missed_hu = [[] for _ in range(player_count)]

    dealer_round_stats = {
        "profits": [],
        "utilities": [],
        "wins": [],
        "deal_in_as_winner": [],
        "deal_in_as_loser": [],
        "missed_hu": [],
        "fans": []
    }

    non_dealer_round_stats = {
        "profits": [],
        "utilities": [],
        "wins": [],
        "deal_in_as_winner": [],
        "deal_in_as_loser": [],
        "missed_hu": [],
        "fans": []
    }

    for _ in range(rounds_per_trial):
        round_results, round_meta = simulate_table_round(players, cfg, dealer_index)

        for i, result in enumerate(round_results):
            all_profits[i].append(result["profit"])
            all_utilities[i].append(result["utility"])
            all_fans[i].append(result["fan"])
            all_wins[i].append(result["won"])
            all_deal_in_as_winner[i].append(result["deal_in_as_winner"])
            all_deal_in_as_loser[i].append(result["deal_in_as_loser"])
            all_missed_hu[i].append(result["missed_hu"])

            target_stats = dealer_round_stats if i == dealer_index else non_dealer_round_stats
            target_stats["profits"].append(result["profit"])
            target_stats["utilities"].append(result["utility"])
            target_stats["wins"].append(result["won"])
            target_stats["deal_in_as_winner"].append(result["deal_in_as_winner"])
            target_stats["deal_in_as_loser"].append(result["deal_in_as_loser"])
            target_stats["missed_hu"].append(result["missed_hu"])
            target_stats["fans"].append(result["fan"])

        if not round_meta["dealer_continues"]:
            dealer_index = (dealer_index + 1) % player_count

    def_stats = {
        "profits": [],
        "utilities": [],
        "fans": [],
        "wins": [],
        "deal_in_as_winner": [],
        "deal_in_as_loser": [],
        "missed_hu": []
    }

    agg_stats = {
        "profits": [],
        "utilities": [],
        "fans": [],
        "wins": [],
        "deal_in_as_winner": [],
        "deal_in_as_loser": [],
        "missed_hu": []
    }

    per_player_stats = []

    for i, player in enumerate(players):
        profit_sum = np.sum(all_profits[i])
        utility_sum = baseline_utility + np.sum(all_utilities[i])
        wins = all_wins[i]
        fans = [f for f in all_fans[i] if f > 0]

        stats = {
            "profit": profit_sum,
            "utility": utility_sum,
            "mean_fan": np.mean(fans) if len(fans) > 0 else 0.0,
            "win_rate": np.mean(wins),
            "deal_in_rate": np.mean(all_deal_in_as_winner[i]),
            "deal_in_loss_rate": np.mean(all_deal_in_as_loser[i]),
            "missed_win_rate": np.mean(all_missed_hu[i]),
            "fan_distribution": all_fans[i]
        }

        per_player_stats.append({
            "player_index": i,
            "strategy_type": player.get("strategy_type"),
            **stats
        })

        if player.get("strategy_type") == "DEF":
            def_stats["profits"].append(stats["profit"])
            def_stats["utilities"].append(stats["utility"])
            def_stats["fans"].extend(fans)
            def_stats["wins"].append(stats["win_rate"])
            def_stats["deal_in_as_winner"].append(stats["deal_in_rate"])
            def_stats["deal_in_as_loser"].append(stats["deal_in_loss_rate"])
            def_stats["missed_hu"].append(stats["missed_win_rate"])
        elif player.get("strategy_type") == "AGG":
            agg_stats["profits"].append(stats["profit"])
            agg_stats["utilities"].append(stats["utility"])
            agg_stats["fans"].extend(fans)
            agg_stats["wins"].append(stats["win_rate"])
            agg_stats["deal_in_as_winner"].append(stats["deal_in_rate"])
            agg_stats["deal_in_as_loser"].append(stats["deal_in_loss_rate"])
            agg_stats["missed_hu"].append(stats["missed_win_rate"])

    return {
        "defensive": def_stats,
        "aggressive": agg_stats,
        "dealer": dealer_round_stats,
        "non_dealer": non_dealer_round_stats,
        "per_player": per_player_stats
    }


def simulate_table(composition, cfg, baseline_utility=200):
    """
    Simulate a full table trial (200 rounds) with given composition.
    """
    num_def = composition
    num_agg = 4 - composition

    players = []
    fan_min = cfg["fan_min"]
    t_fan_threshold = cfg["t_fan_threshold"]

    for _ in range(num_def):
        players.append({
            "strategy": lambda f, fm=fan_min: defensive_strategy(f, fm),
            "strategy_type": "DEF"
        })
    for _ in range(num_agg):
        players.append({
            "strategy": lambda f, th=t_fan_threshold: aggressive_strategy(f, th),
            "strategy_type": "AGG"
        })

    result = _run_table(players, cfg, cfg["rounds_per_trial"], baseline_utility)
    result["composition"] = composition
    return result


def simulate_custom_table(players, cfg, rounds_per_trial=None, baseline_utility=200):
    """
    Run a table simulation with a custom list of players.
    """
    rounds = rounds_per_trial or cfg["rounds_per_trial"]
    result = _run_table(players, cfg, rounds, baseline_utility)
    result["composition"] = None
    return result


def run_composition_experiments(cfg, num_trials=1000):
    """
    Run experiments for all table compositions (θ = 0 to 4).
    
    Args:
        cfg: Configuration dictionary
        num_trials: Number of trials per composition (default: 1000)
    
    Returns:
        Dictionary with results for each composition
    """
    compositions = [0, 1, 2, 3, 4]  # θ = number of DEF players
    results = {}
    
    for composition in compositions:
        print(f"Running composition θ={composition} ({composition} DEF, {4-composition} AGG)...")
        
        all_def_profits = []
        all_def_utilities = []
        all_agg_profits = []
        all_agg_utilities = []
        all_def_win_rates = []
        all_agg_win_rates = []
        all_def_deal_in_rates = []
        all_agg_deal_in_rates = []
        all_def_missed_hu_rates = []
        all_agg_missed_hu_rates = []
        all_def_fans = []
        all_agg_fans = []
        all_dealer_profits = []
        all_dealer_utilities = []
        all_dealer_wins = []
        all_dealer_deal_in_rates = []
        all_dealer_deal_in_loss_rates = []
        all_dealer_missed_hu = []
        all_dealer_fans = []
        all_non_dealer_profits = []
        all_non_dealer_utilities = []
        all_non_dealer_wins = []
        all_non_dealer_deal_in_rates = []
        all_non_dealer_deal_in_loss_rates = []
        all_non_dealer_missed_hu = []
        all_non_dealer_fans = []
        
        for _ in range(num_trials):
            trial_result = simulate_table(composition, cfg)
            
            if len(trial_result["defensive"]["profits"]) > 0:
                all_def_profits.extend(trial_result["defensive"]["profits"])
                all_def_utilities.extend(trial_result["defensive"]["utilities"])
                all_def_win_rates.extend(trial_result["defensive"]["wins"])
                all_def_deal_in_rates.extend(trial_result["defensive"]["deal_in_as_winner"])
                all_def_missed_hu_rates.extend(trial_result["defensive"]["missed_hu"])
                all_def_fans.extend(trial_result["defensive"]["fans"])
            
            if len(trial_result["aggressive"]["profits"]) > 0:
                all_agg_profits.extend(trial_result["aggressive"]["profits"])
                all_agg_utilities.extend(trial_result["aggressive"]["utilities"])
                all_agg_win_rates.extend(trial_result["aggressive"]["wins"])
                all_agg_deal_in_rates.extend(trial_result["aggressive"]["deal_in_as_winner"])
                all_agg_missed_hu_rates.extend(trial_result["aggressive"]["missed_hu"])
                all_agg_fans.extend(trial_result["aggressive"]["fans"])
            
            dealer_stats = trial_result["dealer"]
            non_dealer_stats = trial_result["non_dealer"]
            
            all_dealer_profits.extend(dealer_stats["profits"])
            all_dealer_utilities.extend(dealer_stats["utilities"])
            all_dealer_wins.extend(dealer_stats["wins"])
            all_dealer_deal_in_rates.extend(dealer_stats["deal_in_as_winner"])
            all_dealer_deal_in_loss_rates.extend(dealer_stats["deal_in_as_loser"])
            all_dealer_missed_hu.extend(dealer_stats["missed_hu"])
            all_dealer_fans.extend(dealer_stats["fans"])
            
            all_non_dealer_profits.extend(non_dealer_stats["profits"])
            all_non_dealer_utilities.extend(non_dealer_stats["utilities"])
            all_non_dealer_wins.extend(non_dealer_stats["wins"])
            all_non_dealer_deal_in_rates.extend(non_dealer_stats["deal_in_as_winner"])
            all_non_dealer_deal_in_loss_rates.extend(non_dealer_stats["deal_in_as_loser"])
            all_non_dealer_missed_hu.extend(non_dealer_stats["missed_hu"])
            all_non_dealer_fans.extend(non_dealer_stats["fans"])
        
        results[composition] = {
            "defensive": {
                "mean_profit": np.mean(all_def_profits) if len(all_def_profits) > 0 else 0.0,
                "std_profit": np.std(all_def_profits) if len(all_def_profits) > 0 else 0.0,
                "mean_utility": np.mean(all_def_utilities) if len(all_def_utilities) > 0 else 0.0,
                "std_utility": np.std(all_def_utilities) if len(all_def_utilities) > 0 else 0.0,
                "win_rate": np.mean(all_def_win_rates) if len(all_def_win_rates) > 0 else 0.0,
                "deal_in_rate": np.mean(all_def_deal_in_rates) if len(all_def_deal_in_rates) > 0 else 0.0,
                "missed_hu_rate": np.mean(all_def_missed_hu_rates) if len(all_def_missed_hu_rates) > 0 else 0.0,
                "mean_fan": np.mean(all_def_fans) if len(all_def_fans) > 0 else 0.0,
                "fan_distribution": all_def_fans
            },
            "aggressive": {
                "mean_profit": np.mean(all_agg_profits) if len(all_agg_profits) > 0 else 0.0,
                "std_profit": np.std(all_agg_profits) if len(all_agg_profits) > 0 else 0.0,
                "mean_utility": np.mean(all_agg_utilities) if len(all_agg_utilities) > 0 else 0.0,
                "std_utility": np.std(all_agg_utilities) if len(all_agg_utilities) > 0 else 0.0,
                "win_rate": np.mean(all_agg_win_rates) if len(all_agg_win_rates) > 0 else 0.0,
                "deal_in_rate": np.mean(all_agg_deal_in_rates) if len(all_agg_deal_in_rates) > 0 else 0.0,
                "missed_hu_rate": np.mean(all_agg_missed_hu_rates) if len(all_agg_missed_hu_rates) > 0 else 0.0,
                "mean_fan": np.mean(all_agg_fans) if len(all_agg_fans) > 0 else 0.0,
                "fan_distribution": all_agg_fans
            },
            "dealer": {
                "mean_profit": np.mean(all_dealer_profits) if len(all_dealer_profits) > 0 else 0.0,
                "std_profit": np.std(all_dealer_profits) if len(all_dealer_profits) > 0 else 0.0,
                "mean_utility": np.mean(all_dealer_utilities) if len(all_dealer_utilities) > 0 else 0.0,
                "std_utility": np.std(all_dealer_utilities) if len(all_dealer_utilities) > 0 else 0.0,
                "win_rate": np.mean(all_dealer_wins) if len(all_dealer_wins) > 0 else 0.0,
                "deal_in_rate": np.mean(all_dealer_deal_in_rates) if len(all_dealer_deal_in_rates) > 0 else 0.0,
                "deal_in_loss_rate": np.mean(all_dealer_deal_in_loss_rates) if len(all_dealer_deal_in_loss_rates) > 0 else 0.0,
                "missed_hu_rate": np.mean(all_dealer_missed_hu) if len(all_dealer_missed_hu) > 0 else 0.0,
                "mean_fan": np.mean([f for f in all_dealer_fans if f > 0]) if len(all_dealer_fans) > 0 else 0.0,
                "fan_distribution": all_dealer_fans
            },
            "non_dealer": {
                "mean_profit": np.mean(all_non_dealer_profits) if len(all_non_dealer_profits) > 0 else 0.0,
                "std_profit": np.std(all_non_dealer_profits) if len(all_non_dealer_profits) > 0 else 0.0,
                "mean_utility": np.mean(all_non_dealer_utilities) if len(all_non_dealer_utilities) > 0 else 0.0,
                "std_utility": np.std(all_non_dealer_utilities) if len(all_non_dealer_utilities) > 0 else 0.0,
                "win_rate": np.mean(all_non_dealer_wins) if len(all_non_dealer_wins) > 0 else 0.0,
                "deal_in_rate": np.mean(all_non_dealer_deal_in_rates) if len(all_non_dealer_deal_in_rates) > 0 else 0.0,
                "deal_in_loss_rate": np.mean(all_non_dealer_deal_in_loss_rates) if len(all_non_dealer_deal_in_loss_rates) > 0 else 0.0,
                "missed_hu_rate": np.mean(all_non_dealer_missed_hu) if len(all_non_dealer_missed_hu) > 0 else 0.0,
                "mean_fan": np.mean([f for f in all_non_dealer_fans if f > 0]) if len(all_non_dealer_fans) > 0 else 0.0,
                "fan_distribution": all_non_dealer_fans
            }
        }
    
    return results

