from typing import Union
import numpy as np


def sample_hand_quality(a: int = 2, b: int = 2) -> float:
    """
    Sample hand quality Q in [0, 1].
    
    Q represents the probability that a hand can be completed (ready to Hu).
    Higher Q means better hand quality and higher chance of winning.
    
    Args:
        a: Beta distribution parameter (default: 2)
        b: Beta distribution parameter (default: 2)
    
    Returns:
        Q: Hand quality value in [0, 1]
    """
    return np.random.beta(a, b)


def sample_base_fan() -> int:
    """
    Sample base fan value using discrete Beijing Mahjong distribution.
    
    Fan distribution:
    - 0 fan (Pi Hu):           40%
    - 1 fan (basic patterns):  30%
    - 2 fan:                   20%
    - 4 fan:                   7%
    - 8 fan:                   2%
    - 13 fan:                  1%
    
    Returns:
        Base fan value (0, 1, 2, 4, 8, or 13)
    """
    fan_values = [0, 1, 2, 4, 8, 13]
    probabilities = [0.40, 0.30, 0.20, 0.07, 0.02, 0.01]
    
    return np.random.choice(fan_values, p=probabilities)


def sample_kong_events(lam: float = 0.2, max_kongs: int = 2) -> int:
    """
    Sample number of Kong events per hand.
    
    Kong adds bonus fan to the hand. Typically 0-2 Kongs per hand.
    Capped to prevent fan explosion.
    
    Args:
        lam: Poisson parameter (default: 0.2)
        max_kongs: Maximum number of Kongs allowed (default: 2)
    
    Returns:
        Number of Kong events (non-negative integer, capped at max_kongs)
    """
    kongs = np.random.poisson(lam)
    return min(kongs, max_kongs)


def sample_deal_in_risk(a: int = 2, b: int = 5) -> float:
    """
    Sample deal-in risk R in [0, 1].
    
    R represents the probability that the player will deal-in (discard a winning tile).
    Higher R means higher risk of dealing in to opponents.
    
    Args:
        a: Beta distribution parameter (default: 2)
        b: Beta distribution parameter (default: 5)
    
    Returns:
        R: Deal-in risk probability in [0, 1]
    """
    return np.random.beta(a, b)


def sample_threat_level(k: int = 2, theta: float = 1.0) -> float:
    """
    Sample threat level T (pressure from opponents being close to win).
    
    This is used for advanced strategies but not critical for basic simulation.
    
    Args:
        k: Gamma distribution shape parameter (default: 2)
        theta: Gamma distribution scale parameter (default: 1.0)
    
    Returns:
        Threat level (non-negative float)
    """
    return np.random.gamma(k, theta)


def can_complete_hand(hand_quality: float) -> bool:
    """
    Determine if a hand can be completed based on hand quality Q.
    
    A hand can only be completed ("ready to Hu") with probability = Q.
    
    Args:
        hand_quality: Hand quality Q in [0, 1]
    
    Returns:
        Boolean: True if hand can be completed, False otherwise
    """
    return np.random.rand() < hand_quality
