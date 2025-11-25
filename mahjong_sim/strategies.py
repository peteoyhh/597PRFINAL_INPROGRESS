from typing import Union


def defensive_strategy(fan: Union[int, float], fan_min: int = 1) -> bool:
    """
    Defensive strategy: accepts any hand with fan >= fan_min.
    
    Note: fan_min should be >= 1 because Pi Hu (0 fan) is not a legal winning hand.
    
    Args:
        fan: Total fan count (including Kong bonuses)
        fan_min: Minimum fan threshold (default: 1)
    
    Returns:
        Boolean: True if strategy accepts this fan level
    """
    return fan >= fan_min


def aggressive_strategy(fan: Union[int, float], threshold: int = 3) -> bool:
    """
    Aggressive strategy: only accepts hands with fan >= threshold.
    
    Rejects low-fan hands to pursue higher-value combinations.
    
    Args:
        fan: Total fan count (including Kong bonuses)
        threshold: Minimum fan threshold (default: 3)
    
    Returns:
        Boolean: True if strategy accepts this fan level
    """
    return fan >= threshold
