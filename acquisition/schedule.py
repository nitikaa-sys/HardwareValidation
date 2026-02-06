"""
Event scheduling and randomization for ADS1299 acquisition system.

Functions:
- setup_random_seed() - Set random seed for reproducibility
- generate_event_schedule() - Create randomized event list
- display_event_schedule() - Pretty-print schedule
"""

import random
from typing import List, Tuple, Optional

from .constants import VALID_LED_COLORS


def setup_random_seed(config: dict) -> None:
    """
    Set random seed for reproducible experiments.
    
    Args:
        config: Configuration dictionary containing optional experiment.random_seed
    """
    seed = config.get('experiment', {}).get('random_seed', None)
    
    if seed is not None:
        random.seed(seed)
        print(f"\n[REPRODUCIBILITY] Random seed set to: {seed}")
        print(f"                  Experiment will produce identical event order on re-runs")
    else:
        print(f"\n[REPRODUCIBILITY] No seed specified - using non-deterministic randomization")


def generate_event_schedule(
    num_events: int,
    condition_mapping: dict,
    enforce_equal_count: bool
) -> List[Tuple[str, str]]:
    """
    Generate randomized event schedule with optional balance enforcement.
    
    Args:
        num_events: Number of events to generate
        condition_mapping: Dict mapping LED colors to condition labels
        enforce_equal_count: If True, ensure equal counts of each condition
        
    Returns:
        List of (led_color, condition_label) tuples
        
    Raises:
        ValueError: If enforce_equal_count=True and num_events not divisible by 4
    """
    BASE_COLORS = VALID_LED_COLORS
    
    if enforce_equal_count:
        # Enforce equal count: equal counts, then shuffle
        if num_events % 4 != 0:
            raise ValueError(
                f"With enforce_equal_condition_count=true and 4 conditions, "
                f"num_events must be divisible by 4. Got {num_events}."
            )
        
        reps_per_color = num_events // 4
        color_list = []
        for color in BASE_COLORS:
            color_list.extend([color] * reps_per_color)
        
        # Shuffle once
        random.shuffle(color_list)
    
    else:
        # Non-enforced: random choice per event (already random, no shuffle)
        color_list = [random.choice(BASE_COLORS) for _ in range(num_events)]
    
    # Map colors to condition labels
    schedule = [(color, condition_mapping[color]) for color in color_list]
    
    return schedule


def display_event_schedule(
    schedule: List[Tuple[str, str]],
    enforce_equal_count: bool,
    seed: Optional[int] = None
) -> None:
    """
    Display the complete event schedule to the user for review.
    
    Args:
        schedule: List of (led_color, condition_label) tuples
        enforce_equal_count: Whether balance enforcement was used
        seed: Random seed used (None if not specified)
    """
    num_events = len(schedule)
    
    print(f"\n[EVENT SCHEDULE] {num_events} events generated")
    print(f"                 Equal condition count: {enforce_equal_count}")
    if seed is not None:
        print(f"                 Random seed: {seed}")
    
    # Show first 10 and last 5 events (or all if <= 20)
    if num_events <= 20:
        # Show all events
        for i, (color, label) in enumerate(schedule, 1):
            print(f"    Event {i:3d}: {color:6s} → {label}")
    else:
        # Show first 10
        for i, (color, label) in enumerate(schedule[:10], 1):
            print(f"    Event {i:3d}: {color:6s} → {label}")
        print(f"    ...")
        # Show last 5
        for i, (color, label) in enumerate(schedule[-5:], num_events - 4):
            print(f"    Event {i:3d}: {color:6s} → {label}")
    
    # Calculate and display balance statistics
    print(f"\n    Balance verification:")
    label_counts = {}
    for color, label in schedule:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / num_events) * 100
        print(f"        {label:15s}: {count:3d} events ({percentage:5.1f}%)")
