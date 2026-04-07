"""
A/B Testing orchestrator.

Wraps the simulator for command-line usage from src.training module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.ab_simulator import ABTestSimulator


def run_ab_test(
    num_samples: int = 100,
    use_gpt4: bool = False,
    output_dir: str = "./outputs/ab_test",
    seed: int = 42,
) -> dict:
    """Run A/B test simulation."""
    simulator = ABTestSimulator(
        use_gpt4_judge=use_gpt4,
        seed=seed,
    )
    return simulator.run_ab_test(
        num_samples=num_samples,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    run_ab_test()
