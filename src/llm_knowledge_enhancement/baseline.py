"""Reading the result of the unmodified upstream CoLaKG MovieLens training."""

import math
import re


def completed_metrics(log: str) -> dict[str, float]:
    """Metrics of the final scheduled evaluation; requires all 2,000 epochs in the log."""
    if not re.search(r"^EPOCH\[2000/2000\]", log, re.MULTILINE):
        raise ValueError("Original 2,000-epoch training has not completed")
    match = re.search(r"TEST RESULTS at EPOCH\[1996/2000\]: (.*)", log)
    if not match:
        raise ValueError("Final scheduled evaluation is missing")
    result = {}
    for name in ("recall", "ndcg"):
        values = re.search(r"'" + name + r"': array\(\[([^]]+)\]\)", match[1])
        if values is None:
            raise ValueError(f"Missing {name}")
        numbers = [float(value) for value in values[1].split(",")]
        if len(numbers) != 2 or not all(
            math.isfinite(x) and 0 <= x <= 1 for x in numbers
        ):
            raise ValueError(f"Invalid {name}")
        result.update(
            {f"{name}@{k}": value for k, value in zip((10, 20), numbers, strict=True)}
        )
    return result
