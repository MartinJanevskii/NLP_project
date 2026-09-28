"""The experimental design: configurations, seeds, metrics and pre-registered contrasts."""

CONTEXTS = ("H1", "H2", "H3")
PROMPT_STRATEGIES = ("P1", "P2", "P3")
CONFIGS = [h + p for h in CONTEXTS for p in PROMPT_STRATEGIES]
SEEDS = [42, 123, 2026]
METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")


def _contrasts() -> dict[str, list[tuple[str, float]]]:
    """Weighted configuration sums; each contrast is a difference of means."""
    contrasts = {
        "RQ1_H2_minus_H1": [("H2" + p, 1 / 3) for p in PROMPT_STRATEGIES]
        + [("H1" + p, -1 / 3) for p in PROMPT_STRATEGIES],
    }
    for p in ("P2", "P3"):
        contrasts[f"RQ2_{p}_minus_P1_over_H1_H2"] = [
            (h + p, 0.5) for h in ("H1", "H2")
        ] + [(h + "P1", -0.5) for h in ("H1", "H2")]
        contrasts[f"RQ3_depth_effect_{p}_minus_P1"] = [
            ("H2" + p, 1),
            ("H1" + p, -1),
            ("H2P1", -1),
            ("H1P1", 1),
        ]
    for p in PROMPT_STRATEGIES:
        contrasts[f"RQ4_H3_minus_H2_{p}"] = [("H3" + p, 1), ("H2" + p, -1)]
    return contrasts


CONTRASTS = _contrasts()
