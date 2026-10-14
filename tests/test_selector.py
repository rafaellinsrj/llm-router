from __future__ import annotations

import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from llm_router import (
    CandidateProfile,
    InMemoryStorage,
    Route,
    Sample,
    SelectionRequest,
    Selector,
    SelectorConfig,
    Stage,
)
from llm_router.selector import ModelStats, blocked_by_rejection, compute_rewards, demote, pick
from llm_router.storage import MetricRow

A, B, C = Route("openai", "gpt-large"), Route("anthropic", "claude-large"), Route("google", "gemini-flash")
CHAIN = [A, B, C]
PRICES = {A: 8.75, B: 9.0, C: 2.25}
NO_EXPLORE = SelectorConfig(explore_rate=0.0, cold_explore_rate=0.0)


def price(route):
    return PRICES[route]


def test_best_reward_wins_when_all_have_samples():
    stats = {A: ModelStats(0.70, 40), B: ModelStats(0.90, 40), C: ModelStats(0.60, 40)}
    sel = pick(CHAIN, stats, config=NO_EXPLORE, price=price)
    assert sel.chosen == B and sel.reason == "best reward"
    assert sel.chain == [B, A, C]


def test_cheapest_wins_within_five_points():
    stats = {A: ModelStats(0.90, 40), B: ModelStats(0.88, 40), C: ModelStats(0.86, 40)}
    sel = pick(CHAIN, stats, config=NO_EXPLORE, price=price)
    assert sel.chosen == C and sel.reason == "cheaper within margin"


def test_cheaper_model_outside_margin_loses():
    stats = {A: ModelStats(0.90, 40), B: ModelStats(0.50, 40), C: ModelStats(0.80, 40)}
    assert pick(CHAIN, stats, config=NO_EXPLORE, price=price).chosen == A


def test_provisional_clear_win_overrides_ready_model():
    stats = {A: ModelStats(0.64, 40), B: ModelStats(0.89, 4), C: ModelStats(0.50, 40)}
    sel = pick(CHAIN, stats, config=NO_EXPLORE, price=price)
    assert sel.chosen == B and sel.reason == "provisional clear win"


def test_no_data_keeps_configured_order():
    sel = pick(CHAIN, {}, config=NO_EXPLORE, price=price)
    assert sel.chosen == A and sel.reason == "default (no data)"


def test_exploration_rate_is_15_percent_when_warm():
    stats = {A: ModelStats(0.9, 40), B: ModelStats(0.5, 40), C: ModelStats(0.4, 40)}
    rng = random.Random(1234)
    picks = Counter(pick(CHAIN, stats, config=SelectorConfig(), price=price, rng=rng).explored
                    for _ in range(4000))
    assert abs(picks[True] / 4000 - 0.15) < 0.02


def test_exploration_rate_is_30_percent_with_cold_candidates_and_prefers_them():
    stats = {A: ModelStats(0.9, 40), B: ModelStats(0.5, 40), C: ModelStats(0.0, 2)}  # C is cold
    rng = random.Random(99)
    sels = [pick(CHAIN, stats, config=SelectorConfig(), price=price, rng=rng) for _ in range(4000)]
    explored = [s for s in sels if s.explored]
    assert abs(len(explored) / 4000 - 0.30) < 0.03
    assert {s.chosen for s in explored} == {C}


def test_seeded_selection_is_reproducible():
    stats = {A: ModelStats(0.9, 40), B: ModelStats(0.5, 2), C: ModelStats(0.4, 40)}
    first = [pick(CHAIN, stats, config=SelectorConfig(), price=price, rng=random.Random(7)).chosen
             for _ in range(5)]
    second = [pick(CHAIN, stats, config=SelectorConfig(), price=price, rng=random.Random(7)).chosen
              for _ in range(5)]
    assert first == second


def test_rewards_mix_quality_and_engagement_percentile():
    now = datetime(2026, 1, 10, tzinfo=UTC)
    old = now - timedelta(days=5)
    samples = [Sample(A, "pt", 1.0, old, engagement=10), Sample(B, "pt", 0.5, old, engagement=1000),
               Sample(C, "pt", 0.8, now)]  # C is immature: quality only
    rewards = compute_rewards(samples, SelectorConfig(), now)
    assert rewards[(A, "pt")]["reward"] == 0.4 * 1.0 + 0.6 * 0.25
    assert rewards[(B, "pt")]["reward"] == 0.4 * 0.5 + 0.6 * 0.75
    assert rewards[(C, "pt")]["reward"] == 0.8


# --------------------------------------------------------------- deterministic vetoes


def test_rejection_lock_demotes_a_high_scoring_model():
    config = SelectorConfig()
    blocked = blocked_by_rejection({A: (0.75, 40), B: (0.9, 10), C: (0.1, 100)}, config)
    assert blocked == {A}  # B has too few samples to be locked
    assert demote(CHAIN, blocked) == [B, C, A]


def test_gates_veto_a_high_score_in_the_selector():
    storage = InMemoryStorage()
    for route, reward in ((A, 0.99), (B, 0.70), (C, 0.60)):
        storage.add_metric(MetricRow("synthesis", "pt", route.provider, route.model, reward, reward, 50))
    profiles = [CandidateProfile(A, quality=Decimal("0.6")),  # best measured reward, low static quality
                CandidateProfile(B, quality=Decimal("0.9"), reliability=Decimal("0.9")),
                CandidateProfile(C, quality=Decimal("0.9"), reliability=Decimal("0.9"))]
    selector = Selector(storage, price, NO_EXPLORE, stage_tasks={Stage.SYNTHESIZE: "synthesis"},
                        profiles=profiles)

    normal = selector.choose(Stage.SYNTHESIZE, CHAIN, segment="pt")
    sensitive = selector.choose(Stage.SYNTHESIZE, CHAIN, segment="pt",
                                request=SelectionRequest(language="pt", sensitive=True))

    assert normal.chosen == A
    assert sensitive.chosen == B
    assert A not in sensitive.chain
    assert "openai/gpt-large:sensitive_floor" in sensitive.gated


def test_rejection_rate_from_metrics_locks_model_in_selector():
    storage = InMemoryStorage()
    storage.add_metric(MetricRow("synthesis", "pt", "openai", "gpt-large", 0.95, 0.95, 60,
                                 {"reject_rate": 0.8, "reject_total": 60}))
    storage.add_metric(MetricRow("synthesis", "pt", "anthropic", "claude-large", 0.80, 0.8, 60))
    selector = Selector(storage, price, NO_EXPLORE, stage_tasks={"synthesize": "synthesis"})
    sel = selector.choose("synthesize", CHAIN, segment="pt")
    assert sel.chosen == B and sel.chain[-1] == A


def test_selector_refresh_and_router_integration(router_factory):
    storage = InMemoryStorage()
    selector = Selector(storage, price, NO_EXPLORE, stage_tasks={"synthesize": "synthesis"})
    now = datetime.now(UTC)
    samples = [Sample(C, "pt", 0.9, now) for _ in range(20)] + [Sample(A, "pt", 0.7, now) for _ in range(20)]
    assert selector.refresh("synthesis", samples) == 2

    router = router_factory(storage=storage, selector=selector)
    result = router.call(Stage.SYNTHESIZE, "s", "u", segment="pt")

    assert result.route == C
    assert result.selection.reason == "best reward"
