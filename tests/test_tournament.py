from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from llm_router import InMemoryStorage, Route, Stage, Tournament, TournamentConfig
from llm_router.providers import fake_providers
from llm_router.router import FORCED_ROUTE

GOOD, OK, BAD = Route("openai", "gpt-large"), Route("anthropic", "claude-small"), Route("google", "gemini-flash")
QUALITY = {"gpt-large": 0.9, "claude-small": 0.85, "gemini-flash": 0.4}


def build(task, item):
    return Stage.SYNTHESIZE, f"task={task}", {"input": item["text"]}, 512


def judge(task, item, data):
    return QUALITY[data["model"]]


def setup(router_factory, storage, **cfg):
    providers = fake_providers()
    for p in providers.values():
        p.default = None
        p.script = {m: (lambda model, s, u: {"model": model}) for m in QUALITY}
    router = router_factory(storage=storage, providers=providers)
    return router, Tournament(router, storage, judge, build, TournamentConfig(**cfg)), providers


def test_every_model_gets_the_same_inputs_without_fallback(router_factory):
    storage = InMemoryStorage()
    _router, tour, providers = setup(router_factory, storage)
    providers["google"].script["gemini-flash"] = [RuntimeError("down")]
    inputs = {"pt": [{"text": "a"}, {"text": "b"}]}

    rows = tour.run([GOOD, OK, BAD], ["summary"], inputs)

    by_route = {r.route: r for r in rows}
    assert by_route[GOOD].n == 2 and by_route[OK].n == 2
    assert by_route[BAD].n == 0 and by_route[BAD].failures == 2  # no silent fallback to another provider
    assert [c.user for c in providers["openai"].calls] == [c.user for c in providers["anthropic"].calls]
    assert by_route[GOOD].cost_per_piece_usd > 0
    assert FORCED_ROUTE.get() is None


def test_only_approved_models_compete(router_factory):
    storage = InMemoryStorage()
    _router, tour, _ = setup(router_factory, storage)
    tour.run([GOOD, OK, BAD], ["summary"], {"pt": [{"text": "a"}, {"text": "b"}]})

    assert tour.approved("summary", "pt") == [GOOD, OK]  # BAD is below floor and >15 pts behind
    assert tour.approved("summary", "es") is None       # no tournament for that segment


def test_stale_tournament_is_ignored(router_factory):
    storage = InMemoryStorage()
    _router, tour, _ = setup(router_factory, storage)
    tour.run([GOOD, OK], ["summary"], {"pt": [{"text": "a"}, {"text": "b"}]})
    tour.clock = lambda: datetime.now(UTC) + timedelta(days=30)
    assert tour.approved("summary", "pt") is None


def test_tournament_stops_at_cost_cap(router_factory):
    storage = InMemoryStorage()
    _router, tour, providers = setup(router_factory, storage, max_cost_usd=Decimal("0.0000001"))
    tour.run([GOOD, OK], ["summary"], {"pt": [{"text": "a"}, {"text": "b"}]})
    assert len(providers["openai"].calls) + len(providers["anthropic"].calls) == 1
