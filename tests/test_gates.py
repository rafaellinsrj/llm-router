from llm_router import Issue, Verdict, apply_gates


def test_clean_high_score_is_accepted():
    assert apply_gates(Verdict(92), "A clear text.").decision == "accept"


def test_low_score_is_rejected():
    result = apply_gates(Verdict(40), "text")
    assert result.decision == "reject" and result.flags == ("score",)


def test_refusal_vetoes_a_high_score():
    result = apply_gates(Verdict(98), "I'm sorry, but I can't help with that.")
    assert result.decision == "reject" and "refusal" in result.flags


def test_high_severity_issue_vetoes_a_high_score():
    verdict = Verdict(95, (Issue("fabricated_quote", "high"), Issue("style", "low")))
    result = apply_gates(verdict, "text")
    assert result.decision == "regenerate" and result.flags == ("fabricated_quote",)


def test_sensitive_without_primary_source_vetoes_a_perfect_score():
    result = apply_gates(Verdict(100), "text", facts={"sensitive": True})
    assert result.decision == "reject"
    assert apply_gates(Verdict(100), "text", facts={"sensitive": True, "primary_source": True}).decision == "accept"


def test_invalid_verdict_never_accepts():
    verdict = Verdict.from_json({"score": "high"})
    assert verdict.valid is False
    assert apply_gates(verdict, "text").decision == "regenerate"
