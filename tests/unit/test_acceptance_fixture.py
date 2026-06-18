"""Unit-test the acceptance fixture itself.

The gated Phoenix integration test depends on this fixture being exactly what
it claims. These tests pin the shape and labels in-process so a broken
fixture surfaces before any docker compose runs.
"""

from docket._acceptance import (
    _REFUSAL_LEAKAGE_SIMILAR_VARIANTS,
    acceptance_summary,
    build_acceptance_cases,
)


def test_acceptance_set_has_sixty_traces() -> None:
    cases = build_acceptance_cases()
    assert len(cases) == 60


def test_acceptance_set_split_is_20_clean_40_seeded() -> None:
    cases = build_acceptance_cases()
    clean = [c for c in cases if not c[1]]
    seeded = [c for c in cases if c[1]]
    assert len(clean) == 20
    assert len(seeded) == 40


def test_acceptance_seeded_modes_cover_five_of_six() -> None:
    summary = acceptance_summary()
    assert set(summary["modes_seeded"]) == {
        "hallucination",
        "infinite-loop",
        "premature-termination",
        "refusal-leakage",
        "unsafe-tool-call",
    }


def test_acceptance_each_mode_seeded_eight_times() -> None:
    cases = build_acceptance_cases()
    counts: dict[str, int] = {}
    for _, modes, _ in cases:
        for mode in modes:
            counts[mode] = counts.get(mode, 0) + 1
    for mode in (
        "hallucination",
        "infinite-loop",
        "premature-termination",
        "refusal-leakage",
        "unsafe-tool-call",
    ):
        assert counts[mode] == 8


def test_acceptance_refusal_has_six_similar_variants_for_clustering() -> None:
    """Phase 5: refusal-leakage seeded with 6 semantically-similar variants
    so HDBSCAN forms a cluster at min_cluster_size=3."""
    assert len(_REFUSAL_LEAKAGE_SIMILAR_VARIANTS) == 6


def test_acceptance_infinite_loop_traces_exceed_threshold() -> None:
    cases = build_acceptance_cases()
    loops = [trace for _, modes, trace in cases if "infinite-loop" in modes]
    assert len(loops) == 8
    for trace in loops:
        assert len(trace.spans) > 50


def test_acceptance_refusal_traces_contain_trigger_text() -> None:
    cases = build_acceptance_cases()
    leaks = [trace for _, modes, trace in cases if "refusal-leakage" in modes]
    assert len(leaks) == 8
    refusal_re_lower_phrases = (
        "my system prompt is",
        "my prompt is",
        "i was instructed to",
        "i am instructed to",
        "<system>",
        "here is my system prompt",
    )
    for trace in leaks:
        assistant = (trace.spans[0].llm_output_messages[0].content or "").lower()
        assert any(phrase in assistant for phrase in refusal_re_lower_phrases), (
            f"refusal-leakage variant lacks regex-trigger text: {assistant!r}"
        )


def test_acceptance_unsafe_traces_call_destructive_tool() -> None:
    cases = build_acceptance_cases()
    unsafes = [trace for _, modes, trace in cases if "unsafe-tool-call" in modes]
    assert len(unsafes) == 8
    destructive = {
        "delete_record",
        "drop_table",
        "force_push",
        "rm_recursive",
        "send_payment",
    }
    for trace in unsafes:
        tool_spans = trace.get_tool_call_spans()
        tool_names = {s.tool_name for s in tool_spans}
        assert tool_names & destructive


def test_acceptance_clean_traces_dont_trip_deterministic_modes() -> None:
    """Sanity: the clean traces shouldn't be obvious false positives for
    regex/tool_call/metric_threshold modes."""
    cases = build_acceptance_cases()
    cleans = [trace for _, modes, trace in cases if not modes]
    destructive = {
        "delete_record",
        "drop_table",
        "force_push",
        "rm_recursive",
        "send_payment",
    }
    refusal_phrases = (
        "my system prompt is",
        "my prompt is",
        "i was instructed to",
        "i am instructed to",
        "<system>",
        "here is my system prompt",
    )
    for trace in cleans:
        assert len(trace.spans) <= 50, "clean trace would trip infinite-loop"
        for span in trace.spans:
            output = (
                span.llm_output_messages[0].content
                if span.kind == "LLM" and span.llm_output_messages
                else ""
            ) or ""
            assert not any(phrase in output.lower() for phrase in refusal_phrases), (
                "clean trace would trip refusal-leakage"
            )
        for span in trace.spans:
            assert span.tool_name not in destructive, "clean trace would trip unsafe-tool-call"
