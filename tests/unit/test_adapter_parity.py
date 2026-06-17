"""Cross-backend adapter parity tests (design §7 Phase 6).

Identical logical trace data, encoded for each backend's API shape, must
normalize to semantically-equal `OpenInferenceTrace` objects through the
respective adapter's `get_trace`. "Semantically equal" means: the
`TraceLike` projection (which is what the classifier actually consumes) is
equal across both paths.

We also assert that with the same `Classification` list + fixed embedding
vectors, the clusterer produces identical `cluster_id` sets across runs
(`compute_cluster_id` is deterministic by construction; this guards against
upstream perturbation creeping in).
"""

import httpx

from agent_triage.adapters.trace.langfuse import LangfuseAdapter
from agent_triage.adapters.trace.langsmith import LangsmithAdapter
from agent_triage.adapters.trace.phoenix import PhoenixAdapter
from agent_triage.agent.subagents.clusterer import cluster_per_mode
from agent_triage.llm.embeddings import EmbeddingProvider
from agent_triage.models.classification import Classification
from agent_triage.rubric.spec import Clustering, Detection, Mode, Rubric, RubricMetadata

_TRACE_ID = "parity-trace-001"
_USER_MSG = "What is the capital of France?"
_ASSISTANT_MSG = "The capital of France is Paris."
_TOOL_NAME = "lookup_capital"
_TOOL_ARGS = '{"country": "France"}'
_TOOL_OUTPUT = "Paris"


def _phoenix_response() -> dict[str, object]:
    """Phoenix's GraphQL response shape for our canonical trace."""
    return {
        "data": {
            "spans": {
                "edges": [
                    {
                        "node": {
                            "context": {"traceId": _TRACE_ID, "spanId": "phoenix-llm"},
                            "parentId": None,
                            "name": "completion",
                            "startTime": "2026-05-22T00:00:00Z",
                            "endTime": "2026-05-22T00:00:01Z",
                            "attributes": {
                                "openinference.span.kind": "LLM",
                                "llm.model_name": "test-model",
                                "llm.input_messages.0.message.role": "user",
                                "llm.input_messages.0.message.content": _USER_MSG,
                                "llm.output_messages.0.message.role": "assistant",
                                "llm.output_messages.0.message.content": _ASSISTANT_MSG,
                                "llm.token_count.total": 25,
                            },
                            "statusCode": "OK",
                            "statusMessage": None,
                            "events": [],
                        }
                    },
                    {
                        "node": {
                            "context": {"traceId": _TRACE_ID, "spanId": "phoenix-tool"},
                            "parentId": None,
                            "name": _TOOL_NAME,
                            "startTime": "2026-05-22T00:00:01Z",
                            "endTime": "2026-05-22T00:00:02Z",
                            "attributes": {
                                "openinference.span.kind": "TOOL",
                                "tool.name": _TOOL_NAME,
                                "tool.parameters": _TOOL_ARGS,
                                "output.value": _TOOL_OUTPUT,
                            },
                            "statusCode": "OK",
                            "statusMessage": None,
                            "events": [],
                        }
                    },
                ]
            }
        }
    }


def _langfuse_response() -> dict[str, object]:
    """Langfuse's REST response shape carrying the same logical trace."""
    return {
        "id": _TRACE_ID,
        "observations": [
            {
                "id": "langfuse-llm",
                "type": "GENERATION",
                "name": "completion",
                "startTime": "2026-05-22T00:00:00Z",
                "endTime": "2026-05-22T00:00:01Z",
                "model": "test-model",
                "input": [{"role": "user", "content": _USER_MSG}],
                "output": {"role": "assistant", "content": _ASSISTANT_MSG},
                "usage": {"total": 25},
            },
            {
                "id": "langfuse-tool",
                "type": "TOOL",
                "name": _TOOL_NAME,
                "startTime": "2026-05-22T00:00:01Z",
                "endTime": "2026-05-22T00:00:02Z",
                "input": {"country": "France"},
                "output": _TOOL_OUTPUT,
            },
        ],
    }


def _phoenix_handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_phoenix_response())


def _langfuse_handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_langfuse_response())


def _langsmith_response() -> dict[str, object]:
    """LangSmith's REST response shape carrying the same logical trace.

    Root run is the LLM call; the tool call lives as a child_run.
    """
    return {
        "id": _TRACE_ID,
        "name": "completion",
        "run_type": "llm",
        "start_time": "2026-05-22T00:00:00Z",
        "end_time": "2026-05-22T00:00:01Z",
        "extra": {"invocation_params": {"model": "test-model"}},
        "inputs": {
            "messages": [{"role": "user", "content": _USER_MSG}],
        },
        "outputs": {
            "generations": [[{"message": {"role": "assistant", "content": _ASSISTANT_MSG}}]],
            "llm_output": {"token_usage": {"total_tokens": 25}},
        },
        "child_runs": [
            {
                "id": "langsmith-tool",
                "name": _TOOL_NAME,
                "run_type": "tool",
                "start_time": "2026-05-22T00:00:01Z",
                "end_time": "2026-05-22T00:00:02Z",
                "inputs": {"country": "France"},
                "outputs": _TOOL_OUTPUT,
            }
        ],
    }


def _langsmith_handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_langsmith_response())


async def test_all_three_adapters_decode_to_equal_trace_like_view() -> None:
    """The §7 'lossless normalization' bar: same logical trace -> same
    TraceLike out of every adapter."""
    phoenix = PhoenixAdapter(
        base_url="http://phoenix.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_phoenix_handler),
            base_url="http://phoenix.test",
        ),
    )
    langfuse = LangfuseAdapter(
        host="http://langfuse.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_langfuse_handler),
            base_url="http://langfuse.test",
        ),
    )
    langsmith = LangsmithAdapter(
        endpoint="http://langsmith.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_langsmith_handler),
            base_url="http://langsmith.test",
        ),
    )

    p_view = (await phoenix.get_trace(_TRACE_ID)).to_trace_like()
    l_view = (await langfuse.get_trace(_TRACE_ID)).to_trace_like()
    s_view = (await langsmith.get_trace(_TRACE_ID)).to_trace_like()

    # The three views agree on the classifier-visible fields.
    assert p_view.trace_id == l_view.trace_id == s_view.trace_id == _TRACE_ID
    assert p_view.final_response == l_view.final_response == s_view.final_response == _ASSISTANT_MSG
    # The same tool call surfaces from all three adapters.
    assert len(p_view.tool_calls) == len(l_view.tool_calls) == len(s_view.tool_calls) == 1
    assert (
        p_view.tool_calls[0].name
        == l_view.tool_calls[0].name
        == s_view.tool_calls[0].name
        == _TOOL_NAME
    )
    assert p_view.tool_calls[0].arguments == l_view.tool_calls[0].arguments
    assert l_view.tool_calls[0].arguments == s_view.tool_calls[0].arguments
    # full_text contains the same key user + assistant + tool content from all three.
    for fragment in (_USER_MSG, _ASSISTANT_MSG, _TOOL_NAME, _TOOL_OUTPUT):
        assert fragment in p_view.full_text, f"phoenix view missing {fragment!r}"
        assert fragment in l_view.full_text, f"langfuse view missing {fragment!r}"
        assert fragment in s_view.full_text, f"langsmith view missing {fragment!r}"
    # Token total comes through all three paths and equals the metric.
    assert (
        p_view.metrics["total_tokens"]
        == l_view.metrics["total_tokens"]
        == s_view.metrics["total_tokens"]
        == 25.0
    )

    await phoenix.close()
    await langfuse.close()
    await langsmith.close()


async def test_get_trace_preserves_span_kinds_across_all_three_adapters() -> None:
    """Per-kind span counts match: one LLM + one TOOL through every adapter."""
    phoenix = PhoenixAdapter(
        base_url="http://phoenix.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_phoenix_handler),
            base_url="http://phoenix.test",
        ),
    )
    langfuse = LangfuseAdapter(
        host="http://langfuse.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_langfuse_handler),
            base_url="http://langfuse.test",
        ),
    )
    langsmith = LangsmithAdapter(
        endpoint="http://langsmith.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_langsmith_handler),
            base_url="http://langsmith.test",
        ),
    )

    p = await phoenix.get_trace(_TRACE_ID)
    lf = await langfuse.get_trace(_TRACE_ID)
    ls = await langsmith.get_trace(_TRACE_ID)

    assert len(p.get_llm_spans()) == len(lf.get_llm_spans()) == len(ls.get_llm_spans()) == 1
    assert (
        len(p.get_tool_call_spans())
        == len(lf.get_tool_call_spans())
        == len(ls.get_tool_call_spans())
        == 1
    )

    await phoenix.close()
    await langfuse.close()
    await langsmith.close()


class _FixedEmbeddingProvider(EmbeddingProvider):
    """Returns a fixed vector per text. Used to verify clusterer determinism."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.model = "fixed"
        self._by_text = vectors_by_text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._by_text.get(t, [0.0, 0.0, 1.0]) for t in texts]


async def test_clusterer_cluster_ids_stable_under_fixed_embeddings() -> None:
    """Same classifications + same embedding vectors -> same cluster_id set."""
    rubric = Rubric(
        apiVersion="agent-triage.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="parity", version="0.1.0"),
        modes=[
            Mode(id="m1", severity="medium", detection=Detection(type="regex", pattern="x")),
        ],
        clustering=Clustering(
            embedding_model="fixed",
            similarity_threshold=0.82,
            min_cluster_size=3,
        ),
    )
    classifications = [
        Classification(
            trace_id=f"t-{i}",
            rubric_version="parity@0.1.0",
            mode_id="m1",
            positive=True,
            extra={"excerpt": f"excerpt-{i}"},
        )
        for i in range(4)
    ]
    same_vec = [1.0, 0.0]
    vectors = {f"excerpt-{i}": same_vec for i in range(4)}

    clusters_a = await cluster_per_mode(
        classifications, rubric=rubric, embedding_provider=_FixedEmbeddingProvider(vectors)
    )
    clusters_b = await cluster_per_mode(
        classifications, rubric=rubric, embedding_provider=_FixedEmbeddingProvider(vectors)
    )

    ids_a = {c.cluster_id for c in clusters_a}
    ids_b = {c.cluster_id for c in clusters_b}
    assert ids_a == ids_b
    assert len(ids_a) == 1  # all four classifications cluster together


async def test_absent_status_normalizes_to_unset_across_all_three_adapters() -> None:
    """M-9 status parity: when a backend carries no explicit status signal,
    every adapter normalizes to UNSET (no hardcoded OK), so traces without
    explicit status produce identical classifier inputs across backends."""

    def phoenix_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [
                            {
                                "node": {
                                    "context": {"traceId": _TRACE_ID, "spanId": "p-1"},
                                    "parentId": None,
                                    "name": "step",
                                    "startTime": "2026-05-22T00:00:00Z",
                                    "endTime": "2026-05-22T00:00:01Z",
                                    "attributes": {},
                                    # No statusCode field at all.
                                }
                            }
                        ]
                    }
                }
            },
        )

    def langfuse_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": _TRACE_ID,
                "observations": [
                    {
                        "id": "lf-1",
                        "type": "SPAN",
                        "name": "step",
                        "startTime": "2026-05-22T00:00:00Z",
                        "endTime": "2026-05-22T00:00:01Z",
                        # No level / statusMessage.
                    }
                ],
            },
        )

    def langsmith_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/query":
            return httpx.Response(200, json={"runs": []})
        return httpx.Response(
            200,
            json={
                "id": _TRACE_ID,
                "name": "step",
                "run_type": "chain",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:01Z",
                # No error / status fields.
                "child_runs": [],
            },
        )

    phoenix = PhoenixAdapter(
        base_url="http://phoenix.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(phoenix_handler),
            base_url="http://phoenix.test",
        ),
    )
    langfuse = LangfuseAdapter(
        host="http://langfuse.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(langfuse_handler),
            base_url="http://langfuse.test",
        ),
    )
    langsmith = LangsmithAdapter(
        endpoint="http://langsmith.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(langsmith_handler),
            base_url="http://langsmith.test",
        ),
    )

    p = await phoenix.get_trace(_TRACE_ID)
    lf = await langfuse.get_trace(_TRACE_ID)
    ls = await langsmith.get_trace(_TRACE_ID)

    assert p.spans[0].status.code == "UNSET"
    assert lf.spans[0].status.code == "UNSET"
    assert ls.spans[0].status.code == "UNSET"
    # And the derived error_count metric agrees across all three.
    assert (
        p.to_trace_like().metrics["error_count"]
        == lf.to_trace_like().metrics["error_count"]
        == ls.to_trace_like().metrics["error_count"]
        == 0.0
    )

    await phoenix.close()
    await langfuse.close()
    await langsmith.close()
