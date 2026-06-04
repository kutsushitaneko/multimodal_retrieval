import json
from unittest.mock import MagicMock

from app.search_planner import SearchPlanner, SearchTask
from app.react_agentic_rag import ReactAgenticRAGPipeline, ReactToolRegistry
from tests.test_react_agentic_rag import FakeSearchService


def test_search_planner_parses_tasks_with_per_tool_queries():
    llm = MagicMock(
        return_value=json.dumps(
            {
                "is_likely_multihop": False,
                "current_hop_goal": "識別子と説明を取得",
                "dependent_aspects": [],
                "search_tasks": [
                    {
                        "tool": "caption_fulltext_search",
                        "query": "ORA-00923",
                        "exact_terms": ["ORA-00923"],
                        "reason": "識別子",
                    },
                    {
                        "tool": "caption_vector_search",
                        "query": "ORA-00923 とは何か",
                        "reason": "意味",
                    },
                ],
            },
            ensure_ascii=False,
        )
    )
    planner = SearchPlanner(llm)
    plan, error = planner.plan("ORA-00923 とは何ですか？")
    assert error is None
    assert plan is not None
    assert len(plan.search_tasks) == 2
    assert plan.search_tasks[0].tool == "caption_fulltext_search"
    assert plan.search_tasks[1].tool == "caption_vector_search"


def test_search_planner_skips_already_executed_tasks():
    llm = MagicMock(
        return_value=json.dumps(
            {
                "search_tasks": [
                    {"tool": "caption_vector_search", "query": "same query"},
                    {"tool": "caption_vector_search", "query": "new query"},
                ]
            },
            ensure_ascii=False,
        )
    )
    planner = SearchPlanner(llm)
    plan, error = planner.plan(
        "質問",
        executed_searches=[("caption_vector_search", "same query")],
    )
    assert error is None
    assert plan is not None
    assert len(plan.search_tasks) == 1
    assert plan.search_tasks[0].query == "new query"


def test_plan_and_execute_search_runs_different_queries_per_tool():
    search_service = FakeSearchService()
    pipeline = ReactAgenticRAGPipeline(search_service)
    pipeline.current_question = "ORA-00923 とは何ですか？"

    planner_response = json.dumps(
        {
            "search_tasks": [
                {"tool": "caption_fulltext_search", "query": "ORA-00923", "exact_terms": ["ORA-00923"]},
                {"tool": "caption_vector_search", "query": "ORA-00923 エラー 意味"},
            ]
        },
        ensure_ascii=False,
    )
    pipeline.search_planner = SearchPlanner(MagicMock(return_value=planner_response))

    from app.agentic_rag_common import EvidencePool

    registry = ReactToolRegistry(pipeline, None)
    pool = EvidencePool()
    selected = []
    list(
        registry.iter_execute(
            "plan_and_execute_search",
            {"phase": "initial"},
            pool,
            selected,
        )
    )

    fulltext_calls = [c for c in search_service.caption_calls if c[1] == "全文検索"]
    vector_queries = [c[0] for c in search_service.caption_calls if c[1] == "ベクトル検索"]
    assert fulltext_calls
    assert vector_queries
    assert fulltext_calls[0][0] == "{ORA-00923}"
    assert fulltext_calls[0][5] == "agentic_exact"
    assert fulltext_calls[0][0] != vector_queries[0]
