"""
QoS-LSMA Demo / Example Entry Point
=====================================
Demonstrates the full retrieve → compose → store online loop with:
  • Mock service catalog (Tourism domain)
  • Mock embedding function (avoids API cost during testing)
  • Mock service registry (Python callables instead of HTTP calls)

To run with a real LLM (DeepSeek / OpenAI), replace the API keys in
QoSLSMAConfig and set ``use_mock_embed=False``.

Usage:
    python main.py [--real-llm] [--api-key YOUR_KEY] [--base-url URL]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("qos_lsma.demo")


# -----------------------------------------------------------------------
# Mock embedding function  (random unit vectors, reproducible by seed)
# -----------------------------------------------------------------------
def make_mock_embed_fn(dim: int = 64, seed: int = 42):
    """Return a mock embedding function that maps text to random unit vectors.

    Uses a deterministic hash so the same text always gets the same vector.
    Only suitable for testing without an API key.
    """
    rng = random.Random()

    def embed(text: str) -> List[float]:
        rng.seed(hash(text) % (2 ** 32) + seed)
        vec = [rng.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec)) + 1e-10
        return [x / norm for x in vec]

    return embed


# -----------------------------------------------------------------------
# Mock service catalog  (Tourism domain, 10 representative services)
# -----------------------------------------------------------------------
MOCK_SERVICE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "FlightSearchAPI",
        "description": "Search available flights between two cities on a date.",
        "signature": "search_flights(origin: str, destination: str, date: str) -> List[Flight]",
    },
    {
        "name": "FlightBookingAPI",
        "description": "Book a specific flight and return confirmation.",
        "signature": "book_flight(flight_id: str, passenger_name: str) -> Booking",
    },
    {
        "name": "HotelSearchAPI",
        "description": "Search hotels in a city for given check-in/check-out dates.",
        "signature": "search_hotels(city: str, check_in: str, check_out: str) -> List[Hotel]",
    },
    {
        "name": "HotelBookingAPI",
        "description": "Reserve a hotel room and return booking confirmation.",
        "signature": "book_hotel(hotel_id: str, guest_name: str, nights: int) -> Booking",
    },
    {
        "name": "WeatherAPI",
        "description": "Get weather forecast for a city on a specific date.",
        "signature": "get_weather(city: str, date: str) -> WeatherReport",
    },
    {
        "name": "CurrencyConverterAPI",
        "description": "Convert an amount from one currency to another.",
        "signature": "convert_currency(amount: float, from_currency: str, to_currency: str) -> float",
    },
    {
        "name": "TranslationAPI",
        "description": "Translate text to a target language.",
        "signature": "translate(text: str, target_language: str) -> str",
    },
    {
        "name": "MapDirectionsAPI",
        "description": "Get travel directions between two locations.",
        "signature": "get_directions(origin: str, destination: str, mode: str) -> Directions",
    },
    {
        "name": "VisaCheckAPI",
        "description": "Check visa requirements for a passport holder travelling to a country.",
        "signature": "check_visa(passport_country: str, destination_country: str) -> VisaInfo",
    },
    {
        "name": "TravelInsuranceAPI",
        "description": "Quote travel insurance for a trip.",
        "signature": "get_insurance_quote(destination: str, duration_days: int) -> Quote",
    },
]


# -----------------------------------------------------------------------
# Mock service registry (Python callables for testing)
# -----------------------------------------------------------------------
def _mock_flight_search(**kwargs):
    return [
        {"flight_id": "CA123", "price": 320.0, "duration": "3h"},
        {"flight_id": "MU456", "price": 280.0, "duration": "3.5h"},
    ]


def _mock_hotel_search(**kwargs):
    return [
        {"hotel_id": "H001", "name": "Grand Hotel", "price_per_night": 120.0},
        {"hotel_id": "H002", "name": "City Inn", "price_per_night": 85.0},
    ]


def _mock_weather(**kwargs):
    return {"city": kwargs.get("city", "?"), "forecast": "Sunny, 22°C"}


def _mock_currency(**kwargs):
    return kwargs.get("amount", 0) * 7.1  # USD → CNY approx.


MOCK_SERVICE_REGISTRY = {
    "FlightSearchAPI": _mock_flight_search,
    "HotelSearchAPI": _mock_hotel_search,
    "WeatherAPI": _mock_weather,
    "CurrencyConverterAPI": _mock_currency,
}


# -----------------------------------------------------------------------
# Demo queries
# -----------------------------------------------------------------------
DEMO_QUERIES = [
    {
        "query": "I want to book a round-trip flight from Beijing to Paris "
                 "and a hotel near the Eiffel Tower for 5 nights next month. "
                 "Also check if I need a visa and what the weather will be like.",
        "context": {"region": "CN", "domain": "Tourism", "time_bucket": "2026-Q1"},
    },
    {
        "query": "Find the cheapest flight from Shanghai to Tokyo on March 20, "
                 "and book a hotel for 3 nights. Convert the total cost to CNY.",
        "context": {"region": "CN", "domain": "Tourism", "time_bucket": "2026-Q1"},
    },
    {
        "query": "Plan a 7-day trip to Rome. I need flights, hotels, "
                 "travel insurance, and local transportation directions.",
        "context": {"region": "EU", "domain": "Tourism", "time_bucket": "2026-Q2"},
    },
]


# -----------------------------------------------------------------------
# Main demo
# -----------------------------------------------------------------------
def run_demo(use_real_llm: bool, api_key: str, base_url: str) -> None:
    from qos_lsma import QoSLSMA, QoSLSMAConfig

    logger.info("Initialising QoS-LSMA …")

    embed_dim = 64  # use 1536 with real embeddings
    cfg = QoSLSMAConfig(
        llm_model="deepseek-chat",
        llm_api_key=api_key,
        llm_base_url=base_url or "https://api.deepseek.com",
        embedding_dim=embed_dim,
        embedding_api_key=api_key,
        top_k_candidates=5,
        hop_size=2,
        dqn_hidden_dims=[128, 64],  # smaller for demo
        dqn_batch_size=8,
        replay_buffer_size=500,
        max_stm_size=20,
        max_ltm_nodes=200,
    )

    # Use mock embedding unless caller asked for real LLM
    embed_fn = None if use_real_llm else make_mock_embed_fn(dim=embed_dim)

    system = QoSLSMA(
        config=cfg,
        service_catalog=MOCK_SERVICE_CATALOG,
        service_registry=MOCK_SERVICE_REGISTRY,
        embed_fn=embed_fn,
    )

    logger.info("System ready: %s", system)

    for i, item in enumerate(DEMO_QUERIES):
        print(f"\n{'='*70}")
        print(f"  Interaction #{i+1}")
        print(f"  Query: {item['query'][:80]}…")
        print(f"{'='*70}")

        user_score = None if i == 0 else 4.0  # simulate user feedback

        if use_real_llm:
            result = system.run(
                query=item["query"],
                context=item["context"],
                user_score=user_score,
            )
        else:
            # Without a real LLM, demonstrate module-level logic
            result = _run_mock_interaction(system, item, i, user_score)

        _print_result(result)

    print(f"\n{'='*70}")
    print(f"  Final system state: {system}")
    print(f"  LTM: {system.ltm.num_nodes} nodes, {system.ltm.num_edges} edges")
    print(f"  Embedding index size: {system.index.size}")
    print(f"  DQN ε = {system.dqn.epsilon:.4f}")
    print(f"{'='*70}\n")


# -----------------------------------------------------------------------
# Mock interaction (no real LLM calls)
# -----------------------------------------------------------------------
def _run_mock_interaction(
    system,
    item: Dict,
    interaction_idx: int,
    user_score,
) -> Dict[str, Any]:
    """Simulate a full interaction using mock data (no LLM API needed)."""
    from qos_lsma.memory.memory_item import MemoryItem

    query = item["query"]
    context = item["context"]

    # Step 1-5: Memory retrieval (real retrieval module)
    memory_context = system.retrieval.retrieve(query=query, context=context)

    # Step 6: Mock workflow
    workflow = [
        {"step": 1, "subtask": "check_visa", "description": "Check visa requirements",
         "dependencies": [], "expected_output": "visa_info"},
        {"step": 2, "subtask": "search_flights", "description": "Find available flights",
         "dependencies": [], "expected_output": "flight_list"},
        {"step": 3, "subtask": "book_flight", "description": "Book the cheapest flight",
         "dependencies": [2], "expected_output": "booking_confirmation"},
        {"step": 4, "subtask": "search_hotels", "description": "Find hotels near destination",
         "dependencies": [], "expected_output": "hotel_list"},
        {"step": 5, "subtask": "check_weather", "description": "Get weather forecast",
         "dependencies": [], "expected_output": "weather_report"},
    ]

    # Step 7: Mock groundings
    groundings = [
        {"step": 1, "subtask": "check_visa",
         "services": [{"name": "VisaCheckAPI", "arguments": {"passport_country": "CN",
                                                               "destination_country": "FR"}}]},
        {"step": 2, "subtask": "search_flights",
         "services": [{"name": "FlightSearchAPI",
                       "arguments": {"origin": "BJS", "destination": "CDG",
                                     "date": "2026-04-15"}}]},
        {"step": 3, "subtask": "book_flight",
         "services": [{"name": "FlightBookingAPI",
                       "arguments": {"flight_id": "CA123", "passenger_name": "Test User"}}]},
        {"step": 4, "subtask": "search_hotels",
         "services": [{"name": "HotelSearchAPI",
                       "arguments": {"city": "Paris", "check_in": "2026-04-15",
                                     "check_out": "2026-04-20"}}]},
        {"step": 5, "subtask": "check_weather",
         "services": [{"name": "WeatherAPI",
                       "arguments": {"city": "Paris", "date": "2026-04-15"}}]},
    ]

    # Step 8: Execute via real executor + mock registry
    raw_results = system.executor.run(groundings)

    # Step 9: Create mock memory items directly (no LLM)
    mock_items = [
        MemoryItem(
            category="workflow_trace",
            content=f"Booking trip to Paris: check visa → search flights → book hotel",
            entities=[
                {"name": "BookTrip", "type": "Intent"},
                {"name": "FlightSearchAPI", "type": "Service"},
                {"name": "HotelSearchAPI", "type": "Service"},
            ],
            relations=[
                {"head": "BookTrip", "label": "uses", "tail": "FlightSearchAPI"},
                {"head": "BookTrip", "label": "uses", "tail": "HotelSearchAPI"},
            ],
            metadata={"domain": "Tourism", "success": True},
        ),
        MemoryItem(
            category="service_usage",
            content="FlightSearchAPI: BJS→CDG, returns [CA123 $320, MU456 $280]",
            entities=[
                {"name": "FlightSearchAPI", "type": "Service"},
            ],
            relations=[],
            metadata={"domain": "Tourism", "success": True},
        ),
        MemoryItem(
            category="profile_fact",
            content="User prefers cheaper options; travelling from Beijing region",
            entities=[
                {"name": "PreferCheap", "type": "UserProfile"},
            ],
            relations=[],
            metadata={"domain": "Tourism", "success": True},
        ),
    ]

    # Embed items
    for m_item in mock_items:
        m_item.embedding = system.embed_fn(m_item.content)

    # Step 10: DQN decide
    query_emb = system.embed_fn(query)

    if user_score is not None and system._last_retrieved_node_ids:
        system.dqn.receive_delayed_reward(
            retrieved_item_ids=system._last_retrieved_node_ids,
            user_score=user_score,
        )

    system._last_retrieved_node_ids = [
        nid for nid, _ in system.index.top_k(query_emb, k=5)
    ]

    to_store, to_discard = system.dqn.decide_batch(
        stm_items=mock_items,
        query_embedding=query_emb,
        score=4.0,
        n_stm=len(mock_items),
        n_ltm=system.ltm.num_nodes,
    )

    # Step 11: Commit to LTM
    for m_item in to_store:
        node_id = system.ltm.commit_item(m_item)
        if m_item.embedding:
            system.index.add(node_id, m_item.embedding)

    # Step 12: DQN update
    system._interaction_count += 1
    dqn_loss = system.dqn.update()
    system.dqn.decay_epsilon()

    return {
        "workflow": workflow,
        "groundings": groundings,
        "execution_results": raw_results,
        "memory_context": memory_context,
        "stored_items": [m.to_dict() for m in to_store],
        "discarded_items_count": len(to_discard),
        "dqn_loss": dqn_loss,
        "interaction_id": system._interaction_count,
    }


def _print_result(result: Dict[str, Any]) -> None:
    print(f"\n[Workflow]  {len(result['workflow'])} steps planned")
    for step in result["workflow"]:
        print(f"  Step {step.get('step')}: {step.get('subtask')} – {step.get('description','')}")

    print(f"\n[Execution]  {len(result['execution_results'])} service calls")
    for r in result["execution_results"]:
        status = "✓" if r.get("success") else "✗"
        print(
            f"  Step {r.get('step')} [{status}] "
            f"{r.get('service')}  {r.get('latency_ms', 0):.1f}ms"
        )
        if r.get("response") is not None:
            resp_str = str(r["response"])[:80]
            print(f"    → {resp_str}")

    print(f"\n[Memory Update]")
    print(f"  Stored to LTM : {len(result['stored_items'])} items")
    print(f"  DQN loss      : {result.get('dqn_loss')}")

    print(f"\n[Retrieved Memory Context (first 300 chars)]")
    print(f"  {result['memory_context'][:300]}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="QoS-LSMA Demo")
    parser.add_argument(
        "--real-llm", action="store_true",
        help="Use real LLM API (requires --api-key)."
    )
    parser.add_argument("--api-key", default="", help="LLM / Embedding API key.")
    parser.add_argument(
        "--base-url", default="https://api.deepseek.com",
        help="LLM base URL (default: DeepSeek)."
    )
    args = parser.parse_args()

    if args.real_llm and not args.api_key:
        print("ERROR: --real-llm requires --api-key.", file=sys.stderr)
        sys.exit(1)

    run_demo(
        use_real_llm=args.real_llm,
        api_key=args.api_key,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
