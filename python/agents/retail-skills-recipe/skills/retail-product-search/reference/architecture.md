# Product Search Architecture

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Multi-Agent Design](#multi-agent-design)
- [Success Criteria](#success-criteria)
- [Edge Cases](#edge-cases)
- [Constraints and Safety Rules](#constraints-and-safety-rules)

## Architecture Overview

```
User Query: "wireless headphones under $100"
    |
Layer 5: Search Agent (ADK)
    -> Vector Search (semantic product matching)
    -> RAG Agent (rich product details from Vertex AI Search)
    -> Clarification Agent (handles vague queries)
    -> Cart Agent (optional: add to cart actions)
    |
Layer 3: Embeddings (text-embedding-004 or multimodal-embedding-001)
    |
Layer 2: Product Data
    -> BigQuery (product metadata: id, name, price, category, etc.)
    -> GCS (product images)
    -> Vertex AI Search (rich content: descriptions, guides, FAQs)
```

## Multi-Agent Design

**Search Agent**: Queries Vector Search for semantic product matching via `retrieve_docs` tool.

**Clarification Agent**: Handles vague queries ("I need a gift") by asking 1-2 clarifying
questions about category, budget, and recipient before searching.

**RAG Agent** (optional): Retrieves rich product details (sizing guides, specs, FAQs) from Vertex AI Search.

**Orchestrator**: Routes queries to sub-agents:
- Vague query -> clarification agent
- Specific search -> search agent
- Detail question -> RAG agent
- Cart action -> cart tool

## Success Criteria

### Search Quality
- **Precision@5** >= 0.80
- **Recall@10** >= 0.70
- **NDCG@10** >= 0.75

### Performance
- **Search Latency P90** <= 300ms
- **End-to-End Response P90** <= 2s

### User Experience
- **Clarification Rate**: <20% of searches need clarification
- **Zero-Results Rate**: <5% of searches return no products

## Edge Cases

1. **No results** -- Broaden search, suggest similar products or popular items
2. **Ambiguous queries** -- Ask clarifying questions or show diverse options
3. **Out of stock** -- Show with badge, offer alternatives or notify-when-available
4. **Price range mismatch** -- Explain lowest available price, offer to adjust budget
5. **Multimodal search failures** -- Fall back to text description
6. **API failures** -- Fall back to keyword search or cached results
7. **Voice misunderstandings** -- Confirm transcription before searching

## Constraints and Safety Rules

1. **Price accuracy**: Always show current prices from data source; never fabricate
2. **Inventory honesty**: If stock unknown, state "availability to be confirmed"
3. **No upselling**: Don't recommend pricier alternatives unless explicitly asked
4. **Data privacy**: Don't log user queries without consent
5. **Image safety**: Filter inappropriate product images
6. **Factual descriptions**: Only use info from data sources; don't hallucinate features
7. **Budget respect**: Strictly honor user price constraints
