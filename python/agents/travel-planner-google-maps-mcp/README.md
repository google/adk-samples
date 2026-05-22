# 🗺️ Grounded Travel Agent 


Travel planning agent built with the **Google Agent Development Kit (ADK)** and grounded in physical reality using the **Google Maps Platform MCP server**.

Unlike standard LLMs that are "frozen in time" and prone to hallucinating travel times or closed businesses, this agent relies entirely on real-time spatial data to build logistically flawless itineraries.

## ✨ Key Features

* **Real-World Grounding:** Integrates `search_places`, `lookup_weather`, and `compute_routes` via the Model Context Protocol (MCP) to ensure every recommendation is factually accurate.
* **Modular Skill Architecture:** Uses the ADK's file-based `SKILL.md` system for progressive disclosure, optimizing token usage by only loading heavy travel instructions when the context requires it.


---

## 🚀 Getting Started

* [Google Maps MCP server](https://developers.google.com/maps/ai/grounding-lite)
* [Google Maps Platform Demo API Key](https://developers.google.com/maps/documentation/javascript/demo-key)
