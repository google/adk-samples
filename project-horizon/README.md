# Project Horizon

## Exploring the Future of Interoperable, Multimodal AI Agent Systems

---
![Project Horizon Banner](assets/project-horizon.png)

---

## Overview

**Project Horizon explores the future of collaborative, multimodal AI agent systems.** It investigates how sophisticated, responsive, and interoperable agents can be built by integrating cutting-edge agent development frameworks and open communication protocols. This proof-of-concept demonstrates the powerful synergy between **Google's Agent Development Kit (ADK)** (leveraging the **Gemini Live API** for real-time interaction), the **Agent2Agent (A2A) protocol**, and the **Model Context Protocol (MCP)**.

This project serves as a reference implementation and a testbed for:

*   Integrating diverse agent frameworks and tools.
*   Implementing real-time, interactive user experiences (voice, video, text) using ADK's `run_live` powered by the Gemini Live API.
*   Establishing standardized communication patterns between independent agents (A2A).
*   Enabling agents to securely interact with external tools and data sources (MCP).
*   Exploring advanced agent architectures, design patterns, and evaluation techniques.

**Project Horizon aims to capture and demonstrate the future of agent interactions, providing technical implementations, architectural blueprints, and practical design patterns for building the next generation of AI applications.**

## Potential & Vision

Project Horizon is more than just a demonstration of a single capability; it's a foundation for exploring the future of collaborative AI:

*   **Interoperability Blueprint:** Provides a concrete example of how systems built with different frameworks (ADK, potentially LangChain/CrewAI via A2A/MCP wrappers) can communicate using open standards.
*   **Modular Agent Design:** Demonstrates the benefits of breaking down complex tasks into specialized agents communicating via A2A.
*   **Secure Tool Integration:** Showcases MCP as a standardized, potentially more secure way for agents to access external capabilities compared to direct API calls or embedding tool logic within the agent itself.
*   **Multimodal Interaction:** Leverages ADK `run_live` for cutting-edge, real-time voice and potentially video interactions.
*   **Design Pattern Exploration:** Serves as a platform to implement and compare different agent architectures (e.g., hierarchical delegation, parallel execution via A2A, sequential workflows combining A2A and MCP calls).
*   **Foundation for Asynchronicity:** While starting synchronous, the architecture is designed to evolve towards handling asynchronous tasks, long-running operations, and push notifications using A2A/MCP capabilities.

**Ultimately, Project Horizon aims to be a valuable resource for developers seeking to build robust, scalable, and interoperable multi-agent systems.**

## Core Technologies Showcased

*   **[Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/):** Utilized for the primary user-facing "Host Agent", providing the `run_live` framework for agent orchestration and multimodal communication management.
*   **[Gemini Live API](https://ai.google.dev/gemini-api/docs/live):** Enables the low-latency, bidirectional, streaming voice (and potentially video) interactions with the Gemini model, powering the real-time user experience within ADK's `run_live`.
*   **[Agent2Agent Protocol (A2A)](https://google.github.io/A2A/):** The open standard used for communication and task delegation between the ADK Host Agent and backend "Specialist Agents". Enables interoperability between potentially different agent implementations.
*   **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/):** The open standard allowing Specialist Agents (acting as MCP Clients) to securely interact with dedicated "Tool Servers" (MCP Servers) that provide access to specific functionalities or data (e.g., fetching stock prices).

## Architecture

This diagram illustrates the high-level conceptual architecture Project Horizon aims to enable: a user interacting with a primary Host Agent, which delegates tasks to various Specialist Agents (potentially built with different frameworks) via A2A. These Specialist Agents then leverage MCP to securely access specific tools or data sources.

![High-Level Architecture](assets/high-level-architecture.png)



## Current Scenario Demonstrated (v0.1 - Synchronous Stock Lookup)

The initial proof-of-concept focuses on a clear, synchronous end-to-end flow: **Real-Time Stock Price Lookup**.

**User Interaction:**
The user interacts with the Host Agent via a simple web UI (adapted from Project Pastra), primarily using voice input. They ask for the current price of a stock (e.g., *"What is the price of Microsoft?"*).

**Execution Flow:**

1.  **Input (ADK Live):** The frontend captures user audio and streams it via WebSocket to the ADK Live Server (app/live_server.py).
2.  **Host Agent Processing (ADK):** The live_server passes the input to the ADK Runner managing the HostAgent. The HostAgent (using a Live API compatible Gemini model) processes the audio/text input, identifies the intent and the stock ticker symbol (e.g., "MSFT").
3.  **Tool Invocation (ADK -> A2A Client):** The HostAgent decides to use its call_specialist_stock_agent tool (an ADK FunctionTool). This tool acts as an A2A Client.
4.  **Delegation (A2A Request):** The tool constructs an A2A tasks/send request containing the ticker symbol and sends it via HTTP POST to the configured endpoint of the StockInfoAgent.
5.  **Task Reception (A2A Server):** The StockInfoAgent (specialist_agents/stock_info_agent/__main__.py), acting as an A2A Server, receives the HTTP request, parses the A2A task.
6.  **ADK Agent Execution (within A2A Server):** The `StockInfoTaskManager`'s `on_send_task` method is invoked. This method instantiates the ADK `stock_info_agent` (from `specialist_agents/stock_info_agent/agent.py`), which in turn initializes its `MCPToolset` using `StdioServerParameters`. This initialization step **launches the StockToolServer** (`mcp_servers/stock_mcp_server/server.py`) as a subprocess and captures the necessary `exit_stack` for cleanup. The Task Manager then creates and starts a temporary ADK `Runner` configured with this agent instance.
7.  **MCP Tool Call (via ADK Agent & Toolset):** The temporary ADK `Runner` executes the `stock_info_agent`. The agent's LLM processes the input query (e.g., "What is the stock price for AAPL?") and decides to use the `get_current_stock_price` tool. The agent's `MCPToolset` handles establishing the stdio connection and sending the actual MCP `tools/call` request to the running StockToolServer subprocess.
8.  **Tool Execution (MCP Server):** The StockToolServer, acting as an MCP Server (using FastMCP), receives the request via stdin, executes the `get_current_stock_price` function (which uses the `yfinance` library), and gets the stock data.
9.  **MCP Response (stdio):** The StockToolServer sends the result (price/currency dictionary or error) back as an MCP JSON-RPC response via stdout.
10. **ADK Agent Receives Tool Result:** The StockToolServer sends its response via stdout. The ADK `MCPToolset` receives this, parses it, and wraps it in an ADK `FunctionResponse` event. The temporary ADK `Runner` yields this event. **Task Manager Extracts Result & Cleans Up:** The `StockInfoTaskManager`'s `on_send_task` method, iterating through the events from the `Runner`, receives the `FunctionResponse`, extracts the stock data dictionary from it. Once the `Runner`'s execution loop finishes for this task, the `finally` block within `on_send_task` explicitly calls `await exit_stack.aclose()`, **terminating the StockToolServer subprocess.**
11. **Task Completion (A2A Response):** The `StockInfoTaskManager` formats the extracted stock data (or error) into an A2A Artifact (within a Task object marked as COMPLETED or FAILED) and sends it back as the HTTP response to the Host Agent's A2A Client tool.
12. **Tool Result (A2A Client -> ADK):** The `call_specialist_stock_agent` tool in the HostAgent receives the A2A HTTP response, parses the result/artifact, and returns the dictionary to the ADK Runner.
13. **Final Response Generation (ADK):** The HostAgent receives the tool result dictionary. Its LLM formulates a natural language response (e.g., "Microsoft is trading at $XXX.XX USD.").
14. **Audio Output (ADK Live):** ADK synthesizes the text response into audio using the configured Live API model/voice.
15. **Streaming Output (ADK Live -> UI):** The `live_server` streams the audio chunks via WebSocket back to the user interface, where it is played back in real-time.

**This initial phase successfully validates the core integration points between ADK Live, A2A, and MCP over stdio in a synchronous workflow.**

## Getting Started

### Prerequisites

*   Python (>= 3.9 required by MCP, >= 3.11 recommended for latest ADK/async features)
*   `pip` or `uv` (Python package installer)
*   Git
*   Access to Google Cloud / Google AI Studio (for Gemini API Key/Credentials)
*   An API key for a Gemini model supporting the Live API (e.g., `gemini-2.0-flash-exp`).
*   Basic understanding of ADK, A2A, and MCP concepts.
*   Familiarity with `asyncio` and web frameworks (FastAPI/Starlette).

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd project-horizon
    ```
2.  **Set up Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or: uv install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `google-adk`, `uvicorn`, `fastapi`, `starlette`, `websockets`, `python-dotenv`, `mcp[cli]`, `yfinance`, `httpx`, `pyaudio`)*
4.  **Copy A2A Common Library:** Ensure the `common/` directory containing the A2A client/server base code is present in the project root.

### Configuration (`.env`)

Create a `.env` file in the project root (`live-agent-project/`) and populate it with the correct values:

```dotenv
# Google Cloud/Gemini Configuration
# Option 1: Google AI Studio Key (Requires GOOGLE_GENAI_USE_VERTEXAI=FALSE or not set)
GOOGLE_API_KEY=YOUR_GOOGLE_AI_STUDIO_API_KEY
# Option 2: Vertex AI (Requires GOOGLE_GENAI_USE_VERTEXAI=TRUE and ADC login)
# GOOGLE_GENAI_USE_VERTEXAI=TRUE
# PROJECT_ID=your-gcp-project-id
# VERTEX_LOCATION=us-central1

# --- IMPORTANT: Use a Live API compatible model ---
GEMINI_LIVE_MODEL_ID=gemini-2.0-flash-exp # Or other compatible model

# A2A Server Endpoint URL (Where StockInfoAgent listens)
SPECIALIST_AGENT_A2A_URL=http://127.0.0.1:8001/a2a

# MCP Server Path (Absolute path needed by Specialist Agent)
# Replace with the actual absolute path on your system!
STOCK_MCP_SERVER_PATH=/absolute/path/to/project-horizon/mcp_servers/stock_mcp_server/server.py

# ADK Live Server Configuration (Frontend connects here)
LIVE_SERVER_HOST=127.0.0.1
LIVE_SERVER_PORT=8081 # Port for the ADK/WebSocket server

# Specialist A2A Server Configuration
SPECIALIST_SERVER_HOST=127.0.0.1
SPECIALIST_SERVER_PORT=8001 # Port for the A2A server (MUST BE DIFFERENT from LIVE_SERVER_PORT)

# Logging Level (Optional: DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```
**Get Absolute Path:** Use `pwd` (Linux/macOS) or `cd` (Windows) in the `mcp_servers/stock_mcp_server` directory to find the path, then append `/server.py`.

### Running the Demo

Execute the components in separate terminals from the project root directory (`live-agent-project/`), ensuring your virtual environment is active in each.

1.  **Terminal 1: Start Specialist Agent (A2A Server)**
    ```bash
    python -m specialist_agents.stock_info_agent
    # Or: uvicorn specialist_agents.stock_info_agent.server:app --host $SPECIALIST_SERVER_HOST --port $SPECIALIST_SERVER_PORT --log-level $LOG_LEVEL
    ```
    *Verify it starts listening on the correct port (e.g., 8001) and uses the correct MCP server path.*

2.  **Terminal 2: Start ADK Live Server**
    ```bash
    python -m app.live_server
    # Or: uvicorn app.live_server:app --host $LIVE_SERVER_HOST --port $LIVE_SERVER_PORT --log-level $LOG_LEVEL
    ```
    *Verify it starts listening on its port (e.g., 8081).*

3.  **Access UI:** Open your web browser to `http://<LIVE_SERVER_HOST>:<LIVE_SERVER_PORT>` (e.g., `http://127.0.0.1:8081`).

4.  **Interact:**
    *   Click "Connect".
    *   Click the Mic button, speak a stock request (e.g., "What is the price of Apple?").
    *   Click the Mic button again to stop recording.
    *   Listen for the audio response.
    *   Check the logs in both terminals to see the A2A and MCP interactions.
    *   Click "Stop" in the UI when finished.

## Project Structure
```
live-agent-project/
├── host_agent/ # ADK Host Agent (User Facing, Live API)
│ ├── init.py
│ ├── agent.py # ADK Agent definition, instructions, A2A tool config
│ └── tools.py # Custom ADK FunctionTool acting as A2A Client
├── app/ # ADK Live Server and Frontend UI
│ ├── static/ # Simplified Pastra UI files (HTML, CSS, JS)
│ │ ├── index.html
│ │ ├── styles.css
│ │ └── adk-websocket-api.js
│ │ └── audio/ # (Copied from Pastra)
│ │ └── utils/ # (Copied from Pastra)
│ └── live_server.py # FastAPI/WebSocket server bridging UI and ADK Runner
├── common/ # Copied A2A protocol library code (client/server bases)
│ ├── client/
│ ├── server/
│ └── types.py # A2A Pydantic models
├── mcp_servers/ # MCP Tool Servers
│ └── stock_mcp_server/ # Stock Price Tool Server
│ ├── init.py
│ └── server.py # FastMCP server using yfinance (stdio)
├── specialist_agents/ # Backend agents communicating via A2A
│ └── stock_info_agent/ # Stock Information Specialist Agent
│ ├── init.py
│ ├── main.py # Entry point to run the A2A server
│ ├── agent_card.json # A2A discovery file (loaded by main)
│ ├── mcp_client_logic.py # Logic acting as MCP Client (stdio)
│ ├── server.py # A2A Server implementation (Starlette/common)
│ └── task_manager.py # A2A Task handling logic
├── tests/ # Unit and Integration Tests
│ └── test_a2a_stock_client.py # Script to test A2A server directly
├── .env # Environment variables (API Keys, URLs, Ports)
├── requirements.txt # Python dependencies
└── README.md # This file
```


## Future Plans / Roadmap

This initial PoC establishes the synchronous foundation. Future iterations aim to explore:

*   **Asynchronous A2A/MCP:** Implement non-blocking calls, potentially using A2A streaming or push notifications for long-running tool operations within the Specialist Agent -> MCP Server interaction.
*   **Parallel A2A Delegation:** Modify the Host Agent to delegate tasks to multiple Specialist Agents concurrently (e.g., fetch stock price AND company news simultaneously).
*   **MCP over HTTP/SSE:** Migrate the `StockToolServer` from stdio to http/sse transport, requiring security considerations (authentication middleware) for the Specialist Agent's MCP client connection.
*   **Advanced State Management:** Utilize ADK's persistent SessionService options (Database, Vertex AI) and explore state sharing patterns between agents.
*   **Error Handling & Resilience:** Implement more robust error handling, retries, and fallback mechanisms across all communication layers.
*   **Additional Specialist Agents & Tools:** Expand the system with more agents (e.g., NewsAgent, PortfolioAgent) and corresponding MCP Tool Servers.
*   **ADK Evaluation:** Integrate ADK's evaluation framework (`adk eval`) to measure performance and accuracy.
*   **UI Enhancements:** Improve the frontend to better visualize multi-agent activity, tool calls, and potentially handle more complex inputs/outputs.

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file.

## 🤝 Contributing & Disclaimer

This is a personal project by [Heiko Hotz](https://github.com/heiko-hotz) to explore Gemini capabilities. Suggestions and feedback are welcome via Issues or Pull Requests.

**This project is developed independently and does not reflect the views or efforts of Google.**