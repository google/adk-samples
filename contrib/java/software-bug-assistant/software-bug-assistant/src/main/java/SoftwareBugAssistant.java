package com.google.adk.samples.agents.softwarebugassistant;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.adk.agents.BaseAgent;
import com.google.adk.agents.LlmAgent;
import com.google.adk.tools.AgentTool;
import com.google.adk.tools.BaseTool;
import com.google.adk.tools.GoogleSearchTool;
import com.google.adk.tools.mcp.McpToolset;
import com.google.adk.tools.mcp.SseServerParameters;
import com.google.adk.events.Event;
import com.google.adk.runner.InMemoryRunner;
import com.google.adk.sessions.Session;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;


public class SoftwareBugAssistant {

    private static final Logger logger = Logger.getLogger(SoftwareBugAssistant.class.getName());

    // --- Define Constants ---
    private static final String MODEL_NAME = "gemini-2.5-flash";
    private static String NAME = "SoftwareBugAssistant";
    private static String USER_ID = "test-user";

    // ROOT_AGENT needed for ADK Web UI.
    public static final BaseAgent ROOT_AGENT = initAgent();


    public static void main(String[] args) {
      InMemoryRunner runner = new InMemoryRunner(ROOT_AGENT);
      Session session =
          runner
              .sessionService()
              .createSession(NAME, USER_ID)
              .blockingGet();

      Runtime.getRuntime().addShutdownHook(new Thread(() -> {
        System.out.println("\nExiting SoftwareBugAssistant. Goodbye!");
      }));

      try (Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8)) {
        while (true) {
          System.out.print("\nYou > ");
          String userInput = scanner.nextLine();
          if ("quit".equalsIgnoreCase(userInput)) {
            break;
          }
          Content userMsg = Content.fromParts(Part.fromText(userInput));
          Flowable<Event> events = runner.runAsync(USER_ID, session.id(), userMsg);
          System.out.print("\nAgent > ");
          events.blockingForEach(event -> System.out.println(event.stringifyContent()));
        }
      }
    }


    public static BaseAgent initAgent() {
        // Set up MCP Toolbox connection to Cloud SQL
        try {
            String mcpServerUrl = System.getenv("MCP_TOOLBOX_URL");
            if (mcpServerUrl == null || mcpServerUrl.isEmpty()) {
                mcpServerUrl = "http://127.0.0.1:5000/mcp/"; // Fallback URL
                logger.warning("⚠️ MCP_TOOLBOX_URL environment variable not set, using default:" + mcpServerUrl);
            }

            SseServerParameters params = SseServerParameters.builder().url(mcpServerUrl).build();
            logger.info("🕰️ Initializing MCP toolset with params" + params);

            McpToolset.McpToolsAndToolsetResult result = McpToolset.fromServer(params, new ObjectMapper()).get();
            logger.info("⭐ MCP Toolset initialized: " + result.toString());
            if (result.getTools() != null && !result.getTools().isEmpty()) {
                logger.info("⭐ MCP Tools loaded: " + result.getTools().size());
            }
            List<BaseTool> mcpTools = result.getTools().stream()
                    .map(mcpTool -> (BaseTool) mcpTool)
                    .collect(Collectors.toList());
            logger.info("🛠️ MCP TOOLS: " + mcpTools.toString());

            // Add GoogleSearch tool - Workaround for https://github.com/google/adk-python/issues/134 
            LlmAgent googleSearchAgent = LlmAgent.builder()
                    .model("gemini-2.5-flash")
                    .name("google_search_agent")
                    .description("Search Google Search")
                    .instruction("""
                You're a specialist in Google Search
                """)
                    .tools(new GoogleSearchTool()) // Your Google search tool
                    .outputKey("google_search_result")
                    .build();
            AgentTool searchTool = AgentTool.create(googleSearchAgent, false);
            List<BaseTool> allTools = new ArrayList<>(mcpTools);
            allTools.add(searchTool);
            logger.info("🌈 ALL TOOLS: " + allTools.toString());
            return LlmAgent.builder()
                    .model(MODEL_NAME)
                    .name("SoftwareBugAssistant")
                    .description("Helps fix bugs")
                    .instruction(
                            """
                            You are a skilled expert in triaging and debugging software issues for a coffee machine company,QuantumRoast.

                            Your general process is as follows:

                            1. **Understand the user's request.** Analyze the user's initial request to understand the goal - for example, "I am seeing X issue. Can you help me find similar open issues?" If you do not understand the request, ask for more information.   
                            2. **Identify the appropriate tools.** You will be provided with tools for a SQL-based bug ticket database (create, update, search tickets by description). You will also be able to web search via Google Search. Identify one **or more** appropriate tools to accomplish the user's request.  
                            3. **Populate and validate the parameters.** Before calling the tools, do some reasoning to make sure that you are populating the tool parameters correctly. For example, when creating a new ticket, make sure that the Title and Description are different, and that the Priority field is set. Use common sense to assign P0 to high priority issues, down to P3 for low-priority issues. Always set the default status to “open” especially for new bugs.   
                            4. **Call the tools.** Once the parameters are validated, call the tool with the determined parameters.  
                            5. **Analyze the tools' results, and provide insights back to the user.** Return the tools' result in a human-readable format. State which tools you called, if any. If your result is 2 or more bugs, always use a markdown table to report back. If there is any code, or timestamp, in the result, format the code with markdown backticks, or codeblocks.   
                            6. **Ask the user if they need anything else.**  
                """)
                    .tools(allTools)
                    .outputKey("bug_assistant_result")
                    .build();
        } catch (Exception e) {
            logger.info("Error initializing MCP toolset and starting agent " + e.getMessage());
            return null;
        }
    }
}
