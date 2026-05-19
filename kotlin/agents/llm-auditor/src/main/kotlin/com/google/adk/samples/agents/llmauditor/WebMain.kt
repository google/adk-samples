/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.adk.samples.agents.llmauditor

import com.google.adk.kt.artifacts.InMemoryArtifactService
import com.google.adk.kt.runners.InMemoryRunner
import com.google.adk.kt.sessions.InMemorySessionService
import com.google.adk.kt.webserver.AdkWebServer
import com.google.adk.kt.webserver.loaders.SingleAgentLoader
import com.google.adk.kt.webserver.telemetry.ApiServerSpanExporter

fun main() {
    val agent = LlmAuditorAgent.rootAgent
    val sessionService = InMemorySessionService()
    val artifactService = InMemoryArtifactService()

    val server = AdkWebServer(
        port = 8080,
        sessionService = sessionService,
        artifactService = artifactService,
        agentLoader = SingleAgentLoader(agent),
        runner = InMemoryRunner(
            agent = agent,
            sessionService = sessionService,
            artifactService = artifactService,
        ),
        apiServerSpanExporter = ApiServerSpanExporter(),
    )

    println("Starting ADK web server on http://localhost:8080...")
    server.start(wait = true)
}
