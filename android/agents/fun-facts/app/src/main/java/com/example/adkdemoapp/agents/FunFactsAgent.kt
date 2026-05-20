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

package com.example.adkdemoapp.agents

import com.example.adkdemoapp.BuildConfig
import com.google.adk.kt.agents.Instruction
import com.google.adk.kt.agents.LlmAgent
import com.google.adk.kt.models.Gemini
import com.google.adk.kt.types.GenerateContentConfig

//import com.google.adk.kt.tools.GoogleSearchTool

object FunFactsAgent {
    @JvmField
    val rootAgent = LlmAgent(
        name = "fun_facts",
        description = "An agent that provides fun facts about a given topic.",
        model = Gemini(name = "gemini-3-flash-preview", apiKey = BuildConfig.GEMINI_API_KEY),
        instruction = Instruction(
            "Provide the most mind-blowing, obscure, and wacky fun facts about "
                    + "the topic. Aim for maximum 'wow' factor with rare and surprising "
                    + "information."
        ),
        tools = listOf(),//GoogleSearchTool()),
        )
}
