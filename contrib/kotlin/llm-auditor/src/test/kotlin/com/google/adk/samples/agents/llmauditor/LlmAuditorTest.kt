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

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class LlmAuditorTest {
    @Test
    fun testCriticPromptContainsClaimsInstruction() {
        assertNotNull(CRITIC_PROMPT)
        assertTrue(CRITIC_PROMPT.contains("claims"))
    }

    @Test
    fun testReviserPromptContainsReviserInstruction() {
        assertNotNull(REVISER_PROMPT)
        assertTrue(REVISER_PROMPT.contains("fact-check"))
    }

    @Test
    fun testRootAgentInitialization() {
        val rootAgent = LlmAuditorAgent.rootAgent
        assertNotNull(rootAgent)
        assertEquals("llm_auditor", rootAgent.name)
        assertEquals(2, rootAgent.subAgents.size)
        assertEquals("critic_agent", rootAgent.subAgents[0].name)
        assertEquals("reviser_agent", rootAgent.subAgents[1].name)
    }
}
