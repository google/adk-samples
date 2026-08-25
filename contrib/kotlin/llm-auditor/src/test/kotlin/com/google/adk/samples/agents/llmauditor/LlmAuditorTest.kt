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

import com.google.adk.kt.models.Gemini
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class LlmAuditorTest {
    @Test
    fun testCriticPromptConstants() {
        assertNotNull(CRITIC_PROMPT)
        assertTrue(CRITIC_PROMPT.contains("CLAIMS"))
        assertTrue(CRITIC_PROMPT.contains("Verify each CLAIM"))
    }

    @Test
    fun testReviserPromptConstants() {
        assertNotNull(REVISER_PROMPT)
        assertTrue(REVISER_PROMPT.contains("VERDICT"))
        assertEquals("---END-OF-EDIT---", END_MARK)
    }

    @Test
    fun testCriticAgentFactory() {
        val model = Gemini(apiKey = "fake-key-for-test", name = "gemini-flash-latest")
        val critic = createCriticAgent(model)
        assertNotNull(critic)
        assertEquals("critic_agent", critic.name)
        assertEquals(1, critic.tools.size)
    }

    @Test
    fun testReviserAgentFactory() {
        val model = Gemini(apiKey = "fake-key-for-test", name = "gemini-flash-latest")
        val reviser = createReviserAgent(model)
        assertNotNull(reviser)
        assertEquals("reviser_agent", reviser.name)
        assertEquals(1, reviser.afterModelCallbacks.size)
    }
}
