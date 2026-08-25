 /**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Agent module for the customer service agent.
 */

import { LlmAgent, InMemoryRunner, isFinalResponse } from '@google/adk';
import { config } from './config';
import { createUserContent, Part } from '@google/genai';
import { GLOBAL_INSTRUCTION, INSTRUCTION } from './prompts';
import {
  rateLimitCallback,
  beforeAgent,
  beforeTool,
  afterTool,
} from './shared_libraries/callbacks';
import {
    sendCallCompanionLinkTool,
    approveDiscountTool,
    syncAskForApprovalTool,
    updateSalesforceCrmTool,
    accessCartInformationTool,
    modifyCartTool,
    getProductRecommendationsTool,
    checkProductAvailabilityTool,
    schedulePlantingServiceTool,
    getAvailablePlantingTimesTool,
    sendCareInstructionsTool,
    generateQrCodeTool
} from './tools/function_tools';

// Combine instructions as LlmAgent typically takes a single instruction string
// or you can handle globalInstruction separately if your ADK version supports it.
const COMBINED_INSTRUCTION = `${GLOBAL_INSTRUCTION}\n\n${INSTRUCTION}`;

export const rootAgent = new LlmAgent({
  model: config.agentSettings.model,
  name: config.agentSettings.name,
  instruction: COMBINED_INSTRUCTION,
  // Note: Ensure these functions are wrapped as valid Tool objects (e.g., FnTool)
  // if your ADK implementation requires explicit schemas.
  tools: [
    sendCallCompanionLinkTool,
    approveDiscountTool,
    syncAskForApprovalTool,
    updateSalesforceCrmTool,
    accessCartInformationTool,
    modifyCartTool,
    getProductRecommendationsTool,
    checkProductAvailabilityTool,
    schedulePlantingServiceTool,
    getAvailablePlantingTimesTool,
    sendCareInstructionsTool,
    generateQrCodeTool
  ],
  // Mapping callbacks to their TypeScript equivalents
  beforeToolCallback: beforeTool,
  afterToolCallback: afterTool,
  beforeAgentCallback: beforeAgent,
  beforeModelCallback: rateLimitCallback,
});