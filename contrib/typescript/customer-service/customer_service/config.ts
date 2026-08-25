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

import { env } from 'node:process';

/**
 * Configuration module for the customer service agent.
 */

/**
 * Helper function to read environment variables.
 * @param key The environment variable key.
 * @param defaultValue The default value if the key is not set.
 */
function getEnv(key: string, defaultValue: string): string {
  return env[key] || defaultValue;
}

export class AgentModel {
  name: string = 'customer_service_agent';
  model: string = 'gemini-2.5-flash';
}

export class Config {
  agentSettings: AgentModel = new AgentModel();
  appName: string = 'customer_service_app';

  CLOUD_PROJECT: string = getEnv('GOOGLE_CLOUD_PROJECT', 'my_project');
  CLOUD_LOCATION: string = getEnv('GOOGLE_CLOUD_LOCATION', 'us-central1');
  GENAI_USE_VERTEXAI: string = getEnv('GOOGLE_GENAI_USE_VERTEXAI', '1');
  API_KEY: string = getEnv('GOOGLE_API_KEY', 'GOOGLE_API_KEY');
}

// Export a singleton instance of the config
export const config = new Config();