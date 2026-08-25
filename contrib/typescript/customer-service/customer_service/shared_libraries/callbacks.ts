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
 * Callback functions for FOMC Research Agent.
 */

import { BaseTool, LlmRequest, ToolContext, CallbackContext } from '@google/adk';
import { State } from '@google/adk/dist/types/sessions/state';


import { Customer } from '../entities/customer';

const RATE_LIMIT_SECS = 60;
const RPM_QUOTA = 10;

/**
 * Callback function that implements a query rate limit.
 * * FIXED: Converted to object destructuring to match 'BeforeModelCallback' signature.
 * * TYPE: Returns Promise<any> to satisfy 'LlmResponse | undefined' requirement.
 */
export async function rateLimitCallback({
  context: callbackContext,
  request: llmRequest
}: {
    context: CallbackContext;
    request: LlmRequest;
}): Promise<any> {
  // Add null checks for llmRequest and llmRequest.contents
  if (!llmRequest || !llmRequest.contents) {
    console.debug('llmRequest or llmRequest.contents is undefined. Skipping rate limit.');
    return undefined;
  }

  for (const content of llmRequest.contents) {
    if (content.parts) {
      for (const part of content.parts) {
        if ('text' in part && part.text === '') {
          part.text = ' ';
        }
      }
    }
  }

  const now = Date.now() / 1000; // Time in seconds

  if (!callbackContext.state.has('timer_start')) {
    callbackContext.state.set('timer_start', now);
    callbackContext.state.set('request_count', 1);
    console.debug(
      `rate_limit_callback [timestamp: ${now}, req_count: 1, elapsed_secs: 0]`
    );
    return undefined;
  }

  const requestCount = (callbackContext.state.get<number>('request_count') || 0) + 1;
  const timerStart = callbackContext.state.get<number>('timer_start') || now;
  const elapsedSecs = now - timerStart;

  console.debug(
    `rate_limit_callback [timestamp: ${now}, request_count: ${requestCount}, elapsed_secs: ${elapsedSecs}]`
  );

  if (requestCount > RPM_QUOTA) {
    const delay = RATE_LIMIT_SECS - elapsedSecs + 1;
    if (delay > 0) {
      console.debug(`Sleeping for ${delay} seconds`);
      // Async sleep
      await new Promise((resolve) => setTimeout(resolve, delay * 1000));
    }
    // Reset timer
    callbackContext.state.set('timer_start', Date.now() / 1000);
    callbackContext.state.set('request_count', 1);
  } else {
    callbackContext.state.set('request_count', requestCount);
  }

  return undefined;
}

/**
 * Validates the customer ID against the customer profile in the session state.
 */
export function validateCustomerId({
  customerId,
  sessionState,
}: {
  customerId: string;
  sessionState: State;
}): [boolean, string | null] {
  if (!sessionState.has('customer_profile')) {
    return [false, 'No customer profile selected. Please select a profile.'];
  }

  try {
    const profileJson = sessionState.get<string>('customer_profile');
    
    if (!profileJson) {
      return [false, 'Customer profile is empty.'];
    }

    const profileData = JSON.parse(profileJson);
    const c = new Customer(profileData);

    if (customerId === c.customer_id) {
      return [true, null];
    } else {
      return [
        false,
        `You cannot use the tool with customer_id ${customerId}, only for ${c.customer_id}.`,
      ];
    }
  } catch (e) {
    return [
      false,
      "Customer profile couldn't be parsed. Please reload the customer data.",
    ];
  }
}

/**
 * Make dictionary values lowercase recursively.
 */
function lowercaseValue(value: unknown): unknown {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const newDict: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      newDict[k] = lowercaseValue(v);
    }
    return newDict;
  } else if (typeof value === 'string') {
    return value.toLowerCase();
  } else if (Array.isArray(value)) {
    return value.map((i) => lowercaseValue(i));
  } else {
    return value;
  }
}

/**
 * Callback Method: Before Tool Execution
 * * PATTERN: Object Destructuring (Required for Tools)
 */
export function beforeTool({
  tool,
  args,
  context: toolContext
}: {
  tool: BaseTool;
  args: Record<string, any>;
  context: ToolContext;
}): Record<string, any> | undefined {
  // Make sure all values that the agent is sending to tools are lowercase
  const lowercasedArgs = lowercaseValue(args) as Record<string, any>;
  
  // Mutate args in place
  for (const key in args) {
    if (Object.prototype.hasOwnProperty.call(args, key)) {
      delete args[key];
    }
  }
  Object.assign(args, lowercasedArgs);

  // Several tools require customer_id as input. We validate it.
  if ('customer_id' in args) {
    const [valid, err] = validateCustomerId({
      customerId: args['customer_id'],
      sessionState: toolContext.state,
    });
    if (!valid && err) {
      return { error: err };
    }
  }

  if (tool.name === 'sync_ask_for_approval') {
    const amount = args['value'];
    if (amount !== undefined && Number(amount) <= 10) {
      return {
        status: 'approved',
        message: 'You can approve this discount; no manager needed.',
      };
    }
  }

  if (tool.name === 'modify_cart') {
    if (args['items_added'] === true && args['items_removed'] === true) {
      return { result: 'I have added and removed the requested items.' };
    }
  }

  return undefined;
}

/**
 * Callback Method: After Tool Execution
 * * PATTERN: Object Destructuring with 'response' alias (Required for Tools)
 */
export function afterTool({
  tool,
  args,
  context: ToolContext,
  response: toolResponse
}: {
  tool: BaseTool;
  args: Record<string, unknown>;
  context: ToolContext;
  response: Record<string, unknown>;
}): Record<string, unknown> | undefined {
  
  if (tool.name === 'sync_ask_for_approval') {
    if (toolResponse && toolResponse['status'] === 'approved') {
      console.debug('Applying discount to the cart');
      // Actually make changes to the cart
    }
  }

  if (tool.name === 'approve_discount') {
    if (toolResponse && toolResponse['status'] === 'ok') {
      console.debug('Applying discount to the cart');
      // Actually make changes to the cart
    }
  }

  return undefined;
}

/**
 * Callback Method: Before Agent Execution
 * * PATTERN: Positional Argument (Required for Agents)
 * * FIXED: Uses CallbackContext directly.
 */ 
export function beforeAgent(callbackContext: CallbackContext): undefined {
  // Use state directly from the callbackContext, matching your working example.
  if (!callbackContext.state.has('customer_profile')) {
    const customer = Customer.getCustomer('123');
    if (customer) {
      callbackContext.state.set('customer_profile', customer.toJson());
    }
  }
}