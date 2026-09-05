-- Copyright 2026 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     https://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Optimized join of Cloud Logging BQ export data with GCS-stored prompt/response data.
-- This query extracts both input and output messages referenced in logs.
-- Note: Input files contain full conversation history, so messages may appear multiple times.
--
-- Log data is exported directly to BigQuery via log sinks.
-- The table `gen_ai_client_inference_operation_details` is pre-created by Terraform
-- and populated by Cloud Logging via the sink.
-- Labels are flattened into individual columns (dots replaced with underscores).

-- Extract message references from Cloud Logging BQ export (scan once, extract both input/output)
WITH log_refs AS (
  SELECT
    insertId AS insert_id,
    timestamp,
    trace,
    spanId AS span_id,
    labels.gen_ai_conversation_id AS conversation_id,
    labels.gen_ai_input_messages_ref AS input_ref,
    labels.gen_ai_output_messages_ref AS output_ref,
    labels.gen_ai_usage_input_tokens AS usage_input_tokens,
    labels.gen_ai_usage_output_tokens AS usage_output_tokens,
    labels.gen_ai_agent_name AS agent_name,
    labels.gen_ai_response_finish_reasons AS finish_reasons,
    labels.user_id
  FROM `${project_id}.${dataset_id}.gen_ai_client_inference_operation_details`
  WHERE labels.gen_ai_input_messages_ref IS NOT NULL
     OR labels.gen_ai_output_messages_ref IS NOT NULL
),

-- Unpivot to get one row per message reference
unpivoted_refs AS (
  SELECT
    insert_id,
    timestamp,
    trace,
    span_id,
    conversation_id,
    usage_input_tokens,
    usage_output_tokens,
    agent_name,
    finish_reasons,
    user_id,
    input_ref AS messages_ref_uri,
    'input' AS message_type
  FROM log_refs
  WHERE input_ref IS NOT NULL

  UNION ALL

  SELECT
    insert_id,
    timestamp,
    trace,
    span_id,
    conversation_id,
    usage_input_tokens,
    usage_output_tokens,
    agent_name,
    finish_reasons,
    user_id,
    output_ref AS messages_ref_uri,
    'output' AS message_type
  FROM log_refs
  WHERE output_ref IS NOT NULL
),

-- Join with completions external table and extract api_call_id once
joined_data AS (
  SELECT
    lr.insert_id,
    lr.timestamp,
    lr.trace,
    lr.span_id,
    lr.conversation_id,
    lr.usage_input_tokens,
    lr.usage_output_tokens,
    lr.agent_name,
    lr.finish_reasons,
    lr.user_id,
    lr.messages_ref_uri,
    lr.message_type,
    SPLIT(REGEXP_EXTRACT(lr.messages_ref_uri, r'/([^/]+)\.jsonl'), '_')[OFFSET(0)] AS api_call_id,
    c.role,
    c.parts,
    c.index AS message_idx
  FROM unpivoted_refs lr
  JOIN `${project_id}.${dataset_id}.${completions_external_table}` c
    ON lr.messages_ref_uri = c._FILE_NAME
),

-- Flatten the parts array
flattened AS (
  SELECT
    insert_id,
    timestamp,
    trace,
    span_id,
    conversation_id,
    usage_input_tokens,
    usage_output_tokens,
    agent_name,
    finish_reasons,
    user_id,
    messages_ref_uri,
    message_type,
    api_call_id,
    role,
    message_idx,
    part_idx,
    part.type AS part_type,
    part.content,
    part.uri,
    part.mime_type,
    TO_HEX(MD5(part.data)) AS data_md5_hex,
    part.id AS tool_id,
    part.name AS tool_name,
    part.arguments AS tool_args,
    part.response AS tool_response
  FROM joined_data
  CROSS JOIN UNNEST(parts) AS part WITH OFFSET AS part_idx
),

-- Stage 1: Deduplicate within a trace+span, keeping distinct spans
-- (each LLM hop gets its own span_id).
dedup_within_trace AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY conversation_id, trace, span_id, message_type, role, message_idx, part_idx
      ORDER BY timestamp DESC
    ) AS row_num
  FROM flattened
),

-- Stage 2: Deduplicate across traces within a conversation
-- Input files contain full conversation history, so earlier messages repeat in later traces.
-- For inputs: keep only the latest trace's copy of each message.
-- For outputs: each trace has a unique response, so keep all.
dedup_across_traces AS (
  SELECT
    *,
    CASE
      WHEN message_type = 'input' THEN ROW_NUMBER() OVER (
        PARTITION BY conversation_id, message_type, role, message_idx, part_idx
        ORDER BY timestamp ASC
      )
      ELSE 1
    END AS cross_trace_row_num
  FROM dedup_within_trace
  WHERE row_num = 1
)

SELECT
  -- Core identifiers and timestamps
  timestamp,
  insert_id,
  trace,
  span_id,
  conversation_id,
  api_call_id,

  -- Message metadata
  message_type,
  role,
  message_idx,
  part_idx,

  -- Message content
  content,

  -- Tool/function calling
  part_type,
  tool_name,
  tool_args,
  tool_response,

  -- Usage metadata
  usage_input_tokens,
  usage_output_tokens,
  agent_name,
  finish_reasons,
  user_id,

  -- Additional metadata
  uri,
  mime_type,
  data_md5_hex,
  messages_ref_uri
FROM dedup_across_traces
WHERE cross_trace_row_num = 1
  -- Exclude assistant messages echoed back as input context (already captured as output)
  AND NOT (message_type = 'input' AND role = 'assistant')
ORDER BY conversation_id ASC, timestamp ASC, message_type ASC, message_idx ASC, part_idx ASC
