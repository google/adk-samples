// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package auditor

import (
	"context"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/genai"

	"llmauditor/critic"
	"llmauditor/reviser"
)

func GetLLmAuditorAgent(ctx context.Context) agent.Agent {
	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{})
	if err != nil {
		panic(err)
	}

	criticAgent, err := critic.New(model)
	if err != nil {
		panic(err)
	}

	reviserAgent, err := reviser.New(model)
	if err != nil {
		panic(err)
	}

	rootAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "llm_auditor",
			Description: "Evaluates LLM-generated answers.",
			SubAgents: []agent.Agent{
				criticAgent,
				reviserAgent,
			},
		},
	})
	if err != nil {
		panic(err)
	}

	return rootAgent
}
