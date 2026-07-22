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
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func TestHappyPath(t *testing.T) {
	ctx := context.Background()
	llmAuditorAgent := GetLLmAuditorAgent(ctx)

	sessionService := session.InMemoryService()
	config := runner.Config{
		AppName:        "llm_auditor",
		Agent:          llmAuditorAgent,
		SessionService: sessionService,
	}
	r, err := runner.New(config)
	if err != nil {
		t.Fatal(err)
	}

	s, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: "llm_auditor",
		UserID:  "test_user",
	})
	if err != nil {
		t.Fatal(err)
	}

	userInput := `Double check this:
Question: Why is the sky blue?
Answer: Because the water is blue.`

	userMsg := &genai.Content{
		Parts: []*genai.Part{genai.NewPartFromText(userInput)},
		Role:  string(genai.RoleUser),
	}

	var response string
	for event, err := range r.Run(ctx, "test_user", s.Session.ID(), userMsg, agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if event.ErrorCode != "" {
			t.Fatalf("Event error: %s - %s", event.ErrorCode, event.ErrorMessage)
		}
		if event.Content != nil && len(event.Content.Parts) > 0 {
			response += event.Content.Parts[0].Text
		}
	}

	if !strings.Contains(strings.ToLower(response), "scattering") {
		t.Errorf("Expected response to contain 'scattering', but it didn't. Response: %s", response)
	}
}
