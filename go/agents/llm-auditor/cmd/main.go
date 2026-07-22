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

package main

import (
	"context"
	"log"
	"os"

	"llmauditor/auditor"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/session"
)

func main() {
	ctx := context.Background()
	llmAuditorAgent := auditor.GetLLmAuditorAgent(ctx)

	sessionService := session.InMemoryService()
	artifactservice := artifact.InMemoryService()

	config := &launcher.Config{
		SessionService:  sessionService,
		AgentLoader:     agent.NewSingleLoader(llmAuditorAgent),
		ArtifactService: artifactservice,
	}

	args := os.Args[1:]
	if len(args) == 0 {
		args = []string{"web", "api", "webui"}
	}

	l := full.NewLauncher()
	if err := l.Execute(ctx, config, args); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
