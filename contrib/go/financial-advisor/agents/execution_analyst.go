package agents

import (
	"context"
	_ "embed"
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
)

//go:embed prompts/execution_analyst.md
var executionAnalystInstruction string

func NewExecutionAnalyst(ctx context.Context) agent.Agent {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-lite", nil)
	if err != nil {
		log.Fatalf("Failed to create model for execution analyst: %v", err)
	}
	executionAnalyst, err := llmagent.New(llmagent.Config{
		Name:        "execution_analyst_agent",
		Model:       model,
		Description: "Analyze execution strategies and provide insights.",
		Instruction: executionAnalystInstruction,
		OutputKey:   "execution_plan_output",
	})
	if err != nil {
		log.Fatalf("Failed to create execution analyst agent: %v", err)
	}
	return executionAnalyst
}
