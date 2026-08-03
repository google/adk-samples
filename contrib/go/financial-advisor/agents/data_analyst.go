package agents

import (
	"context"
	_ "embed"
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
)

//go:embed prompts/data_analyst.md
var dataAnalystInstruction string

func NewDataAnalyst(ctx context.Context) agent.Agent {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-lite", nil)
	if err != nil {
		log.Fatalf("Failed to create model for data analyst: %v", err)
	}
	dataAnalyst, err := llmagent.New(llmagent.Config{
		Name:        "data_analyst_agent",
		Model:       model,
		Description: "Analyze financial data and provide insights.",
		Instruction: dataAnalystInstruction,
		Tools:       []tool.Tool{geminitool.GoogleSearch{}},
		OutputKey:   "market_data_analysis_output",
	})
	if err != nil {
		log.Fatalf("Failed to create data analyst agent: %v", err)
	}
	return dataAnalyst
}
