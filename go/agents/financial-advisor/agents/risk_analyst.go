package agents

import (
	"context"
	_ "embed"
	"log"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
)

//go:embed prompts/risk_analyst.md
var riskAnalystInstruction string

func NewRiskAnalyst(ctx context.Context) agent.Agent {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-lite", nil)
	if err != nil {
		log.Fatalf("Failed to create model for risk analyst: %v", err)
	}
	riskAnalyst, err := llmagent.New(llmagent.Config{
		Name:        "risk_analyst_agent",
		Model:       model,
		Description: "Analyze risk and provide insights.",
		Instruction: riskAnalystInstruction,
		OutputKey:   "final_risk_assessment_output",
	})
	if err != nil {
		log.Fatalf("Failed to create risk analyst agent: %v", err)
	}
	return riskAnalyst
}
