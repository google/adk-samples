package agents

import (
	"context"
	_ "embed"
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
)

//go:embed prompts/trading_analyst.md
var tradingAnalystInstruction string

func NewTradingAnalyst(ctx context.Context) agent.Agent {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-lite", nil)
	if err != nil {
		log.Fatalf("Failed to create model for trading analyst: %v", err)
	}
	tradingAnalyst, err := llmagent.New(llmagent.Config{
		Name:        "trading_analyst_agent",
		Model:       model,
		Description: "Analyze trading strategies and provide insights.",
		Instruction: tradingAnalystInstruction,
		OutputKey:   "proposed_trading_strategies_output",
	})
	if err != nil {
		log.Fatalf("Failed to create trading analyst agent: %v", err)
	}
	return tradingAnalyst
}
