package main

import (
	"context"
	_ "embed"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"

	"github.com/google/adk-samples/go/agents/financial-advisor/agents"
)

//go:embed agents/prompts/financial_coordinator.md
var instruction string

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	dataAnalyst := agents.NewDataAnalyst(ctx)
	tradingAnalyst := agents.NewTradingAnalyst(ctx)
	executionAnalyst := agents.NewExecutionAnalyst(ctx)
	riskAnalyst := agents.NewRiskAnalyst(ctx)

	financialCoordinator, err := llmagent.New(llmagent.Config{
		Name:  "financial_coordinator",
		Model: model,
		Description: "guide users through a structured process to receive financial " +
			"advice by orchestrating a series of expert subagents. help them " +
			"analyze a market ticker, develop trading strategies, define " +
			"execution plans, and evaluate the overall risk.",
		Instruction: instruction,
		OutputKey:   "financial_coordinator_output",
		Tools: []tool.Tool{
			agenttool.New(dataAnalyst, nil),
			agenttool.New(tradingAnalyst, nil),
			agenttool.New(executionAnalyst, nil),
			agenttool.New(riskAnalyst, nil),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(financialCoordinator),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
