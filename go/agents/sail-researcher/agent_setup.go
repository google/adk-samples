package main

import (
	"context"
	_ "embed"
	"fmt"

	"github.com/tpryan/navalplan/services/researcher/config"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/genai"
)

//go:embed prompts/search_specialist.md
var searchSpecialistPrompt string

//go:embed prompts/stop_agent.md
var stopAgentPrompt string

//go:embed prompts/voyage_agent.md
var voyageAgentPrompt string

//go:embed prompts/discovery_agent.md
var discoveryAgentPrompt string

const maxOutputTokens = 65536

type agentConfig struct {
	name        string
	description string
	instruction string
	tools       []tool.Tool
	temperature float32
}

func constructAgent(ctx context.Context, cfg *config.Config, monitor *ToolMonitor, acfg *agentConfig) (agent.Agent, error) {
	genConfig := &genai.GenerateContentConfig{
		MaxOutputTokens: maxOutputTokens,
		Temperature:     genai.Ptr[float32](acfg.temperature),
	}

	m, err := gemini.NewModel(ctx, cfg.ModelName, &genai.ClientConfig{
		APIKey: cfg.GeminiAPIKey,
	})
	if err != nil {
		return nil, err
	}

	return llmagent.New(llmagent.Config{
		Name:                  acfg.name,
		Model:                 m,
		Description:           acfg.description,
		Instruction:           acfg.instruction,
		Tools:                 acfg.tools,
		BeforeToolCallbacks:   []llmagent.BeforeToolCallback{monitor.OnBeforeTool},
		AfterToolCallbacks:    []llmagent.AfterToolCallback{monitor.OnAfterTool},
		GenerateContentConfig: genConfig,
	})
}

func NewVoyageAgent(ctx context.Context, cfg *config.Config, monitor *ToolMonitor) (agent.Agent, error) {
	return constructAgent(ctx, cfg, monitor, &agentConfig{
		name:        "guide_agent",
		description: "A Local Knowledge Expert and Sailing Guide.",
		instruction: voyageAgentPrompt,
		tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
		temperature: 0.4,
	})
}

func NewStopAgent(ctx context.Context, cfg *config.Config, monitor *ToolMonitor, researcherTools []tool.Tool) (agent.Agent, error) {
	searchAgent, err := constructAgent(ctx, cfg, monitor, &agentConfig{
		name:        "search_specialist",
		description: "Finds information on the web (facilities, reviews).",
		instruction: searchSpecialistPrompt,
		tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
		temperature: 0.4,
	})
	if err != nil {
		return nil, fmt.Errorf("creating search agent: %w", err)
	}

	allTools := append(researcherTools, agenttool.New(searchAgent, nil))

	return constructAgent(ctx, cfg, monitor, &agentConfig{
		name:        "researcher_agent",
		description: "A Virtual Harbourmaster that researches sailing destinations.",
		instruction: stopAgentPrompt,
		tools:       allTools,
		temperature: 0.4,
	})
}

func NewDiscoveryAgent(ctx context.Context, cfg *config.Config, monitor *ToolMonitor) (agent.Agent, error) {
	return constructAgent(ctx, cfg, monitor, &agentConfig{
		name:        "discovery_agent",
		description: "The Commodore - Global Seasonal Discovery Expert.",
		instruction: discoveryAgentPrompt,
		tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
		temperature: 0.2,
	})
}