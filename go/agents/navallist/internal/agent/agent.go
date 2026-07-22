package agent

import (
	"context"
	_ "embed"

	"navallist/internal/data"

	adkagent "google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"
)

//go:embed instruction.md
var instruction string

// NewChecklistAgent initializes a new ADK agent for boat checklists.
func NewChecklistAgent(ctx context.Context, store data.Store, modelName, apikey string) (adkagent.Agent, error) {
	// Model Setup
	model, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{
		APIKey: apikey,
	})
	if err != nil {
		return nil, err
	}

	// Initialize Tools Handler
	handler := &ChecklistTool{Store: store}

	// Define Tools
	updateTool, err := functiontool.New(
		functiontool.Config{
			Name:        "update_checklist_items",
			Description: "Updates the status and location of one or more items on the boat's checklist. You can only assign items to names exactly as they appear in 'get_crew_list'. Assigning to anyone else is forbidden.",
		},
		handler.UpdateItems,
	)
	if err != nil {
		return nil, err
	}

	crewTool, err := functiontool.New(
		functiontool.Config{
			Name:        "get_crew_list",
			Description: "Returns a list of all crew members currently participating in this trip. Use this to find the correct names for assignments.",
		},
		handler.GetCrewList,
	)
	if err != nil {
		return nil, err
	}

	statusTool, err := functiontool.New(
		functiontool.Config{
			Name:        "get_checklist_status",
			Description: "Returns the current state of the boat checklist, including who is assigned to each item and what has been completed.",
		},
		handler.GetChecklistStatus,
	)
	if err != nil {
		return nil, err
	}

	metaTool, err := functiontool.New(
		functiontool.Config{
			Name:        "update_trip_details",
			Description: "Updates the boat name or captain name for the trip.",
		},
		handler.UpdateMetadata,
	)
	if err != nil {
		return nil, err
	}

	return llmagent.New(llmagent.Config{
		Name:        "navallist_agent",
		Description: "Manages boat checklists",
		Instruction: instruction,
		Model:       model,
		Tools: []tool.Tool{
			updateTool,
			crewTool,
			statusTool,
			metaTool,
		},
	})
}
