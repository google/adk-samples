package agent

import (
	"context"
	"fmt"
	"strings"

	"navallist/internal/data"

	"github.com/charmbracelet/log"
	"google.golang.org/adk/v2/agent"
)

// UpdateChecklistArgs defines the input for the checklist item tool.
type UpdateChecklistArgs struct {
	ItemName        string `json:"item_name" jsonschema:"The name of the checklist item to update (e.g., 'Engine Oil', 'Flares')"`
	IsChecked       bool   `json:"is_checked" jsonschema:"Whether the item is checked off (true) or unchecked (false). IMPORTANT: Set to false if you are ONLY assigning the item and not confirming it is done."`
	Location        string `json:"location,omitempty" jsonschema:"Where the item was found or stored (optional)"`
	PhotoArtifactID string `json:"photo_artifact_id,omitempty" jsonschema:"Optional: The ID of an associated photo artifact (if known by the agent)"`
	AssignedToName  string `json:"assigned_to_name,omitempty" jsonschema:"The name of the crew member to assign this item to (e.g. 'Sarah', 'Engineer')"`
}

// UpdateTripArgs defines the input for the trip metadata tool.
type UpdateTripArgs struct {
	BoatName    string `json:"boat_name,omitempty" jsonschema:"The name of the boat"`
	CaptainName string `json:"captain_name,omitempty" jsonschema:"The name of the captain"`
}

// ToolResult defines a generic output for ADK tools.
type ToolResult struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

// ChecklistTool holds dependencies for the agent's checklist tools.
type ChecklistTool struct {
	Store data.Store
}

// resolveTripID looks up the database PK from the ADK Session ID.
func (t *ChecklistTool) resolveTripID(ctx context.Context, adkSessionID string) (string, error) {
	id, err := t.Store.GetTripIDBySessionID(ctx, adkSessionID)
	if err != nil {
		return "", fmt.Errorf("trip not found for session '%s' (ensure you are logged in or trip is created): %w", adkSessionID, err)
	}
	return id, nil
}

// GetCrewList returns the names of all people currently participating in the trip.
func (t *ChecklistTool) GetCrewList(ctx agent.Context, _ struct{}) (interface{}, error) {
	adkID := ctx.SessionID()
	tripID, err := t.resolveTripID(ctx, adkID)
	if err != nil {
		return nil, err
	}

	crew, err := t.Store.GetActiveCrewNames(ctx, tripID)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"crew": crew,
	}, nil
}

// GetChecklistStatus returns the current state of all items in the checklist.
func (t *ChecklistTool) GetChecklistStatus(ctx agent.Context, _ struct{}) (interface{}, error) {
	adkID := ctx.SessionID()
	tripID, err := t.resolveTripID(ctx, adkID)
	if err != nil {
		return nil, err
	}

	items, err := t.Store.GetTripReport(ctx, tripID)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"items": items,
	}, nil
}

// UpdateItemsArgs defines the input for the plural update tool.
type UpdateItemsArgs struct {
	Updates []UpdateChecklistArgs `json:"updates" jsonschema:"The list of individual item updates to apply."`
}

// UpdateItems allows the agent to update one or more items in a single tool call.
func (t *ChecklistTool) UpdateItems(ctx agent.Context, args UpdateItemsArgs) (ToolResult, error) {
	log.Info("Tool UpdateItems called", "count", len(args.Updates), "session_id", ctx.SessionID())

	var successes []string
	var errors []string
	var warnings []string

	for _, update := range args.Updates {
		// reuse the logic from updateItemInternal
		res, err := t.updateItemInternal(ctx, update)
		if err != nil {
			log.Error("Update failed for item", "item", update.ItemName, "error", err)
			errors = append(errors, fmt.Sprintf("%s: %v", update.ItemName, err))
		} else {
			successes = append(successes, update.ItemName)
			if res.Status == "warning" {
				warnings = append(warnings, res.Message)
			}
		}
	}

	msg := fmt.Sprintf("Updated %d items: %s.", len(successes), strings.Join(successes, ", "))
	status := "success"

	if len(errors) > 0 {
		msg += fmt.Sprintf(" Failed items: %s.", strings.Join(errors, ", "))
		status = "partial_success"
	}

	if len(warnings) > 0 {
		msg += fmt.Sprintf(" Warnings: %s.", strings.Join(warnings, ", "))
		if status == "success" {
			status = "warning"
		}
	}

	return ToolResult{Status: status, Message: msg}, nil
}

// updateItemInternal is the internal helper for updating checklist items.
func (t *ChecklistTool) updateItemInternal(ctx agent.Context, args UpdateChecklistArgs) (result ToolResult, err error) {
	log.Info("Internal updateItemInternal called", "args", args, "session_id", ctx.SessionID())

	adkID := ctx.SessionID()
	if adkID == "" {
		return ToolResult{Status: "error"}, fmt.Errorf("session_id missing from context")
	}

	// Resolve the real DB ID
	tripID, err := t.resolveTripID(ctx, adkID)
	if err != nil {
		return ToolResult{Status: "error"}, err
	}

	// --- Check for Artifacts (Photos) in Context ---
	photoID := args.PhotoArtifactID
	if photoID != "" && !strings.Contains(photoID, "?v=") {
		photoID = ""
	}

	updated, matchFound, err := t.Store.UpdateItemWithAssignment(ctx, tripID, args.ItemName, args.IsChecked, args.Location, photoID, ctx.UserID(), args.AssignedToName)
	if err != nil {
		return ToolResult{Status: "error"}, fmt.Errorf("failed to update: %w", err)
	}

	loc := ""
	if updated.LocationText != nil {
		loc = *updated.LocationText
	}
	msg := fmt.Sprintf("Updated %s: Checked=%v, Location=%s", updated.Name, updated.IsChecked, loc)
	status := "success"

	if updated.AssignedToName != nil {
		msg += fmt.Sprintf(", Assigned To=%s", *updated.AssignedToName)
		if !matchFound {
			msg += " (Warning: Name not in crew list)"
			status = "warning"
		}
	}
	if photoID != "" {
		msg += " (Photo attached)"
	}
	return ToolResult{Status: status, Message: msg}, nil
}

// UpdateMetadata is the function called by the agent to update trip details.
func (t *ChecklistTool) UpdateMetadata(ctx agent.Context, args UpdateTripArgs) (ToolResult, error) {
	log.Info("Tool UpdateMetadata called", "args", args, "session_id", ctx.SessionID())

	adkID := ctx.SessionID()
	if adkID == "" {
		return ToolResult{Status: "error"}, fmt.Errorf("session_id missing from context")
	}

	var bName, cName *string
	if args.BoatName != "" {
		bName = &args.BoatName
	}
	if args.CaptainName != "" {
		cName = &args.CaptainName
	}

	updated, err := t.Store.UpdateTripMetadata(ctx, adkID, bName, cName)
	if err != nil {
		return ToolResult{Status: "error"}, fmt.Errorf("failed to update metadata: %w", err)
	}

	b := ""
	if updated.BoatName != nil {
		b = *updated.BoatName
	}
	c := ""
	if updated.CaptainName != nil {
		c = *updated.CaptainName
	}

	msg := fmt.Sprintf("Updated Trip: Boat='%s', Captain='%s'", b, c)
	log.Info("UpdateMetadata success", "boat_name", b)
	return ToolResult{Status: "success", Message: msg}, nil
}
