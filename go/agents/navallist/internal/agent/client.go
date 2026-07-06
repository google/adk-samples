package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"navallist/internal/data"

	adkagent "google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// LocalAgentClient implements data.AgentClient by calling the agent directly via ADK runner.
type LocalAgentClient struct {
	Runner   *runner.Runner
	Sessions session.Service
	Agent    adkagent.Agent
}

// NewLocalClient creates a new LocalAgentClient.
func NewLocalClient(a adkagent.Agent, s session.Service) (*LocalAgentClient, error) {
	r, err := runner.New(runner.Config{
		AppName:        a.Name(),
		Agent:          a,
		SessionService: s,
	})
	if err != nil {
		return nil, err
	}
	return &LocalAgentClient{
		Runner:   r,
		Sessions: s,
		Agent:    a,
	}, nil
}

// CreateSession ensures the session exists in the session service.
func (c *LocalAgentClient) CreateSession(ctx context.Context, appName, userID, sessionID string) error {
	// Check name match
	if c.Agent.Name() != appName {
		// Log warning or error? For now just proceed as we only have one agent.
	}

	_, err := c.Sessions.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	return err
}

// GetSession retrieves the session state.
func (c *LocalAgentClient) GetSession(ctx context.Context, appName, userID, sessionID string) (map[string]interface{}, error) {
	resp, err := c.Sessions.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		return nil, err
	}

	// We need to return a map that represents the session or its state.
	// The ADK REST API returns the session object.
	// We'll marshal the session to JSON and back to map.
	b, err := json.Marshal(resp.Session)
	if err != nil {
		return nil, err
	}
	var result map[string]interface{}
	err = json.Unmarshal(b, &result)
	return result, err
}

// RunInteraction sends an interaction payload to the agent.
func (c *LocalAgentClient) RunInteraction(ctx context.Context, payload interface{}) (interface{}, error) {
	// Marshall payload to JSON then Unmarshal to a struct that matches /run body
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	fmt.Printf("[DEBUG] Raw Interaction Payload: %s\n", string(b))

	// Define a local struct that matches the incoming JSON exactly for debugging
	type jsonPart struct {
		Text       string `json:"text"`
		InlineData *struct {
			MIMEType string `json:"mime_type"`
			Data     string `json:"data"`
		} `json:"inline_data"`
	}
	type jsonContent struct {
		Role  string     `json:"role"`
		Parts []jsonPart `json:"parts"`
	}

	var req struct {
		AppName    string       `json:"app_name"`
		UserID     string       `json:"user_id"`
		SessionID  string       `json:"session_id"`
		NewMessage *jsonContent `json:"new_message"`
	}
	if err := json.Unmarshal(b, &req); err != nil {
		return nil, fmt.Errorf("invalid payload structure: %w", err)
	}

	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return nil, fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}

	// Convert our debug struct back to the real genai.Content
	var realNewMessage *genai.Content
	if req.NewMessage != nil {
		realNewMessage = &genai.Content{
			Role: req.NewMessage.Role,
		}
		for i, p := range req.NewMessage.Parts {
			if p.Text != "" {
				fmt.Printf("[DEBUG] Part %d: Text: %q\n", i, p.Text)
				realNewMessage.Parts = append(realNewMessage.Parts, &genai.Part{Text: p.Text})
			}
			if p.InlineData != nil {
				fmt.Printf("[DEBUG] Part %d: InlineData MIMEType: %q, Data Length: %d\n", i, p.InlineData.MIMEType, len(p.InlineData.Data))
				data, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
				if err != nil {
					return nil, fmt.Errorf("failed to decode base64 data for part %d: %w", i, err)
				}
				realNewMessage.Parts = append(realNewMessage.Parts, &genai.Part{
					InlineData: &genai.Blob{
						MIMEType: p.InlineData.MIMEType,
						Data:     data,
					},
				})
			}
		}
	}

	// Run
	var events []*session.Event
	for event, err := range c.Runner.Run(ctx, req.UserID, req.SessionID, realNewMessage, adkagent.RunConfig{}) {
		if err != nil {
			if strings.Contains(err.Error(), "session") && strings.Contains(err.Error(), "not found") {
				return nil, data.ErrNotFound
			}
			return nil, err
		}
		events = append(events, event)
	}

	return events, nil
}
