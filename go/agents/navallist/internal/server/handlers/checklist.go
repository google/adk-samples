package handlers

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"navallist/internal/agent"
	"navallist/internal/data"

	"github.com/charmbracelet/log"
)

type ChecklistHandler struct {
	Client    *agent.LocalAgentClient
	Store     data.Store
	AgentName string
}

func NewChecklistHandler(client *agent.LocalAgentClient, store data.Store) *ChecklistHandler {
	return &ChecklistHandler{
		Client:    client,
		Store:     store,
		AgentName: "navallist_agent",
	}
}

// CreateSession - POST /api/agent/sessions
func (h *ChecklistHandler) CreateSession(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SessionID   string `json:"session_id"`
		UserID      string `json:"user_id"`
		DisplayName string `json:"display_name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.SessionID == "" || req.UserID == "" {
		http.Error(w, "session_id and user_id are required", http.StatusBadRequest)
		return
	}

	// 1. Sync with Agent
	err := h.Client.CreateSession(r.Context(), h.AgentName, req.UserID, req.SessionID)
	if err != nil {
		log.Error("Failed to create agent session", "err", err, "session", req.SessionID, "user", req.UserID)

		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// 2. Sync with local Trip Crew (for GetCrewList tool)
	tripID, err := h.Store.GetTripIDBySessionID(r.Context(), req.SessionID)
	if err == nil && tripID != "" {
		_ = h.Store.AddTripCrew(r.Context(), tripID, req.UserID, req.DisplayName)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// GetSession - GET /api/agent/sessions/{sessionID}?userId=...
func (h *ChecklistHandler) GetSession(w http.ResponseWriter, r *http.Request) {
	sessionID := r.PathValue("sessionID")
	userID := r.URL.Query().Get("userId")

	if sessionID == "" || userID == "" {
		http.Error(w, "Missing session or user id", http.StatusBadRequest)
		return
	}

	data, err := h.Client.GetSession(r.Context(), h.AgentName, userID, sessionID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if data == nil {
		http.Error(w, "Session not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(data)
}

// RunInteraction - POST /api/agent/run
func (h *ChecklistHandler) RunInteraction(w http.ResponseWriter, r *http.Request) {
	var payload map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, "Invalid json", http.StatusBadRequest)
		return
	}

	// For logging and self-healing
	sessionID, _ := payload["session_id"].(string)
	if sessionID == "" {
		sessionID, _ = payload["sessionId"].(string)
	}
	userID, _ := payload["user_id"].(string)
	if userID == "" {
		userID, _ = payload["userId"].(string)
	}

	resp, err := h.Client.RunInteraction(r.Context(), payload)
	if errors.Is(err, data.ErrNotFound) {
		// Self-healing: Try to create session and retry
		if sessionID != "" && userID != "" {
			// Try to create the session
			if createErr := h.Client.CreateSession(r.Context(), h.AgentName, userID, sessionID); createErr == nil {
				// Retry Run if creation succeeded
				resp, err = h.Client.RunInteraction(r.Context(), payload)
			} else {
				log.Error("Self-healing session creation failed", "err", createErr, "session", sessionID)
			}
		}
	}

	if err != nil {
		log.Error("Agent interaction failed", "err", err, "session", sessionID)

		w.Header().Set("Content-Type", "application/json")

		status := http.StatusInternalServerError
		code := "internal_error"

		if errors.Is(err, data.ErrNotFound) {
			status = http.StatusNotFound
			code = "session_not_found"
		} else if strings.Contains(strings.ToLower(err.Error()), "overloaded") ||
			strings.Contains(strings.ToLower(err.Error()), "rate limit") {
			status = http.StatusTooManyRequests
			code = "overloaded"
		}

		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"status": "error",
			"code":   code,
			"error":  err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
