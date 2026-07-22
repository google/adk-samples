package main

import (
	"log/slog"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// ToolMonitor tracks tool execution times.
type ToolMonitor struct {
	mu      sync.Mutex
	timings map[string]time.Time
}

// NewToolMonitor creates a new ToolMonitor.
func NewToolMonitor() *ToolMonitor {
	return &ToolMonitor{
		timings: make(map[string]time.Time),
	}
}

// OnBeforeTool records the start time of a tool execution.
func (tm *ToolMonitor) OnBeforeTool(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.timings[ctx.FunctionCallID()] = time.Now()
	return nil, nil
}

// OnAfterTool records the end time and logs the duration of a tool execution.
func (tm *ToolMonitor) OnAfterTool(ctx agent.Context, t tool.Tool, args map[string]any, result map[string]any, err error) (map[string]any, error) {
	tm.mu.Lock()
	startTime, ok := tm.timings[ctx.FunctionCallID()]
	if ok {
		delete(tm.timings, ctx.FunctionCallID())
	}
	tm.mu.Unlock()

	if ok {
		duration := time.Since(startTime)
		slog.Debug("tool execution", "tool", t.Name(), "duration", duration.String())
	}
	return result, nil
}
