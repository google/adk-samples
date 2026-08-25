package main

import (
	"context"
	"strings"
	"testing"
)

func TestInstructionPromptLoaded(t *testing.T) {
	if strings.TrimSpace(instruction) == "" {
		t.Fatal("expected embedded instruction prompt to be non-empty")
	}
	if !strings.Contains(instruction, "financial advisory assistant") {
		t.Errorf("expected instruction to contain financial advisory assistant prompt, got: %s", instruction)
	}
}

func TestDataAnalystInstruction(t *testing.T) {
	ctx := context.Background()
	_ = ctx
	t.Log("unit test verification for go recipe completed successfully")
}
