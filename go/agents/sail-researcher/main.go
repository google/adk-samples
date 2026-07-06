// Package main is the entry point for the researcher service, orchestrating multiple AI agents
// (Researcher, Guide, Discovery) to assist with sailing voyage planning.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/tpryan/navalplan/services/researcher/config"
	"github.com/tpryan/navalplan/services/researcher/logging"
	"github.com/tpryan/navalplan/services/researcher/tools"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

type Provider interface {
	Close() error
}

func main() {
	// Load .env file (try current dir, then project root)
	godotenv.Load(".env")

	cfg, err := config.New(os.Getenv)
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	logging.InitLogging(cfg.Env)

	slog.Info("config", "modelName", cfg.ModelName)
	slog.Info("config", "port", cfg.Port)
	if len(cfg.MapsAPIKey) > 5 {
		slog.Info("config", "MapsAPIKey", cfg.MapsAPIKey[:5]+"...")
	}

	ctx := context.Background()
	srv := &Server{
		config: cfg,
	}
	defer srv.Close()

	if err := srv.run(ctx); err != nil {
		slog.Error("Application error", "error", err)
		os.Exit(1)
	}
}

type Server struct {
	config *config.Config
	providers []Provider
}

func (s *Server) Close() {
	for _, p := range s.providers {
		if err := p.Close(); err != nil {
			slog.Error("Failed to close provider", "error", err)
		}
	}
}

func (s *Server) run(ctx context.Context) error {
	researcherTools, err := s.setupTools(ctx)
	if err != nil {
		return fmt.Errorf("setting up tools: %w", err)
	}

	toolMonitor := NewToolMonitor()

	voyageAgent, err := NewVoyageAgent(ctx, s.config, toolMonitor)
	if err != nil {
		return fmt.Errorf("creating guide agent: %w", err)
	}

	stopAgent, err := NewStopAgent(ctx, s.config, toolMonitor, researcherTools)
	if err != nil {
		return fmt.Errorf("creating researcher agent: %w", err)
	}

	discoveryAgent, err := NewDiscoveryAgent(ctx, s.config, toolMonitor)
	if err != nil {
		return fmt.Errorf("creating discovery agent: %w", err)
	}

	loader, err := agent.NewMultiLoader(stopAgent, voyageAgent, discoveryAgent)
	if err != nil {
		return fmt.Errorf("creating multi loader: %w", err)
	}

	// Create the ADK HTTP Handler
	adkHandler, err := adkrest.NewServer(adkrest.ServerConfig{
		AgentLoader:     loader,
		SessionService:  session.InMemoryService(),
		SSEWriteTimeout: 120 * time.Second,
	})
	if err != nil {
		return fmt.Errorf("creating REST server: %w", err)
	}

	// Start Custom Server
	mux := http.NewServeMux()

	// Mount ADK under /api/
	mux.Handle("/api/", http.StripPrefix("/api", adkHandler))

	slog.Info("Starting custom server", "port", s.config.Port)
	// Apply Trace Middleware then Logging Middleware
	return http.ListenAndServe(":"+s.config.Port, TraceMiddleware(s.config.Project, LoggingMiddleware(mux)))
}

func (s *Server) setupTools(ctx context.Context) ([]tool.Tool, error) {
	weatherTool, wp, err := tools.NewWeatherTool()
	if err != nil {
		return nil, err
	}
	s.providers = append(s.providers, wp)

	tideTool, tp, err := tools.NewTideTool()
	if err != nil {
		return nil, err
	}
	s.providers = append(s.providers, tp)

	sunriseTool, sp, err := tools.NewSunriseTool(s.config.MapsAPIKey)
	if err != nil {
		return nil, err
	}
	s.providers = append(s.providers, sp)

	placesTool, pp, err := tools.NewPlacesTool(ctx, s.config.MapsAPIKey)
	if err != nil {
		return nil, err
	}
	s.providers = append(s.providers, pp)

	return []tool.Tool{weatherTool, tideTool, sunriseTool, placesTool}, nil
}
