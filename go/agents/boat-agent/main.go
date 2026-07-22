package main

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/log"
	"github.com/joho/godotenv"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/genai"
)

// Global styling for logging allows us to flag long running processes
var (
	timeWarn            = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFFF00"))
	timeUrgentWarn      = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF0000"))
	thresholdWarn       = time.Second * 10
	thresholdUrgentWarn = time.Second * 30
)

//go:embed instruction.md
var instruction string

// Config holds the application configuration.
type Config struct {
	Port         string
	ModelName    string
	GoogleAPIKey string
}

// LoadConfig loads configuration from environment variables.
func LoadConfig() Config {

	if err := godotenv.Load(); err != nil {
		log.Debug("No .env file found in current directory")
	}

	cfg := Config{
		Port:         os.Getenv("PORT"),
		ModelName:    os.Getenv("GEMINI_MODEL_NAME"),
		GoogleAPIKey: os.Getenv("GOOGLE_API_KEY"),
	}

	if cfg.Port == "" {
		cfg.Port = "8081" // Default fallback
	}

	if cfg.ModelName == "" {
		cfg.ModelName = "gemini-flash-latest"
	}

	return cfg
}

func main() {
	// Configure charmbracelet/log
	log.SetOutput(os.Stdout)
	log.SetLevel(log.DebugLevel)
	log.SetPrefix("boat-agent")

	cfg := LoadConfig()
	log.Info("Starting boat-agent", "config", cfg.ModelName, "port", cfg.Port)

	ctx := context.Background()

	// 1. Initialize Gemini Model
	model, err := gemini.NewModel(ctx, cfg.ModelName, &genai.ClientConfig{
		APIKey: cfg.GoogleAPIKey,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// 2. Create Agent
	boatAgent, err := llmagent.New(llmagent.Config{
		Name:        "boat_agent",
		Model:       model,
		Description: "Agent designed to gather information about sailboats.",
		Instruction: instruction,
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})

	if err != nil {
		log.Fatalf("Failed to create boat agent: %v", err)
	}

	// 3. Create Loader
	loader := agent.NewSingleLoader(boatAgent)

	// 4. Create ADK HTTP Handler
	adkHandler, err := adkrest.NewServer(adkrest.ServerConfig{
		AgentLoader:     loader,
		SessionService:  session.InMemoryService(),
		SSEWriteTimeout: 120 * time.Second,
	})
	if err != nil {
		log.Fatalf("Failed to create REST handler: %v", err)
	}

	// 6. Setup Router
	mux := http.NewServeMux()
	mux.Handle("/api/", http.StripPrefix("/api", adkHandler))

	// 7. Setup Server with Middleware
	srv := &http.Server{
		Addr:    ":" + cfg.Port,
		Handler: recoveryMiddleware(loggingMiddleware(mux)),
	}

	// 8. Start Server (Graceful Shutdown)
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("Server failed: %v", err)
		}
	}()
	log.Info("Server started", "addr", srv.Addr)

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("Shutting down server...")

	// Create a deadline to wait for.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Info("Server exited")
}

type responseWriter struct {
	http.ResponseWriter
	statusCode  int
	wroteHeader bool
}

func (rw *responseWriter) WriteHeader(code int) {
	if rw.wroteHeader {
		return
	}
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
	rw.wroteHeader = true
}

func recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Error("Panic recovered", "error", err, "path", r.URL.Path)
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(ww, r)

		timesince := time.Since(start)
		str := timesince.String()

		switch {
		case timesince > thresholdUrgentWarn:
			str = timeUrgentWarn.Render(str)
		case timesince > thresholdWarn:
			str = timeWarn.Render(str)
		}

		log.Info(fmt.Sprintf("%s %s %s %d %s", r.Method, r.URL.Path, r.RemoteAddr, ww.statusCode, str))
	})
}
