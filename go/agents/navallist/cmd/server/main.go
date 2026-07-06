package main

import (
	"context"
	"fmt"
	"navallist/internal/agent"
	"navallist/internal/config"
	"navallist/internal/data"
	"navallist/internal/realtime"
	"navallist/internal/server"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/charmbracelet/log"
	"github.com/felixge/httpsnoop"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
	"google.golang.org/adk/v2/session"
)

const (
	ThresholdWarn       = time.Second * 10
	ThresholdUrgentWarn = time.Second * 30
	ShutdownTimeout     = 5 * time.Second
)

func main() {
	log.SetPrefix("backend")

	ctx := context.Background()
	if err := run(ctx); err != nil {
		log.Error("Application error", "err", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	// 1. Load Configuration
	cfg, err := config.Load(os.LookupEnv)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	log.Info("Starting backend", "port", cfg.Port)
	log.Info("config", "Env", cfg.Env)
	log.Info("config", "Port", cfg.Port)
	log.Info("config", "DatabaseDSN", maskSecret(cfg.DB.DSN()))
	log.Info("config", "SiteURL", cfg.SiteURL)
	log.Info("config", "NAVALLIST_DB_USER", cfg.DB.User)
	log.Info("config", "NAVALLIST_DB_PASS", maskSecret(cfg.DB.Password))
	log.Info("config", "NAVALLIST_DB_HOST", cfg.DB.Host)
	log.Info("config", "NAVALLIST_DB_PORT", cfg.DB.Port)
	log.Info("config", "NAVALLIST_DB_NAME", cfg.DB.Name)
	log.Info("config", "NAVALLIST_DB_SSLMODE", cfg.DB.SSLMode)
	log.Info("config", "Frontend", cfg.FrontendDir)
	log.Info("config", "ModelName", cfg.ModelName)

	// 1b. Setup Frontend Filesystem
	if cfg.FrontendDir == "" {
		cfg.FrontendDir = "web"
	}
	log.Info("Serving frontend from disk", "path", cfg.FrontendDir)
	frontendFS := http.Dir(cfg.FrontendDir)

	// 2. Database Connection
	db, err := sqlx.Connect("pgx", cfg.DB.DSN())
	if err != nil {
		obsc := maskSecret(cfg.DB.DSN())
		return fmt.Errorf("unable to connect to database dsn(%s): %w,", obsc, err)
	}
	defer func() {
		if err := db.Close(); err != nil {
			log.Error("failed to close database connection", "err", err)
		}
	}()

	// 3. Initialize Services
	store := data.NewSQLStore(db)

	// 3c. Initialize Blob Storage (Disk only for sample)
	storageDir := "data/artifacts"
	storage := data.NewDiskStorage(storageDir)
	log.Info("Using Disk Storage", "dir", storageDir)
	// --- Embedded Agent Initialization ---
	// 1. Initialize Agent
	checklistAgent, err := agent.NewChecklistAgent(ctx, store, cfg.ModelName, cfg.GoogleAPIKey)
	if err != nil {
		return fmt.Errorf("failed to create checklist agent: %w", err)
	}

	// 2. Services for ADK
	// Use InMemoryService for sessions (transient history)
	sessionService := session.InMemoryService()

	// 3. Create Local Client
	agentClient, err := agent.NewLocalClient(checklistAgent, sessionService)
	if err != nil {
		return fmt.Errorf("failed to create local agent client: %w", err)
	}
	log.Info("Embedded agent initialized")
	// -------------------------------------

	// 4. Setup Server
	srv := server.NewServer(store, cfg, agentClient, storage, frontendFS)

	// 4b. Setup Realtime
	rtService, err := realtime.NewService(store)
	if err != nil {
		return fmt.Errorf("failed to init realtime service: %w", err)
	}
	go rtService.ListenToDB(ctx, cfg.DB.DSN())

	// Mount WebSocket handler using the new method
	srv.MountWebSocket(rtService)

	// Wrap the mux with middleware
	handler := recoveryMiddleware(loggingMiddleware(srv.Mux))

	httpServer := &http.Server{
		Addr:    ":" + cfg.Port,
		Handler: handler,
	}

	// 5. Start Server with Graceful Shutdown
	// Create a channel to listen for interrupt signals
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	// Run server in a goroutine
	errChan := make(chan error, 1)
	go func() {
		log.Info("Server started", "addr", httpServer.Addr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errChan <- fmt.Errorf("server failed: %w", err)
		}
	}()

	// Wait for signal or error
	select {
	case err := <-errChan:
		return err
	case <-stop:
		log.Info("Shutting down server...")
	case <-ctx.Done():
		log.Info("Context canceled, shutting down...")
	}

	// Create a deadline for shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), ShutdownTimeout)
	defer cancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("server shutdown failed: %w", err)
	}

	log.Info("Server exited properly")
	return nil
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
		m := httpsnoop.CaptureMetrics(next, w, r)

		var logFunc func(msg interface{}, keyvals ...interface{})
		if m.Code >= 500 {
			logFunc = log.Error
		} else if m.Code >= 400 {
			logFunc = log.Warn
		} else if m.Duration > ThresholdUrgentWarn {
			logFunc = log.Warn
		} else {
			logFunc = log.Info
		}

		logFunc("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", m.Code,
			"duration", m.Duration,
			"ip", r.RemoteAddr,
		)
	})
}

// maskSecret replaces most of the string with asterisks for logging.
func maskSecret(s string) string {
	if len(s) <= 4 {
		return "****"
	}
	return "****" + s[len(s)-4:]
}
