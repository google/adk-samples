package config

import (
	"fmt"
)

const (
	DefaultModelName = "gemini-flash-latest"
	DefaultPort      = "8081"
	DefaultEnv       = "development"
)

type Config struct {
	Env          string
	Project      string
	ModelName    string
	GeminiAPIKey string
	MapsAPIKey   string
	Port         string
}

func New(getEnv func(string) string) (*Config, error) {
	mapsKey := getEnv("GOOGLE_MAPS_KEY")
	if mapsKey == "" {
		return nil, fmt.Errorf("GOOGLE_MAPS_KEY is not set")
	}

	modelName := getEnv("MODEL")
	if modelName == "" {
		modelName = DefaultModelName
	}

	geminiKey := getEnv("GEMINI_API_KEY")
	if geminiKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is not set")
	}

	port := getEnv("PORT")
	if port == "" {
		port = DefaultPort
	}

	env := getEnv("ENV")
	if env == "" {
		env = DefaultEnv
	}

	project := getEnv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return nil, fmt.Errorf("GOOGLE_CLOUD_PROJECT is not set")
	}

	cfg := &Config{
		Env:          env,
		Project:      project,
		ModelName:    modelName,
		GeminiAPIKey: geminiKey,
		MapsAPIKey:   mapsKey,
		Port:         port,
	}

	return cfg, nil
}
