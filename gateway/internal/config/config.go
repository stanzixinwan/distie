package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds gateway runtime settings.
// Prefer env vars in production so the same binary works across environments.
type Config struct {
	// ListenAddr is the host:port for the public gRPC server.
	ListenAddr string
	// WorkerAddr is the host:port of a single upstream Worker.
	// Empty means no Worker is configured; the handler returns Unavailable.
	WorkerAddr string
	// ShutdownTimeout is how long GracefulStop may wait for in-flight RPCs.
	ShutdownTimeout time.Duration
}

// Load reads configuration from environment variables with safe defaults.
func Load() (Config, error) {
	cfg := Config{
		ListenAddr:      envOr("DISTIE_LISTEN_ADDR", ":50051"),
		WorkerAddr:      strings.TrimSpace(os.Getenv("DISTIE_WORKER_ADDR")),
		ShutdownTimeout: 15 * time.Second,
	}

	if v := os.Getenv("DISTIE_SHUTDOWN_TIMEOUT_SEC"); v != "" {
		sec, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("DISTIE_SHUTDOWN_TIMEOUT_SEC: %w", err)
		}
		if sec <= 0 {
			return Config{}, fmt.Errorf("DISTIE_SHUTDOWN_TIMEOUT_SEC must be > 0")
		}
		cfg.ShutdownTimeout = time.Duration(sec) * time.Second
	}

	return cfg, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
