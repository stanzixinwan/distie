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
	// RateLimitRPS is the leaky-bucket leak rate in requests/second.
	// 0 disables rate limiting.
	RateLimitRPS float64
	// RateLimitBurst is the bucket capacity (max requests that may sit
	// in the bucket before the next leak). Ignored when RateLimitRPS is 0.
	RateLimitBurst float64
}

// Load reads configuration from environment variables with safe defaults.
func Load() (Config, error) {
	cfg := Config{
		ListenAddr:      envOr("DISTIE_LISTEN_ADDR", ":50051"),
		WorkerAddr:      strings.TrimSpace(os.Getenv("DISTIE_WORKER_ADDR")),
		ShutdownTimeout: 15 * time.Second,
		RateLimitRPS:    20,
		RateLimitBurst:  40,
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

	rps, err := envFloat("DISTIE_RATE_LIMIT_RPS", cfg.RateLimitRPS)
	if err != nil {
		return Config{}, err
	}
	if rps < 0 {
		return Config{}, fmt.Errorf("DISTIE_RATE_LIMIT_RPS must be >= 0")
	}
	cfg.RateLimitRPS = rps

	burst, err := envFloat("DISTIE_RATE_LIMIT_BURST", cfg.RateLimitBurst)
	if err != nil {
		return Config{}, err
	}
	if rps > 0 && burst < 1 {
		return Config{}, fmt.Errorf("DISTIE_RATE_LIMIT_BURST must be >= 1 when rate limiting is on")
	}
	cfg.RateLimitBurst = burst

	return cfg, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envFloat(key string, fallback float64) (float64, error) {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback, nil
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	return f, nil
}
