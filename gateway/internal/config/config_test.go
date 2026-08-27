package config

import "testing"

func TestLoad_DefaultWorkerAddrEmpty(t *testing.T) {
	t.Setenv("DISTIE_WORKER_ADDR", "")
	t.Setenv("DISTIE_RATE_LIMIT_RPS", "")
	t.Setenv("DISTIE_RATE_LIMIT_BURST", "")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.WorkerAddr != "" {
		t.Fatalf("WorkerAddr = %q, want empty", cfg.WorkerAddr)
	}
}

func TestLoad_WorkerAddr(t *testing.T) {
	t.Setenv("DISTIE_WORKER_ADDR", "  127.0.0.1:50052  ")
	t.Setenv("DISTIE_RATE_LIMIT_RPS", "")
	t.Setenv("DISTIE_RATE_LIMIT_BURST", "")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.WorkerAddr != "127.0.0.1:50052" {
		t.Fatalf("WorkerAddr = %q", cfg.WorkerAddr)
	}
}

func TestLoad_RateLimitDefaults(t *testing.T) {
	t.Setenv("DISTIE_RATE_LIMIT_RPS", "")
	t.Setenv("DISTIE_RATE_LIMIT_BURST", "")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.RateLimitRPS != 20 || cfg.RateLimitBurst != 40 {
		t.Fatalf("rps=%v burst=%v", cfg.RateLimitRPS, cfg.RateLimitBurst)
	}
}

func TestLoad_RateLimitDisabled(t *testing.T) {
	t.Setenv("DISTIE_RATE_LIMIT_RPS", "0")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.RateLimitRPS != 0 {
		t.Fatalf("rps=%v", cfg.RateLimitRPS)
	}
}

func TestLoad_RateLimitInvalidBurst(t *testing.T) {
	t.Setenv("DISTIE_RATE_LIMIT_RPS", "10")
	t.Setenv("DISTIE_RATE_LIMIT_BURST", "0")
	if _, err := Load(); err == nil {
		t.Fatal("expected error")
	}
}
