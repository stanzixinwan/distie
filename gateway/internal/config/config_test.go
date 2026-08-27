package config

import "testing"

func TestLoad_DefaultWorkerAddrEmpty(t *testing.T) {
	t.Setenv("DISTIE_WORKER_ADDR", "")
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
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.WorkerAddr != "127.0.0.1:50052" {
		t.Fatalf("WorkerAddr = %q", cfg.WorkerAddr)
	}
}
