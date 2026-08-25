package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/omniserve/llm_inference_server/gateway/internal/config"
	"github.com/omniserve/llm_inference_server/gateway/internal/handler"
	"github.com/omniserve/llm_inference_server/gateway/internal/server"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	cfg, err := config.Load()
	if err != nil {
		log.Error("load config", "err", err)
		os.Exit(1)
	}

	h := handler.NewInferenceHandler(log)
	srv, err := server.New(cfg.ListenAddr, h, log)
	if err != nil {
		log.Error("create server", "err", err)
		os.Exit(1)
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Serve()
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	select {
	case <-ctx.Done():
		log.Info("shutdown signal received")
	case err := <-errCh:
		if err != nil {
			log.Error("server exited", "err", err)
			os.Exit(1)
		}
	}

	// Bound graceful stop so a stuck stream cannot hang the process forever.
	done := make(chan struct{})
	go func() {
		srv.GracefulStop()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(cfg.ShutdownTimeout):
		log.Warn("graceful shutdown timed out")
	}
}
