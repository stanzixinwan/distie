package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/stanzixinwan/distie/gateway/internal/config"
	"github.com/stanzixinwan/distie/gateway/internal/handler"
	"github.com/stanzixinwan/distie/gateway/internal/server"
	"github.com/stanzixinwan/distie/gateway/internal/upstream"
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

	var up upstream.Forwarder
	if cfg.WorkerAddr != "" {
		client, dialErr := upstream.Dial(cfg.WorkerAddr)
		if dialErr != nil {
			log.Error("dial worker", "addr", cfg.WorkerAddr, "err", dialErr)
			os.Exit(1)
		}
		defer func() {
			if closeErr := client.Close(); closeErr != nil {
				log.Error("close worker connection", "err", closeErr)
			}
		}()
		up = client
		log.Info("upstream worker configured", "addr", cfg.WorkerAddr)
	}

	h := handler.NewInferenceHandler(log, up)
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
