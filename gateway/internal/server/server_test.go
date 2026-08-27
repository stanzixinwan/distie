package server_test

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"github.com/stanzixinwan/distie/gateway/internal/handler"
	"github.com/stanzixinwan/distie/gateway/internal/ratelimit"
	"github.com/stanzixinwan/distie/gateway/internal/server"
	pb "github.com/stanzixinwan/distie/proto/gen/go"
)

func TestRateLimit_RejectsAfterBurst(t *testing.T) {
	// Tiny leak rate so sequential RPCs cannot refill a slot mid-test.
	limiter, err := ratelimit.New(0.01, 2)
	if err != nil {
		t.Fatal(err)
	}

	h := handler.NewInferenceHandler(slog.Default(), nil)
	srv, err := server.New("127.0.0.1:0", h, slog.Default(), limiter)
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(srv.GracefulStop)

	conn, err := grpc.NewClient(srv.Addr(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	stub := pb.NewInferenceServiceClient(conn)
	req := &pb.InferenceRequest{ModelName: "fake-lm", Prompt: "hello"}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	codesGot := make([]codes.Code, 0, 3)
	for i := 0; i < 3; i++ {
		_, err := stub.Infer(ctx, req, grpc.WaitForReady(true))
		codesGot = append(codesGot, status.Code(err))
	}
	if codesGot[0] != codes.Unavailable || codesGot[1] != codes.Unavailable {
		t.Fatalf("first two should be Unavailable (no worker), got %v", codesGot)
	}
	if codesGot[2] != codes.ResourceExhausted {
		t.Fatalf("third should be ResourceExhausted, got %v", codesGot)
	}
}
