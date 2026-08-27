package handler_test

import (
	"context"
	"log/slog"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/stanzixinwan/llm_inference_server/gateway/internal/handler"
	pb "github.com/stanzixinwan/llm_inference_server/proto/gen/go"
)

func TestInfer_RequiresPrompt(t *testing.T) {
	h := handler.NewInferenceHandler(slog.Default())
	_, err := h.Infer(context.Background(), &pb.InferenceRequest{
		ModelName: "llama-3-8b",
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("want InvalidArgument, got %v", err)
	}
}

func TestInfer_RequiresModelName(t *testing.T) {
	h := handler.NewInferenceHandler(slog.Default())
	_, err := h.Infer(context.Background(), &pb.InferenceRequest{
		Prompt: "hello",
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("want InvalidArgument, got %v", err)
	}
}

func TestInfer_NoWorkers(t *testing.T) {
	h := handler.NewInferenceHandler(slog.Default())
	_, err := h.Infer(context.Background(), &pb.InferenceRequest{
		RequestId: "req-1",
		ModelName: "llama-3-8b",
		Prompt:    "hello",
	})
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("want Unavailable, got %v", err)
	}
}

func TestInferStream_RequiresPrompt(t *testing.T) {
	h := handler.NewInferenceHandler(slog.Default())
	err := h.InferStream(&pb.InferenceRequest{ModelName: "llama-3-8b"}, nil)
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("want InvalidArgument, got %v", err)
	}
}
