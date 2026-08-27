package handler

import (
	"context"
	"log/slog"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/stanzixinwan/distie/proto/gen/go"
)

// InferenceHandler implements InferenceService.
// Phase 1: validate + stub. Later: forward to a selected Worker via gRPC client.
type InferenceHandler struct {
	pb.UnimplementedInferenceServiceServer
	log *slog.Logger
}

func NewInferenceHandler(log *slog.Logger) *InferenceHandler {
	if log == nil {
		log = slog.Default()
	}
	return &InferenceHandler{log: log}
}

func (h *InferenceHandler) Infer(
	ctx context.Context,
	req *pb.InferenceRequest,
) (*pb.InferenceResponse, error) {
	if err := validateInferenceRequest(req); err != nil {
		return nil, err
	}

	h.log.Info("infer requested",
		"request_id", req.GetRequestId(),
		"model", req.GetModelName(),
		"prompt_len", len(req.GetPrompt()),
	)

	// Scheduler / worker pool is not wired yet.
	return nil, status.Error(codes.Unavailable, "no inference workers registered")
}

func (h *InferenceHandler) InferStream(
	req *pb.InferenceRequest,
	stream pb.InferenceService_InferStreamServer,
) error {
	if err := validateInferenceRequest(req); err != nil {
		return err
	}

	h.log.Info("infer stream requested",
		"request_id", req.GetRequestId(),
		"model", req.GetModelName(),
		"prompt_len", len(req.GetPrompt()),
	)

	_ = stream // will Send() tokens once a Worker is attached
	return status.Error(codes.Unavailable, "no inference workers registered")
}

func validateInferenceRequest(req *pb.InferenceRequest) error {
	if req == nil {
		return status.Error(codes.InvalidArgument, "request is required")
	}
	if req.GetPrompt() == "" {
		return status.Error(codes.InvalidArgument, "prompt is required")
	}
	if req.GetModelName() == "" {
		return status.Error(codes.InvalidArgument, "model_name is required")
	}
	return nil
}
