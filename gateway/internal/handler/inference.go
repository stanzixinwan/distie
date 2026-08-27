package handler

import (
	"context"
	"errors"
	"io"
	"log/slog"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/stanzixinwan/distie/gateway/internal/upstream"
	pb "github.com/stanzixinwan/distie/proto/gen/go"
)

// InferenceHandler implements InferenceService by validating requests and
// forwarding them to a single upstream Worker.
type InferenceHandler struct {
	pb.UnimplementedInferenceServiceServer
	log      *slog.Logger
	upstream upstream.Forwarder
}

func NewInferenceHandler(log *slog.Logger, up upstream.Forwarder) *InferenceHandler {
	if log == nil {
		log = slog.Default()
	}
	return &InferenceHandler{log: log, upstream: up}
}

func (h *InferenceHandler) Infer(
	ctx context.Context,
	req *pb.InferenceRequest,
) (*pb.InferenceResponse, error) {
	if err := validateInferenceRequest(req); err != nil {
		return nil, err
	}
	if h.upstream == nil {
		return nil, status.Error(codes.Unavailable, "no inference workers registered")
	}

	h.log.Info("infer requested",
		"request_id", req.GetRequestId(),
		"model", req.GetModelName(),
		"prompt_len", len(req.GetPrompt()),
	)

	resp, err := h.upstream.Infer(ctx, req)
	if err != nil {
		h.log.Error("upstream infer failed",
			"request_id", req.GetRequestId(),
			"err", err,
		)
		return nil, mapUpstreamError(err)
	}
	return resp, nil
}

func (h *InferenceHandler) InferStream(
	req *pb.InferenceRequest,
	stream pb.InferenceService_InferStreamServer,
) error {
	if err := validateInferenceRequest(req); err != nil {
		return err
	}
	if h.upstream == nil {
		return status.Error(codes.Unavailable, "no inference workers registered")
	}

	h.log.Info("infer stream requested",
		"request_id", req.GetRequestId(),
		"model", req.GetModelName(),
		"prompt_len", len(req.GetPrompt()),
	)

	ctx := stream.Context()
	upstreamStream, err := h.upstream.InferStream(ctx, req)
	if err != nil {
		h.log.Error("upstream infer stream failed",
			"request_id", req.GetRequestId(),
			"err", err,
		)
		return mapUpstreamError(err)
	}

	for {
		token, err := upstreamStream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			if ctx.Err() != nil {
				h.log.Info("client cancelled", "request_id", req.GetRequestId())
				return status.FromContextError(ctx.Err()).Err()
			}
			h.log.Error("upstream stream recv failed",
				"request_id", req.GetRequestId(),
				"err", err,
			)
			return mapUpstreamError(err)
		}
		if err := stream.Send(token); err != nil {
			h.log.Info("send to client failed",
				"request_id", req.GetRequestId(),
				"err", err,
			)
			return err
		}
	}
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

func mapUpstreamError(err error) error {
	st, ok := status.FromError(err)
	if !ok {
		return status.Error(codes.Internal, "upstream inference failed")
	}
	switch st.Code() {
	case codes.InvalidArgument, codes.NotFound, codes.ResourceExhausted, codes.FailedPrecondition:
		return err
	case codes.Unavailable, codes.DeadlineExceeded:
		return status.Error(codes.Unavailable, "inference worker unavailable")
	case codes.Canceled:
		return err
	default:
		return status.Error(codes.Internal, "upstream inference failed")
	}
}
