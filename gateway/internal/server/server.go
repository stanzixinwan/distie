package server

import (
	"fmt"
	"log/slog"
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	"github.com/stanzixinwan/llm_inference_server/gateway/internal/handler"
	pb "github.com/stanzixinwan/llm_inference_server/proto/gen/go"
)

// Server wraps a gRPC server and its listener.
type Server struct {
	log      *slog.Logger
	grpcSrv  *grpc.Server
	listener net.Listener
	addr     string
}

// New creates a gRPC server that serves InferenceService on addr.
func New(addr string, h *handler.InferenceHandler, log *slog.Logger) (*Server, error) {
	if log == nil {
		log = slog.Default()
	}
	if h == nil {
		return nil, fmt.Errorf("inference handler is required")
	}

	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("listen %s: %w", addr, err)
	}

	grpcSrv := grpc.NewServer(
		grpc.ChainUnaryInterceptor(unaryLoggingInterceptor(log)),
		grpc.ChainStreamInterceptor(streamLoggingInterceptor(log)),
	)

	pb.RegisterInferenceServiceServer(grpcSrv, h)
	// reflection: lets grpcurl discover methods without a local .proto file.
	reflection.Register(grpcSrv)

	return &Server{
		log:      log,
		grpcSrv:  grpcSrv,
		listener: lis,
		addr:     lis.Addr().String(),
	}, nil
}

// Addr returns the actual bound address (useful when ListenAddr is ":0").
func (s *Server) Addr() string {
	return s.addr
}

// Serve blocks until the server stops.
func (s *Server) Serve() error {
	s.log.Info("gRPC gateway listening", "addr", s.addr)
	if err := s.grpcSrv.Serve(s.listener); err != nil {
		return fmt.Errorf("grpc serve: %w", err)
	}
	return nil
}

// GracefulStop stops accepting new RPCs and waits for in-flight ones.
func (s *Server) GracefulStop() {
	s.log.Info("graceful shutdown started")
	s.grpcSrv.GracefulStop()
	s.log.Info("graceful shutdown complete")
}
