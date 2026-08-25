package server

import (
	"context"
	"log/slog"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

// unaryLoggingInterceptor records method, latency, and gRPC status code.
func unaryLoggingInterceptor(log *slog.Logger) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req any,
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (any, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		log.Info("unary rpc",
			"method", info.FullMethod,
			"code", status.Code(err).String(),
			"latency_ms", time.Since(start).Milliseconds(),
		)
		return resp, err
	}
}

func streamLoggingInterceptor(log *slog.Logger) grpc.StreamServerInterceptor {
	return func(
		srv any,
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		start := time.Now()
		err := handler(srv, ss)
		log.Info("stream rpc",
			"method", info.FullMethod,
			"code", status.Code(err).String(),
			"latency_ms", time.Since(start).Milliseconds(),
		)
		return err
	}
}
