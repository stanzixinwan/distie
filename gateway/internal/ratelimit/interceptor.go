package ratelimit

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const rejectedMsg = "rate limit exceeded"

// UnaryInterceptor rejects overflow with ResourceExhausted (HTTP 429 analogue).
func UnaryInterceptor(b *Bucket) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req any,
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (any, error) {
		if !b.Allow() {
			return nil, status.Error(codes.ResourceExhausted, rejectedMsg)
		}
		return handler(ctx, req)
	}
}

// StreamInterceptor rate-limits RPC acceptance, not individual tokens.
func StreamInterceptor(b *Bucket) grpc.StreamServerInterceptor {
	return func(
		srv any,
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		if !b.Allow() {
			return status.Error(codes.ResourceExhausted, rejectedMsg)
		}
		return handler(srv, ss)
	}
}
