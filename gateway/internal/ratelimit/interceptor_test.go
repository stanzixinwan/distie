package ratelimit

import (
	"context"
	"io"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

func TestUnaryInterceptor_AllowsThenRejects(t *testing.T) {
	frozen := time.Unix(1_700_000_000, 0)
	b, err := newBucket(1, 1, func() time.Time { return frozen })
	if err != nil {
		t.Fatal(err)
	}
	interceptor := UnaryInterceptor(b)
	info := &grpc.UnaryServerInfo{FullMethod: "/svc/Infer"}
	passthrough := func(context.Context, any) (any, error) { return "ok", nil }

	resp, err := interceptor(context.Background(), nil, info, passthrough)
	if err != nil || resp != "ok" {
		t.Fatalf("first call: resp=%v err=%v", resp, err)
	}

	called := false
	_, err = interceptor(context.Background(), nil, info, func(context.Context, any) (any, error) {
		called = true
		return "ok", nil
	})
	if status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("want ResourceExhausted, got %v", err)
	}
	if called {
		t.Fatal("handler must not run when limited")
	}
}

func TestStreamInterceptor_RejectsOverflow(t *testing.T) {
	frozen := time.Unix(1_700_000_000, 0)
	b, err := newBucket(1, 1, func() time.Time { return frozen })
	if err != nil {
		t.Fatal(err)
	}
	interceptor := StreamInterceptor(b)
	info := &grpc.StreamServerInfo{FullMethod: "/svc/InferStream"}
	ss := stubServerStream{ctx: context.Background()}

	if err := interceptor(nil, ss, info, func(any, grpc.ServerStream) error { return nil }); err != nil {
		t.Fatalf("first stream: %v", err)
	}

	called := false
	err = interceptor(nil, ss, info, func(any, grpc.ServerStream) error {
		called = true
		return nil
	})
	if status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("want ResourceExhausted, got %v", err)
	}
	if called {
		t.Fatal("handler must not run when limited")
	}
}

type stubServerStream struct {
	ctx context.Context
}

func (s stubServerStream) SetHeader(metadata.MD) error  { return nil }
func (s stubServerStream) SendHeader(metadata.MD) error { return nil }
func (s stubServerStream) SetTrailer(metadata.MD)       {}
func (s stubServerStream) Context() context.Context     { return s.ctx }
func (s stubServerStream) SendMsg(any) error            { return io.EOF }
func (s stubServerStream) RecvMsg(any) error            { return io.EOF }
