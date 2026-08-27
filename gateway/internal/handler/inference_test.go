package handler_test

import (
	"context"
	"io"
	"log/slog"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"github.com/stanzixinwan/distie/gateway/internal/handler"
	"github.com/stanzixinwan/distie/gateway/internal/upstream"
	pb "github.com/stanzixinwan/distie/proto/gen/go"
)

func TestInfer_RequiresPrompt(t *testing.T) {
	h := handler.NewInferenceHandler(slog.Default(), nil)
	_, err := h.Infer(context.Background(), &pb.InferenceRequest{
		ModelName: "llama-3-8b",
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("want InvalidArgument, got %v", err)
	}
}

func TestInfer_RequiresModelName(t *testing.T) {
	h := handler.NewInferenceHandler(slog.Default(), nil)
	_, err := h.Infer(context.Background(), &pb.InferenceRequest{
		Prompt: "hello",
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("want InvalidArgument, got %v", err)
	}
}

func TestInfer_NoWorkers(t *testing.T) {
	h := handler.NewInferenceHandler(slog.Default(), nil)
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
	h := handler.NewInferenceHandler(slog.Default(), nil)
	err := h.InferStream(&pb.InferenceRequest{ModelName: "llama-3-8b"}, nil)
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("want InvalidArgument, got %v", err)
	}
}

func TestInferStream_NoWorkers(t *testing.T) {
	h := handler.NewInferenceHandler(slog.Default(), nil)
	err := h.InferStream(&pb.InferenceRequest{ModelName: "llama-3-8b", Prompt: "hello"}, nil)
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("want Unavailable, got %v", err)
	}
}

func TestInfer_ForwardsToUpstream(t *testing.T) {
	up := &fakeForwarder{inferResp: &pb.InferenceResponse{
		RequestId: "req-1",
		Token:     "hello world",
		Finished:  true,
	}}
	h := handler.NewInferenceHandler(slog.Default(), up)
	resp, err := h.Infer(context.Background(), &pb.InferenceRequest{
		RequestId: "req-1",
		ModelName: "fake-lm",
		Prompt:    "hello world",
	})
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if resp.GetToken() != "hello world" {
		t.Fatalf("token = %q", resp.GetToken())
	}
}

func TestInfer_MapsUpstreamUnavailable(t *testing.T) {
	up := &fakeForwarder{inferErr: status.Error(codes.Unavailable, "connection refused")}
	h := handler.NewInferenceHandler(slog.Default(), up)
	_, err := h.Infer(context.Background(), &pb.InferenceRequest{
		RequestId: "req-1",
		ModelName: "fake-lm",
		Prompt:    "hello",
	})
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("want Unavailable, got %v", err)
	}
	if status.Convert(err).Message() != "inference worker unavailable" {
		t.Fatalf("message = %q", status.Convert(err).Message())
	}
}

func TestInfer_PassesThroughInvalidArgument(t *testing.T) {
	up := &fakeForwarder{inferErr: status.Error(codes.InvalidArgument, "prompt is required")}
	h := handler.NewInferenceHandler(slog.Default(), up)
	_, err := h.Infer(context.Background(), &pb.InferenceRequest{
		RequestId: "req-1",
		ModelName: "fake-lm",
		Prompt:    "hello",
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("want InvalidArgument, got %v", err)
	}
}

func TestInferStream_PumpsTokens(t *testing.T) {
	up := &fakeForwarder{tokens: []*pb.InferenceResponse{
		{RequestId: "r2", Token: "a"},
		{RequestId: "r2", Token: "b", Finished: true},
	}}
	client := startGateway(t, up)
	stream, err := client.InferStream(context.Background(), &pb.InferenceRequest{
		RequestId: "r2",
		ModelName: "fake-lm",
		Prompt:    "a b",
	})
	if err != nil {
		t.Fatalf("InferStream: %v", err)
	}

	var got []string
	var last *pb.InferenceResponse
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		got = append(got, resp.GetToken())
		last = resp
	}
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("tokens = %v", got)
	}
	if last == nil || !last.GetFinished() {
		t.Fatal("last chunk should be finished")
	}
}

func TestInferStream_ClientCancel(t *testing.T) {
	up := &fakeForwarder{block: true}
	client := startGateway(t, up)

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.InferStream(ctx, &pb.InferenceRequest{
		RequestId: "r3",
		ModelName: "fake-lm",
		Prompt:    "hello",
	})
	if err != nil {
		t.Fatalf("InferStream: %v", err)
	}
	cancel()

	_, err = stream.Recv()
	if status.Code(err) != codes.Canceled {
		t.Fatalf("want Canceled, got %v", err)
	}
}

func startGateway(t *testing.T, up upstream.Forwarder) pb.InferenceServiceClient {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterInferenceServiceServer(srv, handler.NewInferenceHandler(slog.Default(), up))
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(func() {
		srv.Stop()
		_ = lis.Close()
	})

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return pb.NewInferenceServiceClient(conn)
}

type fakeForwarder struct {
	inferResp *pb.InferenceResponse
	inferErr  error
	tokens    []*pb.InferenceResponse
	streamErr error
	block     bool
}

func (f *fakeForwarder) Infer(_ context.Context, _ *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	return f.inferResp, f.inferErr
}

func (f *fakeForwarder) InferStream(ctx context.Context, _ *pb.InferenceRequest) (upstream.Stream, error) {
	if f.streamErr != nil {
		return nil, f.streamErr
	}
	return &sliceStream{ctx: ctx, tokens: f.tokens, block: f.block}, nil
}

type sliceStream struct {
	ctx    context.Context
	tokens []*pb.InferenceResponse
	i      int
	block  bool
}

func (s *sliceStream) Recv() (*pb.InferenceResponse, error) {
	if s.block {
		<-s.ctx.Done()
		return nil, s.ctx.Err()
	}
	if s.i >= len(s.tokens) {
		return nil, io.EOF
	}
	tok := s.tokens[s.i]
	s.i++
	return tok, nil
}
