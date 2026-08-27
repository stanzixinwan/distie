package upstream

import (
	"context"
	"io"
	"net"
	"strings"
	"testing"

	"google.golang.org/grpc"

	pb "github.com/stanzixinwan/distie/proto/gen/go"
)

func TestDial_EmptyAddr(t *testing.T) {
	_, err := Dial("")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestClient_InferAndStream(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterInferenceServiceServer(srv, &echoWorker{})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(func() {
		srv.Stop()
		_ = lis.Close()
	})

	client, err := Dial(lis.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })

	ctx := context.Background()
	req := &pb.InferenceRequest{
		RequestId: "r1",
		ModelName: "fake-lm",
		Prompt:    "hello world",
	}

	resp, err := client.Infer(ctx, req)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if resp.GetToken() != "hello world" {
		t.Fatalf("token = %q", resp.GetToken())
	}

	stream, err := client.InferStream(ctx, req)
	if err != nil {
		t.Fatalf("InferStream: %v", err)
	}
	var tokens []string
	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		tokens = append(tokens, msg.GetToken())
	}
	if len(tokens) != 2 || tokens[0] != "hello" || tokens[1] != "world" {
		t.Fatalf("tokens = %v", tokens)
	}
}

type echoWorker struct {
	pb.UnimplementedInferenceServiceServer
}

func (e *echoWorker) Infer(_ context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	return &pb.InferenceResponse{
		RequestId: req.GetRequestId(),
		Token:     req.GetPrompt(),
		Finished:  true,
	}, nil
}

func (e *echoWorker) InferStream(req *pb.InferenceRequest, stream grpc.ServerStreamingServer[pb.InferenceResponse]) error {
	tokens := strings.Fields(req.GetPrompt())
	for i, tok := range tokens {
		if err := stream.Send(&pb.InferenceResponse{
			RequestId: req.GetRequestId(),
			Token:     tok,
			Finished:  i == len(tokens)-1,
		}); err != nil {
			return err
		}
	}
	return nil
}
