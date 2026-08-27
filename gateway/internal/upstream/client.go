package upstream

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/stanzixinwan/distie/proto/gen/go"
)

// Stream is a server-streaming RPC from a Worker. Recv returns io.EOF when
// the Worker finishes normally.
type Stream interface {
	Recv() (*pb.InferenceResponse, error)
}

// Forwarder sends validated inference RPCs to a Worker.
// Handler depends on this interface so tests can inject a fake.
type Forwarder interface {
	Infer(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error)
	InferStream(ctx context.Context, req *pb.InferenceRequest) (Stream, error)
}

// Client is a long-lived gRPC connection to a single Worker.
type Client struct {
	conn *grpc.ClientConn
	stub pb.InferenceServiceClient
}

// Dial opens a ClientConn to addr. The handshake is lazy: a bad Worker
// surfaces on the first RPC, not here.
func Dial(addr string) (*Client, error) {
	if addr == "" {
		return nil, fmt.Errorf("worker address is required")
	}
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial worker %s: %w", addr, err)
	}
	return New(conn), nil
}

// New wraps an existing connection. Tests pass a bufconn/local ClientConn.
func New(conn *grpc.ClientConn) *Client {
	return &Client{
		conn: conn,
		stub: pb.NewInferenceServiceClient(conn),
	}
}

func (c *Client) Infer(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	return c.stub.Infer(ctx, req)
}

func (c *Client) InferStream(ctx context.Context, req *pb.InferenceRequest) (Stream, error) {
	stream, err := c.stub.InferStream(ctx, req)
	if err != nil {
		return nil, err
	}
	return stream, nil
}

func (c *Client) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}
