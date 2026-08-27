package ratelimit

import (
	"fmt"
	"sync"
	"time"
)

// Bucket is a leaky bucket: water (requests) pours in, and leaks at a
// constant rate. A request is rejected when adding it would overflow.
//
// One RPC = one unit of water. InferStream counts as a single request
// at accept time, not per token.
type Bucket struct {
	mu       sync.Mutex
	capacity float64
	leakRate float64 // units per second
	level    float64
	last     time.Time
	now      func() time.Time
}

// New creates a bucket that leaks at rps units/second with capacity burst.
func New(rps, burst float64) (*Bucket, error) {
	return newBucket(rps, burst, time.Now)
}

func newBucket(rps, burst float64, now func() time.Time) (*Bucket, error) {
	if rps <= 0 {
		return nil, fmt.Errorf("rps must be > 0")
	}
	if burst < 1 {
		return nil, fmt.Errorf("burst must be >= 1")
	}
	if now == nil {
		now = time.Now
	}
	return &Bucket{
		capacity: burst,
		leakRate: rps,
		now:      now,
	}, nil
}

// Allow reports whether one request may proceed. Safe for concurrent use.
func (b *Bucket) Allow() bool {
	if b == nil {
		return true
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	now := b.now()
	if !b.last.IsZero() {
		leaked := now.Sub(b.last).Seconds() * b.leakRate
		b.level -= leaked
		if b.level < 0 {
			b.level = 0
		}
	}
	b.last = now

	if b.level+1 > b.capacity {
		return false
	}
	b.level++
	return true
}
