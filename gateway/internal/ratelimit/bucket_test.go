package ratelimit

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNew_RejectsInvalid(t *testing.T) {
	if _, err := New(0, 10); err == nil {
		t.Fatal("rps=0 should fail")
	}
	if _, err := New(-1, 10); err == nil {
		t.Fatal("negative rps should fail")
	}
	if _, err := New(10, 0.5); err == nil {
		t.Fatal("burst < 1 should fail")
	}
}

func TestAllow_BurstThenReject(t *testing.T) {
	frozen := time.Unix(1_700_000_000, 0)
	b, err := newBucket(10, 3, func() time.Time { return frozen })
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if !b.Allow() {
			t.Fatalf("request %d should be allowed", i)
		}
	}
	if b.Allow() {
		t.Fatal("burst+1 should be rejected")
	}
}

func TestAllow_LeakFreesSlot(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	b, err := newBucket(10, 2, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	if !b.Allow() || !b.Allow() {
		t.Fatal("burst should fill")
	}
	if b.Allow() {
		t.Fatal("full bucket should reject")
	}

	// 10 rps → 100ms leaks 1 unit.
	now = now.Add(100 * time.Millisecond)
	if !b.Allow() {
		t.Fatal("after leak, one request should pass")
	}
	if b.Allow() {
		t.Fatal("bucket should be full again")
	}
}

func TestAllow_IdleDrainsCompletely(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	b, err := newBucket(5, 4, func() time.Time { return now })
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if !b.Allow() {
			t.Fatal("fill")
		}
	}
	now = now.Add(time.Second) // leaks 5, capacity 4 → empty
	for i := 0; i < 4; i++ {
		if !b.Allow() {
			t.Fatalf("after idle, request %d should pass", i)
		}
	}
	if b.Allow() {
		t.Fatal("burst must still cap after refill")
	}
}

func TestAllow_ConcurrentRespectsBurst(t *testing.T) {
	frozen := time.Unix(1_700_000_000, 0)
	const burst = 10
	b, err := newBucket(100, burst, func() time.Time { return frozen })
	if err != nil {
		t.Fatal(err)
	}

	var allowed atomic.Int64
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if b.Allow() {
				allowed.Add(1)
			}
		}()
	}
	wg.Wait()
	if got := allowed.Load(); got != burst {
		t.Fatalf("allowed = %d, want %d", got, burst)
	}
}

func TestAllow_NilBucketPasses(t *testing.T) {
	var b *Bucket
	if !b.Allow() {
		t.Fatal("nil bucket must fail-open for disabled limiter")
	}
}
