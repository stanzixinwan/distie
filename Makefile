PYTHON ?= python3
export PYTHON

MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
DEVICE ?= cuda
DTYPE ?= fp32

.PHONY: all proto core gateway test test-go test-worker test-bench correctness clean

all: proto core gateway

proto:
	bash scripts/gen_proto.sh

core:
	bash scripts/build_core.sh

gateway:
	go build -o bin/gateway ./gateway/cmd/gateway

test: test-go test-worker test-bench

test-go:
	go test ./gateway/...

test-worker:
	$(PYTHON) -m unittest discover -s worker/tests -t worker/tests -v

test-bench:
	$(PYTHON) -m unittest discover -s benchmarks/correctness -t benchmarks/correctness -v

correctness:
	$(PYTHON) benchmarks/correctness/hf_parity.py --model $(MODEL) --device $(DEVICE) \
		--dtype $(DTYPE) --out benchmarks/results/correctness-$(DTYPE).json

clean:
	rm -rf bin core/build
