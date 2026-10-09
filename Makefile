VENV := $(CURDIR)/.venv
# Make 3.81 (macOS) resolves metachar-free recipes with its startup PATH,
# so PYTHON must be absolute; the PATH export still reaches child scripts.
ifneq ($(wildcard $(VENV)/bin/python),)
export PATH := $(VENV)/bin:$(PATH)
PYTHON ?= $(VENV)/bin/python
endif

PYTHON ?= python3
export PYTHON

MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
DEVICE ?= cuda
DTYPE ?= fp32

# BACKEND=openai with TARGET=http://localhost:8000 benchmarks vLLM instead.
BACKEND ?= grpc
TARGET ?= localhost:50052
WORKLOAD ?= sharegpt
NUM_REQUESTS ?= 200
RATE ?= inf
MAX_CONCURRENCY ?= 64
SHAREGPT := benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json
SHAREGPT_URL := https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

.PHONY: all setup proto core gateway test test-go test-worker test-bench correctness bench sharegpt clean

all: proto core gateway

setup:
	bash scripts/setup_dev.sh

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
	$(PYTHON) -m unittest discover -s benchmarks/load -t benchmarks/load -v

correctness:
	$(PYTHON) benchmarks/correctness/hf_parity.py --model $(MODEL) --device $(DEVICE) \
		--dtype $(DTYPE) --out benchmarks/results/correctness-$(DTYPE).json

sharegpt: $(SHAREGPT)

$(SHAREGPT):
	mkdir -p $(dir $@)
	curl -fL --retry 3 -o $@.tmp $(SHAREGPT_URL) && mv $@.tmp $@

bench:
	$(PYTHON) benchmarks/load/bench_serving.py --backend $(BACKEND) --target $(TARGET) --workload $(WORKLOAD) \
		$(if $(filter sharegpt,$(WORKLOAD)),--dataset $(SHAREGPT) --tokenizer $(MODEL)) \
		--num-requests $(NUM_REQUESTS) --rate $(RATE) \
		--max-concurrency $(MAX_CONCURRENCY) \
		--out benchmarks/results/serving-$(BACKEND)-$(WORKLOAD)-rate$(RATE)-c$(MAX_CONCURRENCY).json

clean:
	rm -rf bin core/build
