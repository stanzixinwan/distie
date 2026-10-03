PYTHON ?= python3
export PYTHON

.PHONY: all proto core gateway test test-go test-worker clean

all: proto core gateway

proto:
	bash scripts/gen_proto.sh

core:
	bash scripts/build_core.sh

gateway:
	go build -o bin/gateway ./gateway/cmd/gateway

test: test-go test-worker

test-go:
	go test ./gateway/...

test-worker:
	$(PYTHON) -m unittest discover -s worker/tests -t worker/tests -v

clean:
	rm -rf bin core/build
