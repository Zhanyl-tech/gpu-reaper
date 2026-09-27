BINARY  := gpu-reaper
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)
SCENARIOS := healthy idle hung starved flaky unreadable

.PHONY: help build test vet fmt-check lint demo demo-once demo-check run-scenarios demo-record clean docker

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the binary
	go build -ldflags '$(LDFLAGS)' -o bin/$(BINARY) ./cmd/$(BINARY)

test: ## Run tests (race detector + coverage)
	go test -race -cover ./...

vet: ## go vet
	go vet ./...

fmt-check: ## Fail if any file needs gofmt
	@test -z "$$(gofmt -l .)" || { gofmt -l .; echo "run gofmt -w ." >&2; exit 1; }

lint: fmt-check vet test ## gofmt + vet + test

demo: build ## Run the demo daemon (no cluster, no GPUs; Ctrl-C to stop)
	@echo "→ observe mode, fake squeue, simulated 'hung' GPUs"
	@echo "→ metrics on http://localhost:9835/metrics"
	@FAKE_SQUEUE_ANCHOR=$$(date +%s) PATH="$(CURDIR)/demo/bin:$$PATH" ./bin/$(BINARY) --config demo/config.yaml

demo-once: build ## Run a single demo cycle (reports nothing: one snapshot is never a breach)
	@FAKE_SQUEUE_ANCHOR=$$(date +%s) PATH="$(CURDIR)/demo/bin:$$PATH" ./bin/$(BINARY) --config demo/config.yaml --once

demo-check: build ## Run a bounded demo and assert the README's description of it
	@./demo/check.sh hung

run-scenarios: build ## Run every simulator scenario and assert its outcome
	@for s in $(SCENARIOS); do ./demo/check.sh $$s || exit 1; done

demo-record: build ## Capture a bounded demo run to docs/demo.log and render docs/demo.svg from it
	@FAKE_SQUEUE_ANCHOR=$$(date +%s) TZ=UTC PATH="$(CURDIR)/demo/bin:$$PATH" \
		./bin/$(BINARY) --config demo/config.yaml --cycles 9 > docs/demo.log 2>&1
	go run ./hack/demosvg < docs/demo.log > docs/demo.svg

docker: ## Build the container image
	docker build -t gpu-reaper:$(VERSION) .

clean:
	rm -rf bin
