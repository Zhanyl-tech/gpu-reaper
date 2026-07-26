BINARY  := gpu-reaper
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: help build test vet lint demo demo-once run-scenarios clean docker

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the binary
	go build -ldflags '$(LDFLAGS)' -o bin/$(BINARY) ./cmd/$(BINARY)

test: ## Run tests
	go test -race -cover ./...

vet: ## go vet
	go vet ./...

lint: vet test ## vet + test

demo: build ## Run the demo daemon (no cluster, no GPUs; Ctrl-C to stop)
	@echo "→ observe mode, fake squeue, simulated 'hung' GPUs"
	@echo "→ metrics on http://localhost:9835/metrics"
	@PATH="$(CURDIR)/demo/bin:$$PATH" ./bin/$(BINARY) --config demo/config.yaml

demo-once: build ## Run a single demo cycle and exit
	@PATH="$(CURDIR)/demo/bin:$$PATH" ./bin/$(BINARY) --config demo/config.yaml --once

run-scenarios: build ## Show every scenario's verdict side by side
	@for s in healthy idle hung starved flaky; do \
		printf '\n\033[1m── %s ──\033[0m\n' "$$s"; \
		sed "s/scenario: .*/scenario: $$s/" demo/config.yaml > /tmp/gr-$$s.yaml; \
		PATH="$(CURDIR)/demo/bin:$$PATH" ./bin/$(BINARY) --config /tmp/gr-$$s.yaml --once 2>&1 \
			| grep -E 'msg=finding' \
			| sed -E 's/.*verdict=([a-z]+) signature=([a-z]+) job_id=([0-9]+).*mean_util_pct=([0-9.]+).*/  job \3  verdict=\1  signature=\2  util=\4%/' \
			|| echo "  (no findings — healthy)"; \
	done; echo

docker: ## Build the container image
	docker build -t gpu-reaper:$(VERSION) .

clean:
	rm -rf bin /tmp/gr-*.yaml
