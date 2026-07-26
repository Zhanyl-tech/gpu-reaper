FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
RUN CGO_ENABLED=0 go build -ldflags "-s -w -X main.version=${VERSION}" -o /gpu-reaper ./cmd/gpu-reaper

# The runtime image intentionally has no nvidia-smi. On a GPU node the binary is
# run directly or in a container with the NVIDIA runtime mounted in; baking a
# driver-version-specific toolkit into the image would pin it to one node image.
FROM alpine:3.21
RUN apk add --no-cache ca-certificates && adduser -D -u 10001 reaper
COPY --from=build /gpu-reaper /usr/local/bin/gpu-reaper
USER reaper
EXPOSE 9835
ENTRYPOINT ["/usr/local/bin/gpu-reaper"]
