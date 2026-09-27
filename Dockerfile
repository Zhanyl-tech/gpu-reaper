FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /gpu-reaper ./cmd/gpu-reaper

# Runtime: distroless base (Debian, glibc, no shell). glibc rather than Alpine's
# musl because the NVIDIA container toolkit injects the host's glibc-linked
# nvidia-smi and libnvidia-ml; the request for Alpine support
# (NVIDIA/nvidia-container-toolkit issue #270) was closed as not planned.
#
# What works inside this image, as shipped:
#   - slurm.source: rest     (HTTP only)
#   - gpu.source: dcgm       (HTTP only)
#   - gpu.source: nvidia-smi only if the NVIDIA runtime mounts it in
# What does not: slurm.source: squeue, and enforce mode, because the image has
# no squeue, scancel, scontrol or munge. Baking Slurm client binaries in would
# pin the image to one Slurm version and authentication setup.
FROM gcr.io/distroless/base-debian12:nonroot
COPY --from=build /gpu-reaper /usr/local/bin/gpu-reaper
EXPOSE 9835
ENTRYPOINT ["/usr/local/bin/gpu-reaper"]
