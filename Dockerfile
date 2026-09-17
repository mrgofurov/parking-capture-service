FROM golang:1.26-alpine AS builder

WORKDIR /app
RUN apk add --no-cache git

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /capture-service ./cmd/server

FROM alpine:3.19

RUN apk add --no-cache ffmpeg ca-certificates tzdata
WORKDIR /app

COPY --from=builder /capture-service /app/capture-service
COPY .env.example /app/.env

EXPOSE 8086

CMD ["/app/capture-service"]
