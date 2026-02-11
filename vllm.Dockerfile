# Custom vLLM gRPC Server Dockerfile
FROM vllm/vllm-openai:latest

# Install gRPC dependencies
RUN pip install --no-cache-dir grpcio grpcio-tools

# Set working directory
WORKDIR /app

# Copy proto and server implementation
COPY protos/vllm.proto /app/protos/vllm.proto
COPY vllm_grpc_server.py /app/vllm_grpc_server.py

# Generate gRPC stubs
RUN mkdir -p /app/generated && \
    python3 -m grpc_tools.protoc \
    -I/app/protos \
    --python_out=/app/generated \
    --grpc_python_out=/app/generated \
    /app/protos/vllm.proto

# Expose gRPC port
EXPOSE 9000

# Set environment variables
ENV PYTHONPATH=$PYTHONPATH:/app:/app/generated

# Run the gRPC server
# The arguments will be passed via docker-compose command
ENTRYPOINT ["python3", "/app/vllm_grpc_server.py"]
