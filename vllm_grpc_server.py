"""
vLLM gRPC Server - Custom Implementation for Story ML-5.1
Wraps vLLM's AsyncLLMEngine with a gRPC interface.
"""

import argparse
import asyncio
import logging
import sys
import os
import time
from typing import List, AsyncGenerator
import grpc
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid

# Add generated directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))

try:
    import vllm_pb2
    import vllm_pb2_grpc
except ImportError:
    print("❌ gRPC stubs not found. Please run protoc first.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceService(vllm_pb2_grpc.InferenceServiceServicer):
    def __init__(self, engine: AsyncLLMEngine):
        self.engine = engine

    async def Infer(self, request: vllm_pb2.InferRequest, context: grpc.ServicerContext) -> vllm_pb2.InferResponse:
        """Perform non-streaming inference"""
        logger.info(f"Received Infer request for model: {request.model}")
        
        try:
            # Convert messages to prompt (simple concatenation for now, 
            # in production use chat templates)
            prompt = ""
            for msg in request.messages:
                prompt += f"{msg.role}: {msg.content}\n"
            prompt += "assistant: "

            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )

            request_id = random_uuid()
            results_generator = self.engine.generate(prompt, sampling_params, request_id)

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if final_output:
                text = final_output.outputs[0].text
                tokens = len(final_output.outputs[0].token_ids)
                
                return vllm_pb2.InferResponse(
                    content=text,
                    tokens_used=tokens,
                    model=request.model,
                    finish_reason=final_output.outputs[0].finish_reason or "stop"
                )
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("No output generated")
                return vllm_pb2.InferResponse()

        except Exception as e:
            logger.error(f"Inference error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vllm_pb2.InferResponse()

    async def InferStream(self, request: vllm_pb2.InferRequest, context: grpc.ServicerContext) -> AsyncGenerator[vllm_pb2.InferStreamResponse, None]:
        """Perform streaming inference"""
        logger.info(f"Received InferStream request for model: {request.model}")
        
        try:
            prompt = ""
            for msg in request.messages:
                prompt += f"{msg.role}: {msg.content}\n"
            prompt += "assistant: "

            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )

            request_id = random_uuid()
            results_generator = self.engine.generate(prompt, sampling_params, request_id)

            last_text_len = 0
            async for request_output in results_generator:
                full_text = request_output.outputs[0].text
                new_text = full_text[last_text_len:]
                last_text_len = len(full_text)
                
                is_final = request_output.finished
                
                yield vllm_pb2.InferStreamResponse(
                    token=new_text,
                    is_final=is_final
                )

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def Health(self, request: vllm_pb2.HealthRequest, context: grpc.ServicerContext) -> vllm_pb2.HealthResponse:
        """Health check"""
        return vllm_pb2.HealthResponse(healthy=True, status="OK")


async def serve():
    parser = argparse.ArgumentParser(description="vLLM gRPC Server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    
    args = parser.parse_args()

    # Initialize vLLM engine
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info(f"🚀 vLLM Engine initialized for model: {args.model}")

    # Initialize gRPC server
    server = grpc.aio.server()
    vllm_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(engine), server
    )
    
    listen_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(listen_addr)
    
    logger.info(f"📡 gRPC Server listening on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
