from __future__ import annotations

import logging

import grpc

from inference.engine import (
    BlockPoolExhausted,
    FakeEngine,
    GenerateRequest,
    TokenEvent,
)
from proto_gen import inference_pb2, inference_pb2_grpc

_log = logging.getLogger(__name__)


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC adapter: Protobuf <-> FakeEngine.

    No model logic lives here. That stays in inference.engine so we can
    later swap in a Torch engine without rewriting the RPC layer.
    """

    def __init__(self, engine: FakeEngine, logger: logging.Logger | None = None) -> None:
        self._engine = engine
        self._log = logger or _log

    async def Infer(
        self,
        request: inference_pb2.InferenceRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.InferenceResponse:
        pieces: list[str] = []
        last: inference_pb2.InferenceResponse | None = None
        async for resp in self._responses(request, context):
            if resp.token:
                pieces.append(resp.token)
            last = resp
        if last is None:
            await context.abort(grpc.StatusCode.INTERNAL, "engine produced no events")
            return inference_pb2.InferenceResponse()
        return inference_pb2.InferenceResponse(
            request_id=request.request_id,
            token=" ".join(pieces),
            finished=True,
            usage=last.usage,
        )

    async def InferStream(
        self,
        request: inference_pb2.InferenceRequest,
        context: grpc.aio.ServicerContext,
    ):
        async for resp in self._responses(request, context):
            yield resp

    async def _responses(
        self,
        request: inference_pb2.InferenceRequest,
        context: grpc.aio.ServicerContext,
    ):
        err = _validate(request)
        if err:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, err)

        self._log.info(
            "generate start request_id=%s model=%s prompt_len=%s",
            request.request_id,
            request.model_name,
            len(request.prompt),
        )

        gen_req = GenerateRequest(
            request_id=request.request_id,
            model_name=request.model_name,
            prompt=request.prompt,
            max_tokens=request.params.max_tokens,
        )

        try:
            async for event in self._engine.generate(gen_req):
                if context.cancelled():
                    self._log.info("generate cancelled request_id=%s", request.request_id)
                    return
                yield _to_response(request.request_id, event)
        except BlockPoolExhausted as exc:
            self._log.warning(
                "block pool exhausted request_id=%s err=%s",
                request.request_id,
                exc,
            )
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))


def _validate(request: inference_pb2.InferenceRequest) -> str | None:
    if not request.prompt.strip():
        return "prompt is required"
    if not request.model_name.strip():
        return "model_name is required"
    return None


def _to_response(request_id: str, event: TokenEvent) -> inference_pb2.InferenceResponse:
    resp = inference_pb2.InferenceResponse(
        request_id=request_id,
        token=event.token,
        finished=event.finished,
    )
    if event.finished:
        resp.usage.prompt_tokens = event.prompt_tokens
        resp.usage.completion_tokens = event.completion_tokens
        resp.usage.total_tokens = event.prompt_tokens + event.completion_tokens
        resp.usage.time_to_first_token_ms = event.time_to_first_token_ms
        resp.usage.total_latency_ms = event.total_latency_ms
    return resp
