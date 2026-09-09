from __future__ import annotations

import logging

import grpc
from grpc_reflection.v1alpha import reflection

from inference.engine import Engine, FakeEngine
from inference.native import load_block_pool
from proto_gen import inference_pb2, inference_pb2_grpc
from worker.config import Config
from worker.servicer import InferenceServicer

_log = logging.getLogger(__name__)


def build_server(
    cfg: Config, engine: Engine | None = None
) -> tuple[grpc.aio.Server, int]:
    if engine is None:
        engine = _build_engine(cfg)

    server = grpc.aio.server()
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(engine=engine),
        server,
    )

    service_names = (
        inference_pb2.DESCRIPTOR.services_by_name["InferenceService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    bind_addr = _bind_addr(cfg.listen_addr)
    port = server.add_insecure_port(bind_addr)
    if port == 0:
        raise RuntimeError(f"failed to bind {bind_addr}")
    _log.info(
        "worker listening worker_id=%s addr=%s port=%s",
        cfg.worker_id,
        bind_addr,
        port,
    )
    return server, port


async def serve(cfg: Config) -> None:
    server, _port = build_server(cfg)
    await server.start()
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        _log.info("shutdown signal received")
        await server.stop(cfg.shutdown_grace_s)


def _build_engine(cfg: Config) -> Engine:
    pool = None
    if cfg.block_pool_enabled:
        pool = load_block_pool(cfg.num_blocks, cfg.block_size_bytes)
        _log.info(
            "block pool ready worker_id=%s num_blocks=%s block_size_bytes=%s",
            cfg.worker_id,
            cfg.num_blocks,
            cfg.block_size_bytes,
        )

    if cfg.engine_kind == "fake":
        return FakeEngine(token_delay_s=cfg.token_delay_s, pool=pool)
    if cfg.engine_kind == "torch":
        from inference.torch_engine import TorchEngine

        return TorchEngine.load(cfg.model_id, device=cfg.device, pool=pool)
    raise ValueError(f"unknown engine_kind: {cfg.engine_kind}")


def _bind_addr(addr: str) -> str:
    # Go-style ":50052" -> dual-stack listen, matching grpc Python examples.
    if addr.startswith(":"):
        return f"[::]{addr}"
    return addr
