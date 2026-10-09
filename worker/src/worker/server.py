from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Callable

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


async def serve(
    cfg: Config,
    engine: Engine | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    """Run until SIGINT/SIGTERM (or `stop` is set), then drain and close.

    Shutdown order: stop the server first, so in-flight streams finish within
    the grace period or are cancelled (their blocks are freed), then close the
    engine, which stops its GPU loop. A second signal while draining falls back
    to the default handler and interrupts the drain.
    """
    if engine is None:
        engine = _build_engine(cfg)
    stop = stop or asyncio.Event()
    server, _port = build_server(cfg, engine=engine)
    remove_handlers = _install_signal_handlers(stop)
    try:
        await server.start()
        await stop.wait()
        _log.info("shutdown started grace_s=%s", cfg.shutdown_grace_s)
    finally:
        remove_handlers()
        try:
            await server.stop(cfg.shutdown_grace_s)
        finally:
            await engine.close()
            _log.info("worker stopped worker_id=%s", cfg.worker_id)


def _install_signal_handlers(stop: asyncio.Event) -> Callable[[], None]:
    loop = asyncio.get_running_loop()
    signals = (signal.SIGINT, signal.SIGTERM)

    def remove() -> None:
        for sig in signals:
            loop.remove_signal_handler(sig)

    def on_signal(sig: signal.Signals) -> None:
        _log.info("shutdown signal received signal=%s", sig.name)
        remove()
        stop.set()

    try:
        for sig in signals:
            loop.add_signal_handler(sig, on_signal, sig)
    except NotImplementedError:
        # Windows event loops have no add_signal_handler; KeyboardInterrupt
        # still cancels serve() and the finally block runs the same shutdown.
        _log.warning("signal handlers unavailable on this platform")
        return lambda: None
    return remove


def _build_engine(cfg: Config) -> Engine:
    pool = None
    if cfg.block_pool_enabled:
        pool = load_block_pool(cfg.num_blocks)
        _log.info(
            "block pool ready worker_id=%s num_blocks=%s",
            cfg.worker_id,
            cfg.num_blocks,
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
