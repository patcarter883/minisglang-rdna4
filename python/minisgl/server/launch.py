from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo, DpInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        tp_size = server_args.tp_info.size
        dp_size = server_args.dp_info.dp_size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        # Spawn dp_size*tp_size schedulers: one full-model engine replica per dp_rank, each replica's
        # tp_size TP ranks. Each process is tagged with its (dp_rank, tp_rank); the engine maps that to
        # its own card (device_index = dp_rank*tp_size + tp_rank within the lease-visible device set)
        # and its own per-replica ZMQ ingress (zmq_backend_addr keyed by dp_rank). Only the (dp=0,tp=0)
        # process plus each replica's tp-primary participate in routing; replies funnel through one
        # detokenizer. With dp_size=1 this loop is exactly the historical single-replica spawn.
        for dp_rank in range(dp_size):
            for tp_rank in range(tp_size):
                new_args = replace(
                    server_args,
                    tp_info=DistributedInfo(tp_rank, tp_size),
                    dp_info=DpInfo(dp_rank, dp_size),
                )
                mp.Process(
                    target=_run_scheduler,
                    args=(new_args, ack_queue),
                    daemon=False,
                    name=f"minisgl-DP{dp_rank}-TP{tp_rank}-scheduler",
                ).start()

        num_tokenizers = server_args.num_tokenizer
        # Per-replica ingress addresses — the tokenizer/detokenizer round-robins UserMsg across these
        # (one per DP replica's rank-0). dp_size=1 -> a single-element list == the historical behaviour.
        backend_addrs = [server_args.backend_addr_for(dp) for dp in range(dp_size)]
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addrs": backend_addrs,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addrs": backend_addrs,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - dp_size scheduler replicas (each replica's tp-PRIMARY sends one ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: dp_size + num_tokenizers + 1
        for _ in range(dp_size + num_tokenizers + 1):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
