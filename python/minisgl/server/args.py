from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo, DpInfo
from minisgl.rsa.config import RSAParams, add_rsa_args, params_from_args
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger, is_rocm


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    # Reasoning-content extraction for thinking models: split the completion on the model's reasoning
    # close delimiter into reasoning_content + content. "auto" (default) DERIVES the delimiters from
    # the checkpoint's own chat template, so a family with different markup (Gemma-4's asymmetric
    # `<|channel>thought` … `<channel|>`) works with no code change; a named family FORCES that pair;
    # "none" disables extraction. See server/reasoning.py for the cascade.
    reasoning_parser: str = "auto"
    # Chat-template OVERRIDE: a path to a .jinja file, or a literal jinja string. Replaces the
    # checkpoint's baked-in template for the SERVED model everywhere it renders (frontend probes,
    # tokenizer workers) — the escape hatch for a checkpoint that ships a broken or thinking-less
    # template. Both reference engines have this (--chat-template); we did not.
    chat_template: str | None = None
    # Server-level DEFAULT chat_template_kwargs, as a JSON object string. Merged UNDER each
    # request's own chat_template_kwargs (the request wins). NOT applied to the reasoning-delimiter
    # derivation probes, which need the template's own defaults to diff against.
    chat_template_kwargs: str | None = None
    # Server-side DEFAULT Markovian-RSA parameters (set by the --rsa-* flags). A per-request `rsa`
    # field on /v1/chat/completions patches these; RSA runs ONLY when a request opts in (rsa present
    # and enabled), so a normal call is an ordinary single completion. See
    # api_server.v1_chat_completions (RSA is chat-only: /v1/completions 400s an `rsa` field, because
    # the rollout loop drives chat messages and has nothing to do with a raw prompt).
    rsa_defaults: RSAParams = field(default_factory=RSAParams)
    # Name the server ADVERTISES in /v1/models (and echoes in responses), decoupled from the
    # checkpoint path given to --model. Lets serve.sh expose a stable short alias (e.g.
    # "Qwen3.8-27B") so clients don't have to track the publisher/quant suffix of whichever
    # checkpoint happens to be loaded. None => advertise model_path, as before.
    served_model_name: str | None = None

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        # ONE shared rendezvous port for ALL ranks: dp_size>1 forms a single global world over every
        # (dp_rank, tp_rank) on this port (Engine._init_dp_communication then carves TP/DP subgroups);
        # dp_size=1 keeps the historical single TP world on the same port. (server_port+1 is reserved
        # for distributed init and never serves HTTP.)
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=ServerArgs.served_model_name,
        help="Name advertised in /v1/models and echoed in responses. Defaults to the --model "
        "path; set a short alias (e.g. Qwen3.8-27B) so clients don't depend on the checkpoint.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--data-parallel-size",
        "--dp-size",
        type=int,
        default=1,
        help="The data parallelism size: number of full-model replicas behind ONE endpoint. "
        "Each replica runs its own tp_size TP ranks; total backend processes = dp_size*tp_size. "
        "Used for models whose backbone cannot tensor-parallelize (e.g. ZAYA's CCA). Default 1 "
        "(single replica, fully inert DP path).",
    )

    parser.add_argument(
        "--enable-ep",
        action="store_true",
        dest="enable_ep",
        help="Enable expert parallelism: shard the MoE experts across the DP replicas (requires "
        "--data-parallel-size > 1). Off by default; the DP launcher replicates every expert.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    # Weight offload (weights/plan.py). Neither flag is an on/off switch — plan §6.2 forbids one —
    # they CLAMP a decision the resolver makes from config on every serve. Left at their 0.0
    # defaults the engine derives the device tier from the card's total memory, so a model that fits
    # stays entirely VRAM-resident and the offload path costs nothing.
    #
    # BOTH ARE GiB (2**30), not decimal GB, because that is what consumes them: `plan.GIB_PER_UNIT`
    # scales both fields, `Engine._weight_offload_device_budget` converts with the same constant, and
    # `WeightPlanResolution.summary_line()` renders every figure it prints back in GiB. Saying "GB"
    # here would be a 7.4% lie in the help text of a knob whose whole job is to be an exact byte
    # budget — and 7.4% of a tier is enough to move a layer across the greedy fill boundary.
    parser.add_argument(
        "--weight-offload-device-gb",
        type=float,
        default=ServerArgs.weight_offload_device_gb,
        help=(
            "VRAM per rank (GiB) the MoE expert tier may occupy. 0 = derive it from the card's "
            "total memory. Lowering it moves whole MoE layers to the pinned host arena; each GiB "
            "surrendered is roughly 200k KV tokens. Applies to MoE expert stacks only — a dense "
            "model has nothing for this build to offload and the flag is reported as a no-op."
        ),
    )
    parser.add_argument(
        "--weight-offload-gb",
        type=float,
        default=ServerArgs.weight_offload_gb,
        help=(
            "Per-rank pinned host weight arena budget (GiB), replacing the measured P3b default in "
            "BOTH directions. 0 = keep the default. Never an enable switch."
        ),
    )

    parser.add_argument(
        "--weight-offload-cpu-layers",
        type=int,
        default=ServerArgs.weight_offload_cpu_layers,
        help=(
            "How many of the DEEPEST offloadable MoE layers are COMPUTED by host AVX-512 cores "
            "instead of being streamed to the card. 0 = off. A third placement tier: these layers "
            "need neither VRAM nor PINNED host memory (ordinary pageable pages suffice, because "
            "the reader is a CPU core), and nothing but the ~15 KB activation/route per layer per "
            "token crosses PCIe. Costs physical cores (an over-budget request is REFUSED, not "
            "clamped) and int8 activations (8.3e-03 rel_rms, against the 4.1e-02 the GPU's "
            "per-token fp8 costs today). The core is a GEMV: correct at any batch, economic only "
            "at batch 1, so a prefill chunk pays its token count times the decode cost."
        ),
    )

    parser.add_argument(
        "--weight-offload-stream-layers",
        type=int,
        default=ServerArgs.weight_offload_stream_layers,
        help=(
            "How many of the LAST MoE layers stream their routed experts off the checkpoint every "
            "forward instead of occupying VRAM or the pinned host arena. 0 = off. This is the "
            "capacity tier of last resort: it costs a disk read per layer per step, so use it only "
            "when the plan refuses. Requires --cuda-graph-max-bs 0 (the gather is not capturable) "
            "and a checkpoint with per-expert granularity."
        ),
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    # NOTE: argparse (not the dataclass default) governs the `python -m minisgl` CLI path — kwargs
    # below always carries use_pynccl into ServerArgs(**kwargs). PyNCCL is CUDA-only (pynccl.cu →
    # NVIDIA NCCL + apache-tvm-ffi), so default it OFF on ROCm: tp>1 then uses torch.distributed
    # backend="nccl" (→ RCCL). store_false means --disable-pynccl forces it off everywhere; on ROCm
    # it is already off without the flag (mirrors EngineConfig.use_pynccl's ROCm-aware default).
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        default=not is_rocm(),
        help="Disable PyNCCL for tensor parallelism (already the default on ROCm; CUDA-only path).",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--gdn-radix",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.gdn_radix,
        help="Recurrent-radix prefix cache for GDN/CCA hybrid models (reuse linear-attention state on "
        "prefix hits). ON by default; forces the synchronous scheduler loop, so pass --no-gdn-radix to "
        "keep overlap scheduling. Inert for non-recurrent models and under spec-decode / EP.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--chat-template",
        type=str,
        dest="chat_template",
        default=ServerArgs.chat_template,
        help="Override the served checkpoint's chat template: a path to a .jinja file or a literal "
        "jinja string. Applies to every render of the SERVED model (frontend + tokenizer workers); "
        "other tokenizers (drafters, calibrators) are untouched.",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        type=str,
        dest="chat_template_kwargs",
        default=ServerArgs.chat_template_kwargs,
        help="Server-level default chat_template_kwargs as a JSON object (e.g. "
        "'{\"enable_thinking\": false}'). Merged under each request's own kwargs; the request wins.",
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        dest="reasoning_parser",
        default=ServerArgs.reasoning_parser,
        choices=["auto", "none", "qwen3", "qwen", "deepseek_r1", "deepseek-r1", "glm",
                 "poolside_v1", "poolside"],
        help="Reasoning-content parser for thinking models: splits a completion on the model's "
        "reasoning close delimiter into reasoning_content + content on /v1/chat/completions. "
        "'auto' (default) DERIVES the delimiter pair from the served checkpoint's own chat "
        "template — it handles families whose markup is not <think>/</think> (Gemma-4 uses "
        "<|channel>thought … <channel|>) with no code change. Pass a family name to FORCE that "
        "pair (the escape hatch for a checkpoint whose template declares nothing), or 'none' to "
        "disable extraction entirely. The resolved pair and where it came from are logged at boot.",
    )

    # --- speculative decoding (off by default; see SPEC_DECODE.md) -------------------------------
    parser.add_argument(
        "--spec-algorithm",
        type=str,
        default=ServerArgs.spec_algorithm,
        choices=["none", "ngram", "mtp", "eagle3", "dflash", "tidar"],
        help="Speculative-decoding proposer. 'none' disables it; 'ngram' = prompt-lookup; "
        "'mtp' = the model's own appended next-token-prediction head (GLM-4.x / Qwen3.5); "
        "'eagle3' = a separate EAGLE3 draft checkpoint (--spec-draft-model-path); "
        "'dflash' = a separate DFlash block-diffusion draft checkpoint (--spec-draft-model-path); "
        "'tidar' = self-draft block-diffusion on the target (TiDAR; reads tidar_config.json from the "
        "served model dir — no separate checkpoint).",
    )
    parser.add_argument(
        "--spec-draft-model-path",
        "--spec-draft-model",
        type=str,
        dest="spec_draft_model_path",
        default=ServerArgs.spec_draft_model_path,
        help="EAGLE3/DFlash draft checkpoint path (HF repo id or local folder). Required for "
        "--spec-algorithm eagle3.",
    )
    parser.add_argument(
        "--spec-num-draft",
        type=int,
        default=ServerArgs.spec_num_draft,
        help="Draft tokens proposed per step (K); verify runs K+1 query positions per sequence.",
    )
    parser.add_argument(
        "--spec-ngram-max",
        type=int,
        default=ServerArgs.spec_ngram_max,
        help="Largest trailing n-gram window the prompt-lookup proposer matches on.",
    )
    parser.add_argument(
        "--spec-ngram-min",
        type=int,
        default=ServerArgs.spec_ngram_min,
        help="Smallest trailing n-gram window to fall back to.",
    )

    # Markovian-RSA server defaults (--rsa-n/k/t/tail-tokens/max-tokens/...). These populate
    # ServerArgs.rsa_defaults; a per-request `rsa` field patches them at call time.
    add_rsa_args(parser)

    # Parse arguments
    parsed = parser.parse_args(args)
    kwargs = parsed.__dict__.copy()

    # Fold the --rsa-* flags into a single RSAParams and drop the raw keys so ServerArgs(**kwargs)
    # (which only knows `rsa_defaults`) doesn't choke on them.
    kwargs["rsa_defaults"] = params_from_args(parsed)
    for _k in list(kwargs):
        if _k.startswith("rsa_") and _k != "rsa_defaults":
            del kwargs[_k]

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        hf = cached_load_hf_config(kwargs["model_path"])
        dtype_str = getattr(hf, "dtype", None) or getattr(hf, "torch_dtype", None)
        # Multimodal-wrapper configs (e.g. the Qwen3.5 GDN-hybrid 4B / 35B) carry the real
        # compute dtype in `text_config`, not at the top level — so top-level `.dtype` is None
        # there and the engine would crash on `self.dtype.itemsize`. Fall back to text_config,
        # then to bf16 (the weights ship bf16; only the MoE experts are int4).
        if dtype_str is None and (tc := getattr(hf, "text_config", None)) is not None:
            dtype_str = getattr(tc, "dtype", None) or getattr(tc, "torch_dtype", None)
        if dtype_str is None:
            dtype_str = "bfloat16"

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    dp_size = kwargs.pop("data_parallel_size")
    assert dp_size >= 1, f"--data-parallel-size must be >= 1, got {dp_size}"
    kwargs["dp_info"] = DpInfo(0, dp_size)
    if kwargs["enable_ep"]:
        assert dp_size > 1 or kwargs["tp_info"].size > 1, (
            "--enable-ep needs --data-parallel-size > 1 (DP+EP: experts across replicas) OR "
            "--tensor-parallel-size > 1 (EP-over-TP: experts across the TP ranks, vllm-style)")

    # Co-derive the CUDA-graph coverage and the admission cap so no decode batch size runs fully
    # eager. When a graph cap is explicitly requested BELOW the running cap, admit no more reqs than
    # the graph covers — otherwise decode batches in (cuda_graph_max_bs, max_running_req] fall back to
    # a fully-eager forward (the worst launch-overhead case). The complementary direction (an AUTO
    # graph cap RAISED to cover max_running_req) is handled in engine/graph._determine_cuda_graph_bs.
    # cuda_graph_max_bs==0 (graphs disabled) is left alone: that path is uniformly eager by choice, so
    # there is no coverage band to close.
    _cg = kwargs.get("cuda_graph_max_bs")
    if _cg is not None and _cg >= 1 and _cg < kwargs["max_running_req"]:
        kwargs["max_running_req"] = _cg

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
