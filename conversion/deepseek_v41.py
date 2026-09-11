from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

import gguf

from .base import ModelBase
from .deepseek import DeepseekV4Model

logger = logging.getLogger(__name__)


@ModelBase.register("DeepseekV41ForCausalLM")
class DeepseekV41Model(DeepseekV4Model):
    """DeepSeek-V4.1-Flash. Same modules as V4 plus conditional memory (engram).

    V4 already handles the hard parts: MXFP4 expert packing, hyper-connections, compressor,
    indexer and NEXTN layers. V4.1 adds the engram (two tables, ~384M rows each) and a
    hierarchical indexer that picks blocks before positions.
    """

    model_arch = gguf.MODEL_ARCH.DEEPSEEK41

    convert_engram = False

    # The V4.1 MTP block is not the V4 one: it has main_proj/main_norm/markov_head/confidence_head
    # and none of the nextn.e_proj/h_proj/enorm/hnorm that generate_extra_tensors looks for.
    # --mtp stays refused; the block is a DSpark draft and exports with --dspark (see below).
    supports_mtp_export = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["num_hash_layers"] = 0  # V4 gate tid2eid table, absent in V4.1
        self._engram_layer_ids = list(self.hparams.get("engram_layer_ids", []))
        self._engram_num_embeddings = list(self.hparams.get("engram_num_embeddings", []))

    def index_tensors(self, remote_hf_model_id: str | None = None):
        # ModelBase.__init__ calls this before TextModel.__init__ promotes text_config to the
        # root of hparams. V4 had no text_config, so this problem did not exist there.
        if "num_hidden_layers" not in self.hparams and "text_config" in self.hparams:
            self.hparams = {**self.hparams, **self.hparams["text_config"]}
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    _ENGRAM_RAW = ".engram.embed."
    _BYTES_1 = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)

    def _engram_memmap(self, name: str):
        """A janela do safetensors onde a tabela vive, como memmap de bytes.

        Devolve None se o tensor nao for encontrado, e nesse caso o caminho normal segue --
        mais lento e a materializar, mas correcto."""
        import glob
        import io
        import json
        import os
        import struct

        if self._st_index is None:
            idx = glob.glob(os.path.join(self.dir_model, "*.safetensors.index.json"))
            if not idx:
                return None
            with io.open(idx[0], encoding="utf-8") as fh:
                self._st_index = json.load(fh)["weight_map"]

        ficheiro = self._st_index.get(name)
        if ficheiro is None:
            return None

        caminho = os.path.join(self.dir_model, ficheiro)
        cab = self._st_headers.get(caminho)
        if cab is None:
            with open(caminho, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                cab = (json.loads(fh.read(n)), 8 + n)
            self._st_headers[caminho] = cab
        meta, base = cab

        e = meta[name]
        inicio, fim = e["data_offsets"]
        # bytes crus: a forma logica e [linhas, colunas], um byte por elemento
        forma = tuple(e["shape"])
        n_bytes = fim - inicio
        if n_bytes != int(np.prod(forma)):
            logger.warning("%s: %d bytes para uma forma de %s, memmap ignorado", name, n_bytes, forma)
            return None

        return np.memmap(caminho, dtype=np.uint8, mode="r", offset=base + inicio, shape=forma)

    _st_index = None
    _st_headers: dict = {}

    def keeps_raw_dtype(self, name: str) -> bool:
        # the engram table travels as raw fp8 bytes; f32 would be 393 GiB per layer
        return self._ENGRAM_RAW in name

    def dequant_model(self):
        # the inherited pass pairs every .scale with its fp8 .weight and consumes both. For the
        # engram table that would delete the scale the C++ requires and dequantize 384M rows.
        held = {k: v for k, v in self.model_tensors.items() if self._ENGRAM_RAW in k}
        for k in held:
            del self.model_tensors[k]
        super().dequant_model()
        self.model_tensors.update(held)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        h = self.hparams
        w = self.gguf_writer

        for key in ("candidate_block_size", "candidate_topk_blocks", "candidate_source_layer_id"):
            if key in h:
                w.add_uint32(f"deepseek41.attention.{key}", int(h[key]))

        for key in ("kv_source_layer_ids", "index_source_layer_ids", "dspark_target_layer_ids"):
            if key in h:
                w.add_array(f"deepseek41.{key}", [int(v) for v in h[key]])

        if "scoring_func" in h:
            w.add_string("deepseek41.expert_scoring_func", str(h["scoring_func"]))

        w.add_bool("deepseek41.engram.present", bool(self.convert_engram))
        if self._engram_layer_ids:
            w.add_array("deepseek41.engram.layer_ids", [int(v) for v in self._engram_layer_ids])
        if self._engram_num_embeddings:
            w.add_array("deepseek41.engram.num_embeddings", [int(v) for v in self._engram_num_embeddings])
        for key in ("engram_head_dim", "engram_n_heads", "engram_max_ngram_size",
                    "engram_vocab_size", "engram_compressed_vocab_size", "engram_pad_token_id"):
            if key in h:
                w.add_uint32(f"deepseek41.{key}", int(h[key]))

        if not self.convert_engram:
            # zeroing the engram is the exact identity of the module (key 0 -> value 0), but the
            # model that comes out does not produce the same output as the original
            logger.warning("engram not converted: %s rows left out; output will differ from the original",
                           " + ".join(f"{n:,}" for n in self._engram_num_embeddings) or "?")

    # V4.1 indexer keys. V4 built them through an indexer compressor; V4.1 projects them
    # directly, so these two have no V4 equivalent.
    _V41_EXTRA = {
        "attn.indexer.wk.weight":     (gguf.MODEL_TENSOR.INDEXER_ATTN_K, ".weight"),
        "attn.indexer.k_norm.weight": (gguf.MODEL_TENSOR.INDEXER_K_NORM, ".weight"),
    }

    def _map_dsv4_tensor_name(self, name: str, bid: int | None):
        if bid is not None:
            suffix = name.split(f"layers.{bid}.", 1)[-1]
            hit = self._V41_EXTRA.get(suffix)
            if hit is not None:
                return hit
        return super()._map_dsv4_tensor_name(name, bid)

    def _write_engram_tables(self) -> None:
        """The four hash tables. They come from the tokenizer, so they are built here: the C++
        side would need Unicode normalization and a primality test to rebuild them."""
        from transformers import AutoTokenizer

        from .deepseek_v41_engram import (build_compressed_token_map, build_multipliers,
                                          build_offsets, build_primes)

        h = self.hparams
        tok = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        token_map, n_compressed = build_compressed_token_map(tok)
        if n_compressed != h["engram_compressed_vocab_size"]:
            # every multiplier derives from this size, so a mismatch rehashes the whole table
            raise ValueError(f"compressed vocab {n_compressed} != {h['engram_compressed_vocab_size']}")

        primes = build_primes(self._engram_layer_ids, h["engram_max_ngram_size"],
                              h["engram_n_heads"], h["engram_vocab_size"])
        flat = primes.reshape(primes.shape[0], -1)
        for e, rows in enumerate(self._engram_num_embeddings):
            if int(flat[e].sum()) != int(rows):
                raise ValueError(f"engram layer {e}: primes sum to {flat[e].sum()}, table has {rows}")

        # the loader asks for these with a .weight suffix, like every other tensor
        self.gguf_writer.add_tensor("engram_token_map.weight",   np.array(token_map, dtype=np.int32))
        self.gguf_writer.add_tensor("engram_primes.weight",      flat)
        self.gguf_writer.add_tensor("engram_offsets.weight",     build_offsets(primes))
        self.gguf_writer.add_tensor("engram_multipliers.weight",
                                    build_multipliers(self._engram_layer_ids,
                                                      h["engram_max_ngram_size"], n_compressed))
        logger.info("engram: hash tables written, compressed vocab %d, primes sum matches both tables",
                    n_compressed)


    def prepare_tensors(self):
        super().prepare_tensors()
        # last, on purpose: these two tables are read a few rows at a time by the host, and
        # the loader prefetches from the start of the file. Putting them at the front means
        # the prefetch pulls in 189 GiB nobody needs and evicts the weights that matter.
        self._write_engram_rows()

    def _write_engram_rows(self) -> None:
        """The two lookup tables, byte for byte.

        They cannot go through prepare_tensors: nothing there would stop quantize() from
        turning the fp8 bytes into floats, which both doubles the size and destroys the
        contents -- and the loader only checks shapes, never types, so it would load quietly
        and produce garbage. add_tensor with an int8 view maps to I8, one byte per element,
        which is exactly what the C++ side reads."""
        for bid in self._engram_layer_ids:
            for sufixo, chave in (("weight", gguf.MODEL_TENSOR.ENGRAM_EMBED),
                                  ("scale",  gguf.MODEL_TENSOR.ENGRAM_EMBED_SCALE)):
                origem = f"layers.{bid}.engram.embed.{sufixo}"
                mm = self._engram_memmap(origem)
                if mm is None:
                    raise ValueError(f"{origem}: sem memmap, e esta tabela nao pode ser materializada")
                out = self._format_dsv4_tensor_name(chave, int(bid), ".weight")
                self.gguf_writer.add_tensor(out, mm.view(np.int8))
                logger.info("engram_embd_raw %s -> %s: %.1f GiB as I8, byte for byte",
                            origem, out, mm.nbytes/2**30)

    def generate_extra_tensors(self):
        if self.convert_engram and not getattr(self, "_engram_tables_done", False):
            self._engram_tables_done = True
            self._write_engram_tables()
        return super().generate_extra_tensors()

    @classmethod
    def filter_tensors(cls, item):
        name = item[0]
        # image_start/end/newline are vision markers with no prefix, and they follow vision out
        if name.startswith(("vision.", "aligner.")) or name in ("image_start", "image_end", "image_newline"):
            return None
        # bias_vl is the vision-language bias on the MoE gate: unused in a text-only model
        if name.endswith("ffn.gate.bias_vl"):
            return None
        if ".engram." in name and not cls.convert_engram:
            return None
        # the two lookup tables are written directly in generate_extra_tensors; letting them
        # into the tensor loop would upcast them to float32 before modify_tensors even runs
        if cls._ENGRAM_RAW in name:
            return None
        return super().filter_tensors(item)

    _ENGRAM_MAP = {
        "engram.embed.weight": (gguf.MODEL_TENSOR.ENGRAM_EMBED,       ".weight"),
        "engram.embed.scale":  (gguf.MODEL_TENSOR.ENGRAM_EMBED_SCALE, ".weight"),
        "engram.q_weight":     (gguf.MODEL_TENSOR.ENGRAM_Q,           ".weight"),
        "engram.k_weight":     (gguf.MODEL_TENSOR.ENGRAM_K,           ".weight"),
        "engram.wkv.weight":   (gguf.MODEL_TENSOR.ENGRAM_WKV,         ".weight"),
    }

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if ".engram." in name:
            if not self.convert_engram:
                return []
            if name.endswith(".scale") and "embed" not in name:
                return []  # wkv scale is consumed by the fp8 dequantization, like the rest of V4
            suffix = name.split(f"layers.{bid}.", 1)[-1]
            key, tail = self._ENGRAM_MAP[suffix]
            out = self._format_dsv4_tensor_name(key, bid, tail)
            if "embed" in name:
                # the table stays as raw bytes: 24 rows per token are gathered and dequantized
                # on the host, and dequantizing 384M rows would need 393 GiB per layer.
                # keeps_raw_dtype kept the storage intact, so the view is over fp8, not floats.
                # dtype comes off the meta tensor; element_size() is a method, and asking a
                # lazy tensor for one materializes it -- 91 GiB, which is what this guard is
                # here to prevent in the first place
                # itemsize belongs to the dtype, not the tensor, so asking does not
                # materialize anything -- unlike element_size(), which is a method
                if data_torch.dtype.itemsize != 1:
                    raise ValueError(f"{name}: expected raw fp8 bytes, got {data_torch.dtype}")
                # tofile on a lazy tensor materializes it first, and this one is 91 GiB. A
                # memmap over the source file is an ndarray the writer can walk without
                # allocating anything: the bytes go out exactly as they came in.
                mm = self._engram_memmap(name)
                if mm is not None:
                    logger.info("%s: %.1f GiB streamed by memmap, not materialized",
                                name, mm.nbytes/2**30)
                    return [(out, mm)]
                return [(out, data_torch.view(torch.uint8))]
            return [(out, data_torch)]
        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("DeepseekV41DSparkModel")
class DeepseekV41DSparkModel(DeepseekV41Model):
    """The V4.1 DSpark draft: three full V4.1 blocks under mtp.*, exported as a DFLASH model.
    Same layout as DeepseekV4DSparkModel with the V4.1 names for the heads. The block has no
    hc_head weights (V4.1 collapses with the mix the last ffn produced) and no per-head q norm;
    dsv41_semantics tells the C++ side to run it like the main V4.1 model.
    Like every DSpark stage, only the first stage has main_proj/main_norm and only the last has
    norm/markov_head/confidence_head; the reference (DSparkBlock.__init__) builds them that way.
    """
    model_arch = gguf.MODEL_ARCH.DFLASH

    _DSPARK_ROOT_MAP: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
        "main_proj.weight":            (gguf.MODEL_TENSOR.FC,               ".weight"),
        "main_norm.weight":            (gguf.MODEL_TENSOR.ENC_OUTPUT_NORM,  ".weight"),
        "markov_head.embed.weight":    (gguf.MODEL_TENSOR.DSPARK_MARKOV_W1, ".weight"),
        "markov_head.head.weight":     (gguf.MODEL_TENSOR.DSPARK_MARKOV_W2, ".weight"),
        "confidence_head.proj.weight": (gguf.MODEL_TENSOR.DSPARK_CONF_PROJ, ".weight"),
    }
    # main_proj.scale pairs with main_proj.weight in dequant_model; norm.weight is the decoder
    # final norm, mapped by the V4 root rule
    _DSPARK_ROOT_NAMES = ("main_proj.scale", "norm.weight")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block_count = 1 + max(
            int(match.group(1)) for name in self.model_tensors
            if (match := re.match(r"layers\.(\d+)\.", name))
        )
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        h = self.hparams
        h["compress_ratios"] = [0] * self.block_count
        # the draft routes over its own, smaller pool (128 experts, 3 used); the MXFP4 writer
        # and expert_count read these two keys
        h["n_routed_experts"] = h["dspark_n_routed_experts"]
        h["num_experts_per_tok"] = h["dspark_num_experts_per_tok"]
        # no engram in the draft: prepare_tensors streams the tables of these layers
        self._engram_layer_ids = []
        self._engram_num_embeddings = []

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith("mtp."):
            return None
        return super().filter_tensors((cls._rekey_mtp_tensor_name(name), gen))

    @classmethod
    def _rekey_mtp_tensor_name(cls, name: str) -> str:
        match = re.match(r"mtp\.(\d+)\.(.+)$", name)
        if match is None:
            raise ValueError(f"Unexpected DSpark tensor {name!r}")
        stage, rest = match.group(1), match.group(2)
        if rest in cls._DSPARK_ROOT_MAP or rest in cls._DSPARK_ROOT_NAMES:
            return rest
        return f"layers.{stage}.{rest}"

    def _map_dsv4_tensor_name(self, name: str, bid: int | None):
        if name in self._DSPARK_ROOT_MAP:
            return self._DSPARK_ROOT_MAP[name]
        return super()._map_dsv4_tensor_name(name, bid)

    def set_vocab(self):
        if self.target_model_dir is None:
            raise ValueError("DeepSeek-V4.1 DSpark requires --target-model-dir with the target tokenizer")
        original_dir = self.dir_model
        try:
            self.dir_model = self.target_model_dir
            super().set_vocab()
        finally:
            self.dir_model = original_dir
        self.gguf_writer.add_mask_token_id(self.hparams["dspark_noise_token_id"])

    def set_gguf_parameters(self):
        # the V4 keys only: the V4.1 additions (engram, indexer sources) describe the target
        DeepseekV4Model.set_gguf_parameters(self)
        h = self.hparams
        self.gguf_writer.add_block_size(h["dspark_block_size"])
        # the reference takes h.mean(dim=2) BEFORE running a target layer (Transformer.forward),
        # and t_layer_inp[il] in deepseek4.cpp is the input of layer il: the ids go through as
        # they are. V4 takes the mean after the layer, which is why its exporter adds one.
        self.gguf_writer.add_target_layers([int(v) for v in h["dspark_target_layer_ids"]])
        self.gguf_writer.add_bool(f"{gguf.MODEL_ARCH_NAMES[self.model_arch]}.dsv41_semantics", True)
