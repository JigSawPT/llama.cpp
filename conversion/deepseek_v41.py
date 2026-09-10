from __future__ import annotations

import logging
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

        self.gguf_writer.add_tensor("engram_token_map",   np.array(token_map, dtype=np.int32))
        self.gguf_writer.add_tensor("engram_primes",      flat)
        self.gguf_writer.add_tensor("engram_offsets",     build_offsets(primes))
        self.gguf_writer.add_tensor("engram_multipliers",
                                    build_multipliers(self._engram_layer_ids,
                                                      h["engram_max_ngram_size"], n_compressed))
        logger.info("engram: hash tables written, compressed vocab %d, primes sum matches both tables",
                    n_compressed)

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
                # on the host, and dequantizing 384M rows would need 196 GiB per layer
                return [(out, data_torch.view(torch.uint8))]
            return [(out, data_torch)]
        return super().modify_tensors(data_torch, name, bid)
