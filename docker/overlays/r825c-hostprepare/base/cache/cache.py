from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import Type
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..model import Model, Config
    from ..modules import Attention
import weakref
from .mtp_window import MTP_KV_WINDOW as _MTP_KV_WINDOW
import logging
import os

logger = logging.getLogger(__name__)

_SLOTFIX_BUILD = "r1"
# Test hook: the first N state constructions raise, exercising the recovery path.
# If EXL3_SLOTFIX_TEST_FAULT_FILE is set, faults only fire while that path exists:
# the launcher's warmup generation would otherwise consume (and fail on) injected
# faults during boot. The gate touches the file after the model is serving.
_SLOTFIX_TEST_FAULTS = int(os.environ.get("EXL3_SLOTFIX_TEST_FAULTS", "0"))
_SLOTFIX_TEST_FAULT_FILE = os.environ.get("EXL3_SLOTFIX_TEST_FAULT_FILE")

class CacheLayer(ABC):
    """
    Abstract interface for one model layer's cache storage.

    Cache implementations provide the backing tensors and copy/update operations for attention KV pages or
    cache-like layer state. The higher-level Cache class owns one CacheLayer per attention layer and calls this
    interface without needing to know whether the concrete storage is unquantized, quantized, recurrent, or split
    across devices.
    """

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        **kwargs
    ):
        self.config = config
        self.attention = attention
        self.cache_id = cache_id
        self.max_num_tokens = max_num_tokens

    @abstractmethod
    def alloc(self, device: torch.device):
        pass

    @abstractmethod
    def free(self):
        pass

    @abstractmethod
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1) -> tuple:
        pass

    @abstractmethod
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        pass

    @abstractmethod
    def update_kv_direct(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        # Write new contiguous K/V rows (bsz, length, kv_heads, head_dim) into the paged cache
        # at positions cache_seqlens..+length, without materializing full dequantized layers in
        # the case of a quantized cache
        pass

    @abstractmethod
    def copy_page(self, source: CacheLayer, from_page: int, to_page: int, num_tokens: int):
        pass

    @abstractmethod
    def get_tensors(self):
        pass

    @abstractmethod
    def storage_size(self):
        pass

    @abstractmethod
    def overhead_size(self):
        pass

    @abstractmethod
    def tp_export(self, plan):
        pass


class Cache:

    def __init__(
        self,
        model: Model,
        max_num_tokens: int,
        layer_type: Type[CacheLayer] | None = None,
        max_batch_size: int = 16,
        max_history: int = 0,
        **kwargs
    ):
        """
        Create cache for model

        :param model:
            Model for which to create the cache. Once created, the cache is tied to the model. Loading the model
            will create cache tensors and unloading the model will destroy them. To delete the cache itself without
            deleting the reference to the model, use detach_from_model

        :param layer_type:
            Cache layer class, CacheLayer_fp16 or CacheLayer_quant. Only affects global-attn transformer cache layers

        :param max_num_tokens:
            Max number of total tokens in the cache. Must be a multiple of the page size (256). For use with the
            dynamic generator, this is the total number of tokens that can be allocated across concurrent jobs. For
            batched inference, seq_len * batch_size <= max_num_tokens

        :param max_batch_size:
            Max number of recurrent state slots (max supported batch size for recurrent models)

        :param max_history:
            For recurrent models, max number of past states to reserve space for per batch item. For speculative
            decoding on recurrent models, this should be equal to the number of draft tokens.

        :param k_bits:
            If layer_type == CacheLayer_quant, bits per element of the quantized keys tensor

        :param v_bits:
            If layer_type == CacheLayer_quant, bits per element of the quantized values tensor
        """
        self.model = model
        # Set by Model.load() / cleared by Model.unload(): cache tensors are allocated by the
        # model loader (per layer, on the layer's device), so a cache is only usable after a
        # load of the model it was attached to at load time.
        self.initialized = False
        self.config = model.config
        self.max_num_tokens = max_num_tokens

        # EXL3_MTP_KV_WINDOW (cache/mtp_window.py): the cache of an MTP draft model becomes a small
        # per-slot ring pool (max_batch_size slots) instead of a max_num_tokens twin of the main cache.
        # Unset: None, and the cache is built exactly as served
        self.mtp_window = None
        if _MTP_KV_WINDOW and getattr(model, "caps", {}).get("mtp_draft"):
            from .mtp_window import MTPWindow
            self.mtp_window = MTPWindow(_MTP_KV_WINDOW, max_batch_size)
            self.max_num_tokens = self.mtp_window.pool_tokens

        from .fp16 import CacheLayer_fp16
        self.layer_type = layer_type or CacheLayer_fp16
        # self.recurrent_layer_type = recurrent_layer_type or RecurrentLayer_fp16

        # Attach transformer cache layers
        cl = self.model.get_cache_layers()
        self.num_layers = len(cl)
        self.layers = {}
        for attn in cl:
            # Attention variants with a different cache geometry (MLA stores one latent plus one
            # shared rope key instead of per-head K/V) map the requested layer type to their own
            layer_type, layer_kwargs = (
                attn.cache_layer_type(self.layer_type, kwargs)
                if hasattr(attn, "cache_layer_type") else (self.layer_type, kwargs)
            )
            for instance in self.model.get_layer_instances(attn.layer_idx):
                self.layers[instance] = \
                    layer_type(self.config, attn, id(self), self.max_num_tokens, **layer_kwargs)
                if self.mtp_window is not None:
                    # Read by BC_Attention: the graph's sparse scan covers one ring cycle. It must
                    # reach past the dense/sparse threshold, or regime 1 would scan fewer pools
                    # than top-k selects
                    idx = getattr(attn, "qsa_indexer", None)
                    if idx is not None and self.mtp_window.scan_tokens <= idx.sparse_threshold():
                        raise ValueError(
                            f"EXL3_MTP_KV_WINDOW={self.mtp_window.window_tokens}: one ring cycle "
                            f"({self.mtp_window.scan_tokens} tokens) must exceed the QSA sparse "
                            f"threshold ({idx.sparse_threshold()}); use a larger window"
                        )
                    self.layers[instance].mtp_window_scan_tokens = self.mtp_window.scan_tokens

        # Attach recurrent (SWA/linear-attn) layers
        self.num_slots = max_batch_size
        self.max_history = max_history
        self.free_list = deque(range(self.num_slots))
        # Conservation bookkeeping for the slot pool; reported by the exhaustion assert.
        self._state_issued = 0
        self._state_reclaimed = 0
        self.recurrent_state_cls = model.recurrent_state_cls

        rl = self.model.get_recurrent_layers()
        self.num_layers = len(rl)
        self.recurrent_layers = {}
        for layer in rl:
            for instance in self.model.get_layer_instances(layer.layer_idx):
                self.recurrent_layers[instance] = layer.layer_state_cls(layer, max_batch_size, max_history, id(self))

        # Attach
        self.recurrent_instances = {}
        self.attach_to_model()

        if self.mtp_window is not None:
            print(
                f" -- EXL3_MTP_KV_WINDOW: draft cache {self.mtp_window.slots} slots x "
                f"{self.mtp_window.pages_per_slot} pages = {self.max_num_tokens} tokens "
                f"(requested {max_num_tokens})",
                flush = True,
            )


    def attach_to_model(self, model: Model = None):
        """
        Attach cache to model. Registering the cache with the model (done automatically by the Cache constructor)
        is necessary in order to tie loading of the model to allocation of cache tensors. Multiple caches can be
        attached to the same model.
        """
        if model is None:
            model = self.model

        model.cache_weakrefs[id(self)] = weakref.ref(self)

        cl = model.get_cache_layers()
        for module in cl:
            for instance in self.model.get_layer_instances(module.layer_idx):
                layer = self.layers[instance]
                assert layer not in module.cache_layers, "Cannot attach cache twice to the same model."
                module.cache_layers.append(layer)

        rl = model.get_recurrent_layers()
        for module in rl:
            for instance in self.model.get_layer_instances(module.layer_idx):
                layer = self.recurrent_layers[instance]
                assert layer not in module.recurrent_layers, "Cannot attach cache twice to the same model."
                module.recurrent_layers.append(layer)


    def detach_from_model(self, model: Model = None):
        """
        Detach cache from model. Must be called if you want to delete a cache without deleting the model.
        """
        if model is None:
            model = self.model

        del model.cache_weakrefs[id(self)]

        cl = model.get_cache_layers()
        for module in cl:
            for instance in self.model.get_layer_instances(module.layer_idx):
                layer = self.layers[instance]
                module.cache_layers.remove(layer)

        rl = model.get_recurrent_layers()
        for module in rl:
            for instance in self.model.get_layer_instances(module.layer_idx):
                layer = self.recurrent_layers[instance]
                module.recurrent_layers.remove(layer)
        self.recurrent_instances = {}


    def get_layer(
        self,
        idx: int,
        cache_seqlens:
        torch.Tensor,
        block_table: torch.Tensor,
        sliding_window: int = -1,
        instance = None,
    ) -> tuple:
        """
        Return K/V tensors for cache layer, either reference to statically allocated tensors or temporary 
        tensors with dequantized keys/values in the case of a quantized cache
        """
        instance = instance or 0
        return self.layers[idx, instance].get_kv(cache_seqlens, block_table, sliding_window)


    def update_layer(
        self,
        idx: int,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
        instance: None,
    ):
        """
        Update cache layer. Quantizes updated keys/values in the case of a quantized cache, else this will
        be a no-op since attention will have updated a reference to the static paged cache.
        """
        instance = instance or 0
        return self.layers[idx, instance].update_kv(cache_seqlens, block_table, k, v, length)


    def update_layer_direct(
        self,
        idx: int,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
        instance = None,
    ):
        """
        Write new contiguous K/V rows (bsz, length, kv_heads, head_dim) into the cache at positions
        cache_seqlens..+length. Unlike get_layer/update_layer, a quantized cache quantizes the new
        rows in place and never materializes full dequantized layer tensors.
        """
        instance = instance or 0
        return self.layers[idx, instance].update_kv_direct(cache_seqlens, block_table, k, v, length)


    def copy_page(
        self,
        target: Cache,
        from_page: int,
        to_page: int,
        num_tokens: int,
    ):
        """
        Copy one cache page. Only used for partial page reuse in prompt caching.
        """
        assert target == self or (not target.model.loaded_tp and not self.model.loaded_tp), \
            "Cannot copy pages between TP and non-TP caches, or between distinct TP caches."
        assert target.num_layers == self.num_layers
        if not self.model.loaded_tp:
            for instance, src in self.layers.items():
                dst = target.layers[instance]
                assert type(src) is type(dst)
                dst.copy_page(src, from_page, to_page, num_tokens)
        else:
            self.model.tp_cache_page_copy(id(self), from_page, to_page, num_tokens)


    def get_all_tensors(self):
        tensors = []
        for layer in self.layers.values():
            tensors += layer.get_tensors()
        return tensors


    def alloc_state(self, layer_instance, device: torch.Device):
        """
        Create recurrent state layer tensors on device
        """
        self.recurrent_layers[layer_instance].alloc()


    def free_state(self, layer_instance):
        """
        Remove recurrent state layer tensors from device
        """
        self.recurrent_layers[layer_instance].free()


    def get_new_state(self):
        """
        Allocate a new, empty state
        """
        assert len(self.free_list) > 0, \
            f"Cannot create new state: no available slots (issued={self._state_issued} reclaimed={self._state_reclaimed})"
        handle = self.free_list.popleft()
        self._state_issued += 1
        try:
            return self._make_state(handle, 0, clear = True)
        except BaseException:
            # The handle was popped already; without this the slot is burned permanently.
            # Whatever partial state the failed construction left in the slot's planes is
            # overwritten by the next allocation's clear=/unstash.
            self.free_list.appendleft(handle)
            self._state_reclaimed += 1
            logger.warning("state construction failed; returned slot %d to the pool", handle)
            raise


    def get_test_state(self, position: int):
        """
        Allocate a new state for given position, for testing/benchmarking purposes. If position > 0 this will
        not be a valid recurrent state for the model.
        """
        assert len(self.free_list) > 0, \
            f"Cannot create new state: no available slots (issued={self._state_issued} reclaimed={self._state_reclaimed})"
        handle = self.free_list.popleft()
        self._state_issued += 1
        try:
            return self._make_state(handle, position, clear = True, test_state = True)
        except BaseException:
            self.free_list.appendleft(handle)
            self._state_reclaimed += 1
            logger.warning("state construction failed; returned slot %d to the pool", handle)
            raise


    def new_from_stashed(self, stashed_state, position):
        """
        Allocate a new state from a stashed state
        """
        assert len(self.free_list) > 0, \
            f"Cannot create new state: no available slots (issued={self._state_issued} reclaimed={self._state_reclaimed})"
        handle = self.free_list.popleft()
        self._state_issued += 1
        try:
            return self._make_state(handle, position, clear = False, stashed = stashed_state)
        except BaseException:
            self.free_list.appendleft(handle)
            self._state_reclaimed += 1
            logger.warning("state construction failed; returned slot %d to the pool", handle)
            raise


    def _make_state(self, handle, position, **kwargs):
        global _SLOTFIX_TEST_FAULTS
        if _SLOTFIX_TEST_FAULTS and (
            _SLOTFIX_TEST_FAULT_FILE is None or os.path.exists(_SLOTFIX_TEST_FAULT_FILE)
        ):
            _SLOTFIX_TEST_FAULTS -= 1
            raise RuntimeError("slotfix-r1 injected state-construction fault")
        return self.recurrent_state_cls(self, handle, position, **kwargs)


    def release_state(self, state):
        """
        Return state to the pool
        """
        if state.slot in self.free_list:
            # A double release would hand one slot to two live states and silently
            # corrupt both; refuse it and keep the pool consistent.
            logger.error("state slot %d released twice; ignoring", state.slot)
            return
        self.free_list.appendleft(state.slot)
        self._state_reclaimed += 1


    def reset_states(self):
        """
        Return every state slot to the pool. A Generator calls this when it takes ownership of the cache.
        """
        runtime = getattr(self, "_ls_prefill_runtime", None)
        if runtime is not None:
            runtime.job.generator.clear_queue()
        self.free_list = deque(range(self.num_slots))


    def get_recurrent_layer(
        self,
        layer_instance: tuple,
    ):
        return self.recurrent_layers[layer_instance]


    def get_all_recurrent_layers(self):
        return self.recurrent_layers
