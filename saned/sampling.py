"""SANED 的同事件跨被试 batch 采样器。"""

from __future__ import annotations

import random
from collections import defaultdict
from math import ceil

from torch.utils.data import Sampler

from .data import SampleRef


class EventBalancedBatchSampler(Sampler[list[int]]):
    """按 event 分组并保证每个组含多个被试。

    每个事件的引用先按被试打乱，再切成不超过 ``max_subjects_per_event``
    的正样本组；正样本组随后被装入 batch。这样一个 epoch 会覆盖所有训练
    引用，同时 InfoNCE 至少能为每个非单例组找到同事件正样本。
    """

    def __init__(
        self,
        refs: list[SampleRef],
        batch_size: int,
        seed: int,
        balance_ids: dict[int, str] | None = None,
        max_subjects_per_event: int = 4,
    ) -> None:
        if batch_size < 4:
            raise ValueError(
                "SANED contrastive batches require batch_size >= 4 so each batch "
                "can contain cross-subject pairs from at least two events"
            )
        if max_subjects_per_event < 2:
            raise ValueError("max_subjects_per_event must be at least 2")
        self.refs = list(refs)
        self.batch_size = batch_size
        self.seed = seed
        # Reserve two places for a cross-subject pair from a second event.
        self.max_subjects_per_event = min(max_subjects_per_event, batch_size - 2)
        self.balance_ids = balance_ids or {}
        self._epoch = 0
        self.groups: dict[int, list[int]] = defaultdict(list)
        for index, ref in enumerate(self.refs):
            self.groups[int(ref.center)].append(index)

    def _positive_groups(self, rng: random.Random) -> list[tuple[int, list[int]]]:
        centers_by_event: dict[str | int, list[int]] = defaultdict(list)
        for center in self.groups:
            centers_by_event[self.balance_ids.get(center, center)].append(center)
        selected_centers = [rng.choice(centers) for centers in centers_by_event.values()]

        queues: dict[int, list[list[int]]] = {}
        for center in selected_centers:
            local = list(self.groups[center])
            rng.shuffle(local)
            n_groups = ceil(len(local) / self.max_subjects_per_event)
            if len(local) < 2 * n_groups:
                raise ValueError(
                    f"Event {center} has {len(local)} references, which cannot be "
                    f"partitioned into cross-subject groups of size 2 to "
                    f"{self.max_subjects_per_event}; increase batch_size"
                )
            base_size, remainder = divmod(len(local), n_groups)
            group_sizes = [base_size + (i < remainder) for i in range(n_groups)]
            chunks = []
            offset = 0
            for size in group_sizes:
                chunks.append(local[offset : offset + size])
                offset += size
            queues[center] = chunks

        ordered: list[tuple[int, list[int]]] = []
        while queues:
            active = list(queues)
            rng.shuffle(active)
            if ordered and len(active) > 1 and active[0] == ordered[-1][0]:
                replacement = next(
                    i
                    for i, center in enumerate(active[1:], 1)
                    if center != ordered[-1][0]
                )
                active[0], active[replacement] = active[replacement], active[0]
            for center in active:
                ordered.append((center, queues[center].pop()))
                if not queues[center]:
                    del queues[center]
        return ordered

    def _batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch)
        event_groups = self._positive_groups(rng)
        # Keep group identities until tail repair completes. Flattening early
        # would make it impossible to preserve cross-subject positive groups.
        batches: list[list[tuple[int, list[int]]]] = []
        batch: list[tuple[int, list[int]]] = []
        batch_size = 0
        for center, group in event_groups:
            if batch and batch_size + len(group) > self.batch_size:
                if len({batch_center for batch_center, _ in batch}) < 2:
                    raise ValueError(
                        "Cannot form bounded contrastive batches from the available "
                        "event groups; use a larger batch_size"
                    )
                batches.append(batch)
                batch, batch_size = [], 0
            batch.append((center, group))
            batch_size += len(group)
        if batch:
            batches.append(batch)
        if not batches:
            raise ValueError("SANED batches require at least two distinct events with cross-subject pairs")

        self._repair_single_event_tail(batches)
        flat_batches = [[index for _, group in packed for index in group] for packed in batches]
        for packed, indices in zip(batches, flat_batches, strict=True):
            if len(indices) > self.batch_size:
                raise RuntimeError("Internal error: contrastive batch exceeds the configured batch size")
            if len({center for center, _ in packed}) < 2:
                raise RuntimeError("Internal error: contrastive batch contains fewer than two events")
        return flat_batches

    def _repair_single_event_tail(self, batches: list[list[tuple[int, list[int]]]]) -> None:
        """Rebalance a one-event tail batch without dropping training references.

        InfoNCE needs a different-event negative in every batch. Greedy packing
        can leave one valid cross-subject group at the end, especially when a
        larger batch size is used. Move a group from a prior batch only if that
        donor batch still retains at least two distinct events.
        """
        tail = batches[-1]
        tail_centers = {center for center, _ in tail}
        if len(tail_centers) >= 2:
            return
        if len(batches) < 2:
            raise ValueError(
                "Cannot form a contrastive batch with two events from the available event groups"
            )

        tail_size = sum(len(group) for _, group in tail)
        tail_center = next(iter(tail_centers))
        for donor_batch in reversed(batches[:-1]):
            for group_index in range(len(donor_batch) - 1, -1, -1):
                center, group = donor_batch[group_index]
                if center == tail_center or tail_size + len(group) > self.batch_size:
                    continue
                donor_remaining_centers = {
                    donor_center
                    for index, (donor_center, _) in enumerate(donor_batch)
                    if index != group_index
                }
                if len(donor_remaining_centers) < 2:
                    continue
                tail.append(donor_batch.pop(group_index))
                return

        raise ValueError(
            "Cannot rebalance the final contrastive batch to contain two events "
            "without violating batch_size; adjust batch_size or event grouping"
        )

    def __len__(self) -> int:
        return len(self._batches(self._epoch))

    def __iter__(self):
        batches = self._batches(self._epoch)
        self._epoch += 1
        yield from batches
