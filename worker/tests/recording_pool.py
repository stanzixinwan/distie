from __future__ import annotations


class RecordingPool:
    """In-memory BlockAllocator for tests. Same methods as distie_core.BlockPool."""

    def __init__(self, num_blocks: int = 16, block_size: int = 8) -> None:
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))
        self._data = {i: bytearray(block_size) for i in range(num_blocks)}
        self.allocated: list[list[int]] = []
        self.freed: list[list[int]] = []

    def allocate(self, count: int) -> list[int]:
        if count > len(self._free):
            raise RuntimeError(f"out of blocks: requested {count}, free {len(self._free)}")
        ids = [self._free.pop() for _ in range(count)]
        self.allocated.append(list(ids))
        return ids

    def free(self, block_ids: list[int]) -> None:
        self.freed.append(list(block_ids))
        for block_id in block_ids:
            self._free.append(block_id)

    def block_view(self, block_id: int) -> memoryview:
        return memoryview(self._data[block_id])

    @property
    def num_free(self) -> int:
        return len(self._free)
