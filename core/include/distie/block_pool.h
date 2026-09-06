#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace distie {

// Physical slot in the pre-allocated arena. Invalid IDs are negative.
using BlockId = std::int32_t;

// BlockPool pre-allocates a contiguous arena and hands out fixed-size
// blocks. Python only holds integer IDs, so the interpreter GC does not
// run on every KV-cache chunk.
//
// One block is an opaque byte slab. The inference engine decides how to
// lay out K/V tensors inside it (PagedAttention's "page").
class BlockPool {
 public:
  BlockPool(std::size_t num_blocks, std::size_t block_size_bytes);
  ~BlockPool();

  BlockPool(const BlockPool&) = delete;
  BlockPool& operator=(const BlockPool&) = delete;
  BlockPool(BlockPool&&) = delete;
  BlockPool& operator=(BlockPool&&) = delete;

  // All-or-nothing: if fewer than count blocks are free, the pool is
  // unchanged and Allocate throws std::runtime_error.
  std::vector<BlockId> Allocate(std::size_t count);

  // All-or-nothing: invalid or double-freed IDs throw std::invalid_argument
  // and no block is released.
  void Free(const std::vector<BlockId>& ids);

  void* MutableBlock(BlockId id);
  const void* Block(BlockId id) const;

  std::size_t num_blocks() const noexcept { return num_blocks_; }
  std::size_t block_size_bytes() const noexcept { return block_size_bytes_; }
  std::size_t num_free() const;
  std::size_t num_used() const;
  double utilization() const;

 private:
  void CheckInUseUnlocked(BlockId id) const;
  std::uint8_t* BlockPtrUnlocked(BlockId id) noexcept;
  const std::uint8_t* BlockPtrUnlocked(BlockId id) const noexcept;

  mutable std::mutex mu_;
  std::size_t num_blocks_;
  std::size_t block_size_bytes_;
  std::unique_ptr<std::uint8_t[]> arena_;
  std::vector<BlockId> free_list_;
  std::vector<std::uint8_t> in_use_;
};

}  // namespace distie
