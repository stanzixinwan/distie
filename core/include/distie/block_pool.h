#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace distie {

// Index of one KV page. Invalid IDs are negative.
using BlockId = std::int32_t;

// BlockPool is control-plane only: it tracks which KV page IDs are free.
// The K/V tensors live in torch slabs indexed by the same IDs, so this
// class never owns tensor memory.
class BlockPool {
 public:
  explicit BlockPool(std::size_t num_blocks);
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

  std::size_t num_blocks() const noexcept { return num_blocks_; }
  std::size_t num_free() const;
  std::size_t num_used() const;
  double utilization() const;

 private:
  mutable std::mutex mu_;
  std::size_t num_blocks_;
  std::vector<BlockId> free_list_;
  std::vector<std::uint8_t> in_use_;
};

}  // namespace distie
