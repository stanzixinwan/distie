#include "distie/block_pool.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace distie {
namespace {

bool MemoryLogEnabled() {
  const char* env = std::getenv("DISTIE_MEMORY_LOG");
  return env != nullptr && env[0] == '1' && env[1] == '\0';
}

void MemoryLog(const std::string& msg) {
  if (MemoryLogEnabled()) {
    std::fprintf(stderr, "[distie.memory] %s\n", msg.c_str());
  }
}

}  // namespace

BlockPool::BlockPool(std::size_t num_blocks) : num_blocks_(num_blocks) {
  if (num_blocks == 0) {
    throw std::invalid_argument("num_blocks must be > 0");
  }
  if (num_blocks > static_cast<std::size_t>(std::numeric_limits<BlockId>::max())) {
    throw std::invalid_argument("num_blocks exceeds BlockId range");
  }

  in_use_.assign(num_blocks, 0);
  free_list_.reserve(num_blocks);
  for (std::size_t i = num_blocks; i > 0; --i) {
    free_list_.push_back(static_cast<BlockId>(i - 1));
  }

  MemoryLog("pool constructed num_blocks=" + std::to_string(num_blocks));
}

BlockPool::~BlockPool() {
  MemoryLog("pool destroyed used=" + std::to_string(num_blocks_ - free_list_.size()));
}

std::vector<BlockId> BlockPool::Allocate(std::size_t count) {
  if (count == 0) {
    return {};
  }

  std::lock_guard<std::mutex> lock(mu_);
  if (count > free_list_.size()) {
    MemoryLog("allocate failed requested=" + std::to_string(count) +
              " free=" + std::to_string(free_list_.size()));
    throw std::runtime_error("out of blocks: requested " + std::to_string(count) +
                             ", free " + std::to_string(free_list_.size()));
  }

  std::vector<BlockId> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const BlockId id = free_list_.back();
    free_list_.pop_back();
    in_use_[static_cast<std::size_t>(id)] = 1;
    out.push_back(id);
  }
  return out;
}

void BlockPool::Free(const std::vector<BlockId>& ids) {
  if (ids.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mu_);

  // Validate first so a bad ID cannot partially mutate the pool.
  std::vector<std::uint8_t> seen(num_blocks_, 0);
  for (BlockId id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= num_blocks_) {
      throw std::invalid_argument("block id out of range: " + std::to_string(id));
    }
    const std::size_t idx = static_cast<std::size_t>(id);
    if (seen[idx] != 0) {
      throw std::invalid_argument("duplicate block id in free(): " + std::to_string(id));
    }
    seen[idx] = 1;
    if (in_use_[idx] == 0) {
      throw std::invalid_argument("double free of block " + std::to_string(id));
    }
  }

  for (BlockId id : ids) {
    const std::size_t idx = static_cast<std::size_t>(id);
    in_use_[idx] = 0;
    free_list_.push_back(id);
  }
}

std::size_t BlockPool::num_free() const {
  std::lock_guard<std::mutex> lock(mu_);
  return free_list_.size();
}

std::size_t BlockPool::num_used() const {
  std::lock_guard<std::mutex> lock(mu_);
  return num_blocks_ - free_list_.size();
}

double BlockPool::utilization() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<double>(num_blocks_ - free_list_.size()) /
         static_cast<double>(num_blocks_);
}

}  // namespace distie
