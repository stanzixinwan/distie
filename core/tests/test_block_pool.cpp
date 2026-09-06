#include "distie/block_pool.h"

#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using distie::BlockId;
using distie::BlockPool;

TEST(BlockPool, RejectsEmptyConfig) {
  EXPECT_THROW(BlockPool(0, 16), std::invalid_argument);
  EXPECT_THROW(BlockPool(8, 0), std::invalid_argument);
}

TEST(BlockPool, AllocatesSequentialIds) {
  BlockPool pool(4, 8);
  EXPECT_EQ(pool.num_free(), 4u);
  EXPECT_DOUBLE_EQ(pool.utilization(), 0.0);

  const std::vector<BlockId> ids = pool.Allocate(3);
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
  EXPECT_EQ(ids[2], 2);
  EXPECT_EQ(pool.num_used(), 3u);
  EXPECT_EQ(pool.num_free(), 1u);
}

TEST(BlockPool, AllocateIsAllOrNothing) {
  BlockPool pool(4, 8);
  pool.Allocate(3);
  EXPECT_THROW(pool.Allocate(2), std::runtime_error);
  EXPECT_EQ(pool.num_free(), 1u);
}

TEST(BlockPool, FreeReusesLifo) {
  BlockPool pool(4, 8);
  auto first = pool.Allocate(1);
  pool.Free(first);
  auto second = pool.Allocate(1);
  ASSERT_EQ(second.size(), 1u);
  EXPECT_EQ(second[0], first[0]);
}

TEST(BlockPool, DoubleFreeThrowsAndLeavesPoolUnchanged) {
  BlockPool pool(4, 8);
  auto ids = pool.Allocate(2);
  const std::size_t used = pool.num_used();
  EXPECT_THROW(pool.Free({ids[0], ids[0]}), std::invalid_argument);
  EXPECT_EQ(pool.num_used(), used);
  pool.Free(ids);
  EXPECT_EQ(pool.num_used(), 0u);
}

TEST(BlockPool, FreeOfUnallocatedThrows) {
  BlockPool pool(2, 8);
  EXPECT_THROW(pool.Free({0}), std::invalid_argument);
}

TEST(BlockPool, BlockWriteRead) {
  BlockPool pool(2, 4);
  auto ids = pool.Allocate(1);
  auto* p = static_cast<std::uint8_t*>(pool.MutableBlock(ids[0]));
  p[0] = 0xAB;
  p[3] = 0xCD;
  const auto* q = static_cast<const std::uint8_t*>(pool.Block(ids[0]));
  EXPECT_EQ(q[0], 0xAB);
  EXPECT_EQ(q[3], 0xCD);
}

TEST(BlockPool, ViewRejectedAfterFree) {
  BlockPool pool(2, 4);
  auto ids = pool.Allocate(1);
  pool.Free(ids);
  EXPECT_THROW(pool.MutableBlock(ids[0]), std::invalid_argument);
}

TEST(BlockPool, ConcurrentAllocFree) {
  BlockPool pool(64, 16);
  auto worker = [&pool]() {
    for (int i = 0; i < 500; ++i) {
      auto ids = pool.Allocate(2);
      auto* p = static_cast<std::uint8_t*>(pool.MutableBlock(ids[0]));
      p[0] = static_cast<std::uint8_t>(i);
      pool.Free(ids);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back(worker);
  }
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_EQ(pool.num_used(), 0u);
  EXPECT_EQ(pool.num_free(), 64u);
}
