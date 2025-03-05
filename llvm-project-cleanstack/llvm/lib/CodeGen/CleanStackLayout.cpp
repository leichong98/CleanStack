//===- CleanStackLayout.cpp - CleanStack frame layout -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CleanStackLayout.h"
#include "CleanStackColoring.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <random>

using namespace llvm;
using namespace llvm::cleanstack;

#define DEBUG_TYPE "cleanstacklayout"

static cl::opt<bool> ClLayout("clean-stack-layout",
                              cl::desc("enable clean stack layout"), cl::Hidden,
                              cl::init(false));

LLVM_DUMP_METHOD void StackLayout::print(raw_ostream &OS) {
  OS << "Stack regions:\n";
  for (unsigned i = 0; i < Regions.size(); ++i) {
    OS << "  " << i << ": [" << Regions[i].Start << ", " << Regions[i].End
       << "), range " << Regions[i].Range << "\n";
  }
  OS << "Stack objects:\n";
  for (auto &IT : ObjectOffsets) {
    OS << "  at " << IT.getSecond() << ": " << *IT.getFirst() << "\n";
  }
}

void StackLayout::addObject(const Value *V, unsigned Size, unsigned Alignment,
                            const StackColoring::LiveRange &Range) {
  StackObjects.push_back({V, Size, Alignment, Range});
  ObjectAlignments[V] = Alignment;
  MaxAlignment = std::max(MaxAlignment, Alignment);
}

static unsigned AdjustStackOffset(unsigned Offset, unsigned Size,
                                  unsigned Alignment) {
  return alignTo(Offset + Size, Alignment) - Size;
}

void StackLayout::layoutObject(StackObject &Obj) {
  if (!ClLayout) {
    // If layout is disabled, just grab the next aligned address.
    // This effectively disables stack coloring as well.
    unsigned LastRegionEnd = Regions.empty() ? 0 : Regions.back().End;
    unsigned Start = AdjustStackOffset(LastRegionEnd, Obj.Size, Obj.Alignment);
    unsigned End = Start + Obj.Size;
    Regions.emplace_back(Start, End, Obj.Range);
    ObjectOffsets[Obj.Handle] = End;
    return;
  }

  LLVM_DEBUG(dbgs() << "Layout: size " << Obj.Size << ", align "
                    << Obj.Alignment.value() << ", range " << Obj.Range
                    << "\n");
  assert(Obj.Alignment <= MaxAlignment);
  unsigned Start = AdjustStackOffset(0, Obj.Size, Obj.Alignment);
  unsigned End = Start + Obj.Size;
  LLVM_DEBUG(dbgs() << "  First candidate: " << Start << " .. " << End << "\n");
  for (const StackRegion &R : Regions) {
    LLVM_DEBUG(dbgs() << "  Examining region: " << R.Start << " .. " << R.End
                      << ", range " << R.Range << "\n");
    assert(End >= R.Start);
    if (Start >= R.End) {
      LLVM_DEBUG(dbgs() << "  Does not intersect, skip.\n");
      continue;
    }
    if (Obj.Range.Overlaps(R.Range)) {
      // Find the next appropriate location.
      Start = AdjustStackOffset(R.End, Obj.Size, Obj.Alignment);
      End = Start + Obj.Size;
      LLVM_DEBUG(dbgs() << "  Overlaps. Next candidate: " << Start << " .. "
                        << End << "\n");
      continue;
    }
    if (End <= R.End) {
      LLVM_DEBUG(dbgs() << "  Reusing region(s).\n");
      break;
    }
  }

  unsigned LastRegionEnd = Regions.empty() ? 0 : Regions.back().End;
  if (End > LastRegionEnd) {
    // Insert a new region at the end. Maybe two.
    if (Start > LastRegionEnd) {
      LLVM_DEBUG(dbgs() << "  Creating gap region: " << LastRegionEnd << " .. "
                        << Start << "\n");
      Regions.emplace_back(LastRegionEnd, Start, StackColoring::LiveRange());
      LastRegionEnd = Start;
    }
    LLVM_DEBUG(dbgs() << "  Creating new region: " << LastRegionEnd << " .. "
                      << End << ", range " << Obj.Range << "\n");
    Regions.emplace_back(LastRegionEnd, End, Obj.Range);
    LastRegionEnd = End;
  }

  // Split starting and ending regions if necessary.
  for (unsigned i = 0; i < Regions.size(); ++i) {
    StackRegion &R = Regions[i];
    if (Start > R.Start && Start < R.End) {
      StackRegion R0 = R;
      R.Start = R0.End = Start;
      Regions.insert(&R, R0);
      continue;
    }
    if (End > R.Start && End < R.End) {
      StackRegion R0 = R;
      R0.End = R.Start = End;
      Regions.insert(&R, R0);
      break;
    }
  }

  // Update live ranges for all affected regions.
  for (StackRegion &R : Regions) {
    if (Start < R.End && End > R.Start)
      R.Range.Join(Obj.Range);
    if (End <= R.End)
      break;
  }

  ObjectOffsets[Obj.Handle] = End;
}

void StackLayout::computeLayout() {
    // 如果没有对象需要布局，则直接返回
    if (StackObjects.empty()) 
        return;

    // 创建随机数生成器，用当前时间作为种子
    std::random_device rd;
    std::mt19937 g(rd());

    // 保留第一个位置的对象（通常是 StackGuardSlot），不参与随机化
    StackObject *FirstObject = nullptr;

    // 如果第一个对象是 StackGuardSlot，则将其从 StackObjects 中提取出来
    if (!StackObjects.empty() && StackObjects.front().Handle->getName() == "stack_guard") {
        FirstObject = &StackObjects.front();
        // 不移除 StackGuardSlot，保持它在栈的最前面
    }

    // 对剩余的栈对象顺序进行随机化
    std::shuffle(StackObjects.begin() + (FirstObject ? 1 : 0), StackObjects.end(), g);

    // 如果存在 StackGuardSlot，则将其放置在第一个位置
    if (FirstObject) {
        layoutObject(*FirstObject);
    }

    // 布局剩余的随机化后的栈对象
    for (auto &Obj : StackObjects) {
        // 如果是 FirstObject，跳过
        if (&Obj == FirstObject)
            continue;
        layoutObject(Obj);
    }

    // 打印布局后的调试信息
    LLVM_DEBUG(print(dbgs()));
}
