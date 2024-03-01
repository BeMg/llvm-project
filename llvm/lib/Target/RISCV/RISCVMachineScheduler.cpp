//===-- RISCVMachineScheduler.cpp - Create RISC-V Own Scheduler --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "RISCVMachineScheduler.h"
using namespace llvm;

/// Pick the best node to balance the schedule. Implements MachineSchedStrategy.
SUnit *RISCVSchedStrategy::pickNode(bool &IsTopNode) {
  return GenericScheduler::pickNode(IsTopNode);
}
