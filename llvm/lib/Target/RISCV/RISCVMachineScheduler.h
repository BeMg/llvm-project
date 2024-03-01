//===-- RISCVMachineScheduler.h - Create RISC-V Own Scheduler --*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H

#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

/// A MachineSchedStrategy implementation for PowerPC pre RA scheduling.
class RISCVSchedStrategy : public GenericScheduler {
public:
  RISCVSchedStrategy(const MachineSchedContext *C) : GenericScheduler(C) {}

protected:
  SUnit *pickNode(bool &IsTopNode) override;

private:
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H
