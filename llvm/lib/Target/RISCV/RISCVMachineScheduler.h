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

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCVInsertVSETVLI.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

/// A MachineSchedStrategy implementation for PowerPC pre RA scheduling.
class RISCVSchedStrategy : public GenericScheduler {
public:
  RISCVSchedStrategy(const MachineSchedContext *C) : GenericScheduler(C) {
    LIS = C->LIS;
  }

protected:
  void schedNode(SUnit *SU, bool IsTopNode) override;
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

private:
  VSETVLIInfo CurrVSETVLInfo = VSETVLIInfo::getUnknown();

  LiveIntervals *LIS;

  VSETVLIInfo computeInfoForMI(const MachineInstr *MI) const {
    const MachineFunction *MF = MI->getMF();
    const RISCVSubtarget &ST = MF->getSubtarget<RISCVSubtarget>();
    return computeInfoForInstr(*MI, &ST, LIS);
  }

  bool isRVVInst(MachineInstr *MI) const {
    uint64_t TSFlags = MI->getDesc().TSFlags;
    if (!RISCVII::hasSEWOp(TSFlags))
      return false;
    return true;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H