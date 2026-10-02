//===- llvm/CodeGen/NewShrinkWrap.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_NEWSHRINKWRAP_H
#define LLVM_CODEGEN_NEWSHRINKWRAP_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// GCC-style shrink-wrapping, see NewShrinkWrap.cpp.
class NewShrinkWrapPass : public OptionalPassInfoMixin<NewShrinkWrapPass> {
public:
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setNoVRegs();
  }
};

/// Return true if the codegen pipeline should run the NewShrinkWrap pass
/// instead of the ShrinkWrap pass (-enable-new-shrink-wrap).
LLVM_ABI bool isNewShrinkWrapEnabled();

} // namespace llvm

#endif // LLVM_CODEGEN_NEWSHRINKWRAP_H
