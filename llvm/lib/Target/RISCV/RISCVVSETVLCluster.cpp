//===--- RISCVVSETVLClusting.cpp - RISCV VSETVL Clustering  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to cluster VSETVL
///       instruction.
//
//===----------------------------------------------------------------------===//

#include "RISCVVSETVLCluster.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

class VSETVLClustering : public ScheduleDAGMutation {
public:
  VSETVLClustering() = default;
  void apply(ScheduleDAGInstrs *DAG) override;
};

void VSETVLClustering::apply(ScheduleDAGInstrs *DAG) {}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createRISCVVSETVLClusteringDAGMutation() {
  return std::make_unique<VSETVLClustering>();
}

} // end namespace llvm
