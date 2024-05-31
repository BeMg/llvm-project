#include <stdio.h>

static long syscall_impl_5_args(long number, long arg1, long arg2, long arg3,
                                long arg4, long arg5) {
  register long a7 __asm__("a7") = number;
  register long a0 __asm__("a0") = arg1;
  register long a1 __asm__("a1") = arg2;
  register long a2 __asm__("a2") = arg3;
  register long a3 __asm__("a3") = arg4;
  register long a4 __asm__("a4") = arg5;
  __asm__ __volatile__("ecall\n\t"
                       : "=r"(a0)
                       : "r"(a7), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4)
                       : "memory");
  return a0;
}

#define RISCV_HWPROBE_KEY_MVENDORID 0
#define RISCV_HWPROBE_KEY_MARCHID 1
#define RISCV_HWPROBE_KEY_MIMPID 2
#define RISCV_HWPROBE_KEY_BASE_BEHAVIOR 3
#define RISCV_HWPROBE_BASE_BEHAVIOR_IMA (1ULL << 0)
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#define RISCV_HWPROBE_IMA_FD (1ULL << 0)
#define RISCV_HWPROBE_IMA_C (1ULL << 1)
#define RISCV_HWPROBE_IMA_V (1ULL << 2)
#define RISCV_HWPROBE_EXT_ZBA (1ULL << 3)
#define RISCV_HWPROBE_EXT_ZBB (1ULL << 4)
#define RISCV_HWPROBE_EXT_ZBS (1ULL << 5)
#define RISCV_HWPROBE_EXT_ZICBOZ (1ULL << 6)
#define RISCV_HWPROBE_EXT_ZBC (1ULL << 7)
#define RISCV_HWPROBE_EXT_ZBKB (1ULL << 8)
#define RISCV_HWPROBE_EXT_ZBKC (1ULL << 9)
#define RISCV_HWPROBE_EXT_ZBKX (1ULL << 10)
#define RISCV_HWPROBE_EXT_ZKND (1ULL << 11)
#define RISCV_HWPROBE_EXT_ZKNE (1ULL << 12)
#define RISCV_HWPROBE_EXT_ZKNH (1ULL << 13)
#define RISCV_HWPROBE_EXT_ZKSED (1ULL << 14)
#define RISCV_HWPROBE_EXT_ZKSH (1ULL << 15)
#define RISCV_HWPROBE_EXT_ZKT (1ULL << 16)
#define RISCV_HWPROBE_EXT_ZVBB (1ULL << 17)
#define RISCV_HWPROBE_EXT_ZVBC (1ULL << 18)
#define RISCV_HWPROBE_EXT_ZVKB (1ULL << 19)
#define RISCV_HWPROBE_EXT_ZVKG (1ULL << 20)
#define RISCV_HWPROBE_EXT_ZVKNED (1ULL << 21)
#define RISCV_HWPROBE_EXT_ZVKNHA (1ULL << 22)
#define RISCV_HWPROBE_EXT_ZVKNHB (1ULL << 23)
#define RISCV_HWPROBE_EXT_ZVKSED (1ULL << 24)
#define RISCV_HWPROBE_EXT_ZVKSH (1ULL << 25)
#define RISCV_HWPROBE_EXT_ZVKT (1ULL << 26)
#define RISCV_HWPROBE_EXT_ZFH (1ULL << 27)
#define RISCV_HWPROBE_EXT_ZFHMIN (1ULL << 28)
#define RISCV_HWPROBE_EXT_ZIHINTNTL (1ULL << 29)
#define RISCV_HWPROBE_EXT_ZVFH (1ULL << 30)
#define RISCV_HWPROBE_EXT_ZVFHMIN (1ULL << 31)
#define RISCV_HWPROBE_EXT_ZFA (1ULL << 32)
#define RISCV_HWPROBE_EXT_ZTSO (1ULL << 33)
#define RISCV_HWPROBE_EXT_ZACAS (1ULL << 34)
#define RISCV_HWPROBE_EXT_ZICOND (1ULL << 35)
#define RISCV_HWPROBE_KEY_CPUPERF_0 5
#define RISCV_HWPROBE_MISALIGNED_UNKNOWN (0 << 0)
#define RISCV_HWPROBE_MISALIGNED_EMULATED (1ULL << 0)
#define RISCV_HWPROBE_MISALIGNED_SLOW (2 << 0)
#define RISCV_HWPROBE_MISALIGNED_FAST (3 << 0)
#define RISCV_HWPROBE_MISALIGNED_UNSUPPORTED (4 << 0)
#define RISCV_HWPROBE_MISALIGNED_MASK (7 << 0)
#define RISCV_HWPROBE_KEY_ZICBOZ_BLOCK_SIZE 6
/* Increase RISCV_HWPROBE_MAX_KEY when adding items. */

/* Flags */
#define RISCV_HWPROBE_WHICH_CPUS (1ULL << 0)

struct riscv_hwprobe {
  long long key;
  unsigned long long value;
};

/* Size definition for CPU sets.  */
#define __CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(unsigned long int))

/* Data structure to describe CPU mask.  */
typedef struct {
  unsigned long int __bits[__CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

#define SYS_riscv_hwprobe 258
static long sys_riscv_hwprobe(struct riscv_hwprobe *pairs, unsigned pair_count,
                              unsigned cpu_count, cpu_set_t *cpus,
                              unsigned int flags) {
  return syscall_impl_5_args(SYS_riscv_hwprobe, (long)pairs, pair_count,
                             cpu_count, (long)cpus, flags);
}

static long initHwProbe(struct riscv_hwprobe *Hwprobes, int len) {
  return sys_riscv_hwprobe(Hwprobes, len, 0, (cpu_set_t *)((void *)0), 0);
}

struct {
  unsigned length;
  unsigned long long features[2];
} __riscv_feature_bits __attribute__((visibility("hidden"), nocommon));

#define A_GROUPID 0
#define A_BITMASK 4194304ULL
#define C_GROUPID 0
#define C_BITMASK 8796093022208ULL
#define D_GROUPID 0
#define D_BITMASK 17179869184ULL
#define E_GROUPID 0
#define E_BITMASK 2ULL
#define F_GROUPID 0
#define F_BITMASK 8589934592ULL
#define H_GROUPID 1
#define H_BITMASK 8796093022208ULL
#define I_GROUPID 0
#define I_BITMASK 1ULL
#define M_GROUPID 0
#define M_BITMASK 1048576ULL
#define V_GROUPID 1
#define V_BITMASK 4194304ULL
#define ZA128RS_GROUPID 0
#define ZA128RS_BITMASK 33554432ULL
#define ZA64RS_GROUPID 0
#define ZA64RS_BITMASK 16777216ULL
#define ZAAMO_GROUPID 0
#define ZAAMO_BITMASK 67108864ULL
#define ZABHA_GROUPID 0
#define ZABHA_BITMASK 134217728ULL
#define ZACAS_GROUPID 0
#define ZACAS_BITMASK 268435456ULL
#define ZALASR_GROUPID 0
#define ZALASR_BITMASK 536870912ULL
#define ZALRSC_GROUPID 0
#define ZALRSC_BITMASK 1073741824ULL
#define ZAMA16B_GROUPID 0
#define ZAMA16B_BITMASK 2147483648ULL
#define ZAWRS_GROUPID 0
#define ZAWRS_BITMASK 4294967296ULL
#define ZBA_GROUPID 0
#define ZBA_BITMASK 4503599627370496ULL
#define ZBB_GROUPID 0
#define ZBB_BITMASK 9007199254740992ULL
#define ZBC_GROUPID 0
#define ZBC_BITMASK 18014398509481984ULL
#define ZBKB_GROUPID 0
#define ZBKB_BITMASK 72057594037927936ULL
#define ZBKC_GROUPID 0
#define ZBKC_BITMASK 288230376151711744ULL
#define ZBKX_GROUPID 0
#define ZBKX_BITMASK 144115188075855872ULL
#define ZBS_GROUPID 0
#define ZBS_BITMASK 36028797018963968ULL
#define ZCA_GROUPID 0
#define ZCA_BITMASK 17592186044416ULL
#define ZCB_GROUPID 0
#define ZCB_BITMASK 35184372088832ULL
#define ZCD_GROUPID 0
#define ZCD_BITMASK 70368744177664ULL
#define ZCE_GROUPID 0
#define ZCE_BITMASK 1125899906842624ULL
#define ZCF_GROUPID 0
#define ZCF_BITMASK 140737488355328ULL
#define ZCMOP_GROUPID 0
#define ZCMOP_BITMASK 2251799813685248ULL
#define ZCMP_GROUPID 0
#define ZCMP_BITMASK 281474976710656ULL
#define ZCMT_GROUPID 0
#define ZCMT_BITMASK 562949953421312ULL
#define ZDINX_GROUPID 0
#define ZDINX_BITMASK 1099511627776ULL
#define ZFA_GROUPID 0
#define ZFA_BITMASK 274877906944ULL
#define ZFBFMIN_GROUPID 0
#define ZFBFMIN_BITMASK 137438953472ULL
#define ZFH_GROUPID 0
#define ZFH_BITMASK 68719476736ULL
#define ZFHMIN_GROUPID 0
#define ZFHMIN_BITMASK 34359738368ULL
#define ZFINX_GROUPID 0
#define ZFINX_BITMASK 549755813888ULL
#define ZHINX_GROUPID 0
#define ZHINX_BITMASK 4398046511104ULL
#define ZHINXMIN_GROUPID 0
#define ZHINXMIN_BITMASK 2199023255552ULL
#define ZIC64B_GROUPID 0
#define ZIC64B_BITMASK 4ULL
#define ZICBOM_GROUPID 0
#define ZICBOM_BITMASK 8ULL
#define ZICBOP_GROUPID 0
#define ZICBOP_BITMASK 16ULL
#define ZICBOZ_GROUPID 0
#define ZICBOZ_BITMASK 32ULL
#define ZICCAMOA_GROUPID 0
#define ZICCAMOA_BITMASK 64ULL
#define ZICCIF_GROUPID 0
#define ZICCIF_BITMASK 128ULL
#define ZICCLSM_GROUPID 0
#define ZICCLSM_BITMASK 256ULL
#define ZICCRSE_GROUPID 0
#define ZICCRSE_BITMASK 512ULL
#define ZICFILP_GROUPID 0
#define ZICFILP_BITMASK 262144ULL
#define ZICFISS_GROUPID 0
#define ZICFISS_BITMASK 524288ULL
#define ZICNTR_GROUPID 0
#define ZICNTR_BITMASK 2048ULL
#define ZICOND_GROUPID 0
#define ZICOND_BITMASK 4096ULL
#define ZICSR_GROUPID 0
#define ZICSR_BITMASK 1024ULL
#define ZIFENCEI_GROUPID 0
#define ZIFENCEI_BITMASK 8192ULL
#define ZIHINTNTL_GROUPID 0
#define ZIHINTNTL_BITMASK 32768ULL
#define ZIHINTPAUSE_GROUPID 0
#define ZIHINTPAUSE_BITMASK 16384ULL
#define ZIHPM_GROUPID 0
#define ZIHPM_BITMASK 65536ULL
#define ZIMOP_GROUPID 0
#define ZIMOP_BITMASK 131072ULL
#define ZK_GROUPID 1
#define ZK_BITMASK 16ULL
#define ZKN_GROUPID 1
#define ZKN_BITMASK 2ULL
#define ZKND_GROUPID 0
#define ZKND_BITMASK 576460752303423488ULL
#define ZKNE_GROUPID 0
#define ZKNE_BITMASK 1152921504606846976ULL
#define ZKNH_GROUPID 0
#define ZKNH_BITMASK 2305843009213693952ULL
#define ZKR_GROUPID 1
#define ZKR_BITMASK 1ULL
#define ZKS_GROUPID 1
#define ZKS_BITMASK 4ULL
#define ZKSED_GROUPID 0
#define ZKSED_BITMASK 4611686018427387904ULL
#define ZKSH_GROUPID 0
#define ZKSH_BITMASK 9223372036854775808ULL
#define ZKT_GROUPID 1
#define ZKT_BITMASK 8ULL
#define ZMMUL_GROUPID 0
#define ZMMUL_BITMASK 2097152ULL
#define ZTSO_GROUPID 0
#define ZTSO_BITMASK 8388608ULL
#define ZVBB_GROUPID 1
#define ZVBB_BITMASK 268435456ULL
#define ZVBC_GROUPID 1
#define ZVBC_BITMASK 536870912ULL
#define ZVE32F_GROUPID 1
#define ZVE32F_BITMASK 262144ULL
#define ZVE32X_GROUPID 1
#define ZVE32X_BITMASK 131072ULL
#define ZVE64D_GROUPID 1
#define ZVE64D_BITMASK 2097152ULL
#define ZVE64F_GROUPID 1
#define ZVE64F_BITMASK 1048576ULL
#define ZVE64X_GROUPID 1
#define ZVE64X_BITMASK 524288ULL
#define ZVFBFMIN_GROUPID 1
#define ZVFBFMIN_BITMASK 8388608ULL
#define ZVFBFWMA_GROUPID 1
#define ZVFBFWMA_BITMASK 16777216ULL
#define ZVFH_GROUPID 1
#define ZVFH_BITMASK 67108864ULL
#define ZVFHMIN_GROUPID 1
#define ZVFHMIN_BITMASK 33554432ULL
#define ZVKB_GROUPID 1
#define ZVKB_BITMASK 134217728ULL
#define ZVKG_GROUPID 1
#define ZVKG_BITMASK 1073741824ULL
#define ZVKN_GROUPID 1
#define ZVKN_BITMASK 137438953472ULL
#define ZVKNC_GROUPID 1
#define ZVKNC_BITMASK 274877906944ULL
#define ZVKNED_GROUPID 1
#define ZVKNED_BITMASK 2147483648ULL
#define ZVKNG_GROUPID 1
#define ZVKNG_BITMASK 549755813888ULL
#define ZVKNHA_GROUPID 1
#define ZVKNHA_BITMASK 4294967296ULL
#define ZVKNHB_GROUPID 1
#define ZVKNHB_BITMASK 8589934592ULL
#define ZVKS_GROUPID 1
#define ZVKS_BITMASK 1099511627776ULL
#define ZVKSC_GROUPID 1
#define ZVKSC_BITMASK 2199023255552ULL
#define ZVKSED_GROUPID 1
#define ZVKSED_BITMASK 17179869184ULL
#define ZVKSG_GROUPID 1
#define ZVKSG_BITMASK 4398046511104ULL
#define ZVKSH_GROUPID 1
#define ZVKSH_BITMASK 34359738368ULL
#define ZVKT_GROUPID 1
#define ZVKT_BITMASK 68719476736ULL
#define ZVL1024B_GROUPID 1
#define ZVL1024B_BITMASK 1024ULL
#define ZVL128B_GROUPID 1
#define ZVL128B_BITMASK 128ULL
#define ZVL16384B_GROUPID 1
#define ZVL16384B_BITMASK 16384ULL
#define ZVL2048B_GROUPID 1
#define ZVL2048B_BITMASK 2048ULL
#define ZVL256B_GROUPID 1
#define ZVL256B_BITMASK 256ULL
#define ZVL32768B_GROUPID 1
#define ZVL32768B_BITMASK 32768ULL
#define ZVL32B_GROUPID 1
#define ZVL32B_BITMASK 32ULL
#define ZVL4096B_GROUPID 1
#define ZVL4096B_BITMASK 4096ULL
#define ZVL512B_GROUPID 1
#define ZVL512B_BITMASK 512ULL
#define ZVL64B_GROUPID 1
#define ZVL64B_BITMASK 64ULL
#define ZVL65536B_GROUPID 1
#define ZVL65536B_BITMASK 65536ULL
#define ZVL8192B_GROUPID 1
#define ZVL8192B_BITMASK 8192ULL
#define HWPROBE_LENGTH 2

static void initRISCVFeature(struct riscv_hwprobe Hwprobes[]) {
  __riscv_feature_bits.length = 2;
  // Check RISCV_HWPROBE_KEY_BASE_BEHAVIOR
  if (Hwprobes[0].value & RISCV_HWPROBE_BASE_BEHAVIOR_IMA) {
    __riscv_feature_bits.features[I_GROUPID] |= I_BITMASK;
    __riscv_feature_bits.features[M_GROUPID] |= M_BITMASK;
    __riscv_feature_bits.features[A_GROUPID] |= A_BITMASK;
  }

  // Check RISCV_HWPROBE_KEY_IMA_EXT_0
  if (Hwprobes[1].value & RISCV_HWPROBE_IMA_FD) {
    __riscv_feature_bits.features[F_GROUPID] |= F_BITMASK;
    __riscv_feature_bits.features[D_GROUPID] |= D_BITMASK;
  }

  if (Hwprobes[1].value & RISCV_HWPROBE_IMA_C) {
    __riscv_feature_bits.features[C_GROUPID] |= C_BITMASK;
  }

  if (Hwprobes[1].value & RISCV_HWPROBE_IMA_V) {
    __riscv_feature_bits.features[V_GROUPID] |= V_BITMASK;
  }

  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZBA) {
    __riscv_feature_bits.features[ZBA_GROUPID] |= ZBA_BITMASK;
  }

  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZBB) {
    __riscv_feature_bits.features[ZBB_GROUPID] |= ZBB_BITMASK;
  }

  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZBS) {
    __riscv_feature_bits.features[ZBS_GROUPID] |= ZBS_BITMASK;
  }

  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZICBOZ) {
    __riscv_feature_bits.features[ZICBOZ_GROUPID] |= ZICBOZ_BITMASK;
  }

  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZBC) {
    __riscv_feature_bits.features[ZBC_GROUPID] |= ZBC_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZBKB) {
    __riscv_feature_bits.features[ZBKB_GROUPID] |= ZBKB_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZBKC) {
    __riscv_feature_bits.features[ZBKC_GROUPID] |= ZBKC_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZBKX) {
    __riscv_feature_bits.features[ZBKX_GROUPID] |= ZBKX_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZKND) {
    __riscv_feature_bits.features[ZKND_GROUPID] |= ZKND_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZKNE) {
    __riscv_feature_bits.features[ZKNE_GROUPID] |= ZKNE_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZKNH) {
    __riscv_feature_bits.features[ZKNH_GROUPID] |= ZKNH_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZKSED) {
    __riscv_feature_bits.features[ZKSED_GROUPID] |= ZKSED_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZKSH) {
    __riscv_feature_bits.features[ZKSH_GROUPID] |= ZKSH_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZKT) {
    __riscv_feature_bits.features[ZKT_GROUPID] |= ZKT_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVBB) {
    __riscv_feature_bits.features[ZVBB_GROUPID] |= ZVBB_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVBC) {
    __riscv_feature_bits.features[ZVBC_GROUPID] |= ZVBC_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVKB) {
    __riscv_feature_bits.features[ZVKB_GROUPID] |= ZVKB_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVKG) {
    __riscv_feature_bits.features[ZVKG_GROUPID] |= ZVKG_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVKNED) {
    __riscv_feature_bits.features[ZVKNED_GROUPID] |= ZVKNED_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVKNHA) {
    __riscv_feature_bits.features[ZVKNHA_GROUPID] |= ZVKNHA_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVKNHB) {
    __riscv_feature_bits.features[ZVKNHB_GROUPID] |= ZVKNHB_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVKSED) {
    __riscv_feature_bits.features[ZVKSED_GROUPID] |= ZVKSED_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVKSH) {
    __riscv_feature_bits.features[ZVKSH_GROUPID] |= ZVKSH_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVKT) {
    __riscv_feature_bits.features[ZVKT_GROUPID] |= ZVKT_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZFH) {
    __riscv_feature_bits.features[ZFH_GROUPID] |= ZFH_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZFHMIN) {
    __riscv_feature_bits.features[ZFHMIN_GROUPID] |= ZFHMIN_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZIHINTNTL) {
    __riscv_feature_bits.features[ZIHINTNTL_GROUPID] |= ZIHINTNTL_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVFH) {
    __riscv_feature_bits.features[ZVFH_GROUPID] |= ZVFH_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZVFHMIN) {
    __riscv_feature_bits.features[ZVFHMIN_GROUPID] |= ZVFHMIN_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZFA) {
    __riscv_feature_bits.features[ZFA_GROUPID] |= ZFA_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZTSO) {
    __riscv_feature_bits.features[ZTSO_GROUPID] |= ZTSO_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZACAS) {
    __riscv_feature_bits.features[ZACAS_GROUPID] |= ZACAS_BITMASK;
  }
  if (Hwprobes[1].value & RISCV_HWPROBE_EXT_ZICOND) {
    __riscv_feature_bits.features[ZICOND_GROUPID] |= ZICOND_BITMASK;
  }
}

void __init_riscv_features_bit() {

  // TODO: Cache the result, then we could only run once.
  struct riscv_hwprobe Hwprobes[HWPROBE_LENGTH];
  Hwprobes[0].key = RISCV_HWPROBE_KEY_BASE_BEHAVIOR;
  Hwprobes[1].key = RISCV_HWPROBE_KEY_IMA_EXT_0;
  initHwProbe(Hwprobes, HWPROBE_LENGTH);

  initRISCVFeature(Hwprobes);
}
