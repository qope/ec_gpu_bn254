// Defines to make the code work with both, CUDA and OpenCL
#ifdef __NVCC__
  #define DEVICE __device__
  #define GLOBAL
  #define KERNEL extern "C" __global__
  #define LOCAL __shared__
  #define CONSTANT __constant__

  #define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x
  #define GET_GROUP_ID() blockIdx.x
  #define GET_LOCAL_ID() threadIdx.x
  #define GET_LOCAL_SIZE() blockDim.x
  #define BARRIER_LOCAL() __syncthreads()

  typedef unsigned char uchar;

  #define CUDA
#else // OpenCL
  #define DEVICE
  #define GLOBAL __global
  #define KERNEL __kernel
  #define LOCAL __local
  #define CONSTANT __constant

  #define GET_GLOBAL_ID() get_global_id(0)
  #define GET_GROUP_ID() get_group_id(0)
  #define GET_LOCAL_ID() get_local_id(0)
  #define GET_LOCAL_SIZE() get_local_size(0)
  #define BARRIER_LOCAL() barrier(CLK_LOCAL_MEM_FENCE)
#endif

#ifdef __NV_CL_C_VERSION
#define OPENCL_NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    ulong lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
  #else
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
  #endif
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    ulong lo, hi;
    asm("add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64 %1, 0, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
    *b = hi;
    return lo;
  #else
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
  ulong res = (ulong)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    uint lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
  #else
    uint lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Reverse the given bits. It's used by the FFT kernel.
DEVICE uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

#ifdef CUDA
// CUDA doesn't support local buffers ("dynamic shared memory" in CUDA lingo) as function
// arguments, but only a single globally defined extern value. Use `uchar` so that it is always
// allocated by the number of bytes.
extern LOCAL uchar cuda_shared[];

typedef uint uint32_t;
typedef int  int32_t;
typedef uint limb;

DEVICE inline uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}


DEVICE inline uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

typedef struct {
  int32_t _position;
} chain_t;

DEVICE inline
void chain_init(chain_t *c) {
  c->_position = 0;
}

DEVICE inline
uint32_t chain_add(chain_t *ch, uint32_t a, uint32_t b) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=add_cc(a, b);
  else
    r=addc_cc(a, b);
  return r;
}

DEVICE inline
uint32_t chain_madlo(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madlo_cc(a, b, c);
  else
    r=madloc_cc(a, b, c);
  return r;
}

DEVICE inline
uint32_t chain_madhi(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madhi_cc(a, b, c);
  else
    r=madhic_cc(a, b, c);
  return r;
}
#endif


#define bn254_builder__Scalar_limb ulong
#define bn254_builder__Scalar_LIMBS 4
#define bn254_builder__Scalar_LIMB_BITS 64
#define bn254_builder__Scalar_INV 14042775128853446655
typedef struct { bn254_builder__Scalar_limb val[bn254_builder__Scalar_LIMBS]; } bn254_builder__Scalar;
CONSTANT bn254_builder__Scalar bn254_builder__Scalar_ONE = { { 12436184717236109307, 3962172157175319849, 7381016538464732718, 1011752739694698287 } };
CONSTANT bn254_builder__Scalar bn254_builder__Scalar_P = { { 4891460686036598785, 2896914383306846353, 13281191951274694749, 3486998266802970665 } };
CONSTANT bn254_builder__Scalar bn254_builder__Scalar_R2 = { { 1997599621687373223, 6052339484930628067, 10108755138030829701, 150537098327114917 } };
CONSTANT bn254_builder__Scalar bn254_builder__Scalar_ZERO = { { 0, 0, 0, 0 } };
#if defined(OPENCL_NVIDIA) || defined(CUDA)

DEVICE bn254_builder__Scalar bn254_builder__Scalar_sub_nvidia(bn254_builder__Scalar a, bn254_builder__Scalar b) {
asm("sub.cc.u64 %0, %0, %4;\r\n"
"subc.cc.u64 %1, %1, %5;\r\n"
"subc.cc.u64 %2, %2, %6;\r\n"
"subc.u64 %3, %3, %7;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
return a;
}
DEVICE bn254_builder__Scalar bn254_builder__Scalar_add_nvidia(bn254_builder__Scalar a, bn254_builder__Scalar b) {
asm("add.cc.u64 %0, %0, %4;\r\n"
"addc.cc.u64 %1, %1, %5;\r\n"
"addc.cc.u64 %2, %2, %6;\r\n"
"addc.u64 %3, %3, %7;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define bn254_builder__Scalar_BITS (bn254_builder__Scalar_LIMBS * bn254_builder__Scalar_LIMB_BITS)
#if bn254_builder__Scalar_LIMB_BITS == 32
  #define bn254_builder__Scalar_mac_with_carry mac_with_carry_32
  #define bn254_builder__Scalar_add_with_carry add_with_carry_32
#elif bn254_builder__Scalar_LIMB_BITS == 64
  #define bn254_builder__Scalar_mac_with_carry mac_with_carry_64
  #define bn254_builder__Scalar_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool bn254_builder__Scalar_gte(bn254_builder__Scalar a, bn254_builder__Scalar b) {
  for(char i = bn254_builder__Scalar_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool bn254_builder__Scalar_eq(bn254_builder__Scalar a, bn254_builder__Scalar b) {
  for(uchar i = 0; i < bn254_builder__Scalar_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA)
  #define bn254_builder__Scalar_add_ bn254_builder__Scalar_add_nvidia
  #define bn254_builder__Scalar_sub_ bn254_builder__Scalar_sub_nvidia
#else
  DEVICE bn254_builder__Scalar bn254_builder__Scalar_add_(bn254_builder__Scalar a, bn254_builder__Scalar b) {
    bool carry = 0;
    for(uchar i = 0; i < bn254_builder__Scalar_LIMBS; i++) {
      bn254_builder__Scalar_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  bn254_builder__Scalar bn254_builder__Scalar_sub_(bn254_builder__Scalar a, bn254_builder__Scalar b) {
    bool borrow = 0;
    for(uchar i = 0; i < bn254_builder__Scalar_LIMBS; i++) {
      bn254_builder__Scalar_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE bn254_builder__Scalar bn254_builder__Scalar_sub(bn254_builder__Scalar a, bn254_builder__Scalar b) {
  bn254_builder__Scalar res = bn254_builder__Scalar_sub_(a, b);
  if(!bn254_builder__Scalar_gte(a, b)) res = bn254_builder__Scalar_add_(res, bn254_builder__Scalar_P);
  return res;
}

// Modular addition
DEVICE bn254_builder__Scalar bn254_builder__Scalar_add(bn254_builder__Scalar a, bn254_builder__Scalar b) {
  bn254_builder__Scalar res = bn254_builder__Scalar_add_(a, b);
  if(bn254_builder__Scalar_gte(res, bn254_builder__Scalar_P)) res = bn254_builder__Scalar_sub_(res, bn254_builder__Scalar_P);
  return res;
}


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void bn254_builder__Scalar_reduce(uint32_t accLow[bn254_builder__Scalar_LIMBS], uint32_t np0, uint32_t fq[bn254_builder__Scalar_LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = bn254_builder__Scalar_LIMBS;
  uint32_t accHigh[bn254_builder__Scalar_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void bn254_builder__Scalar_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = bn254_builder__Scalar_LIMBS;
  const uint32_t yLimbs  = bn254_builder__Scalar_LIMBS;
  const uint32_t xyLimbs = bn254_builder__Scalar_LIMBS * 2;
  uint32_t temp[bn254_builder__Scalar_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE bn254_builder__Scalar bn254_builder__Scalar_mul_nvidia(bn254_builder__Scalar a, bn254_builder__Scalar b) {
  // Perform full multiply
  limb ab[2 * bn254_builder__Scalar_LIMBS];
  bn254_builder__Scalar_mult_v1(a.val, b.val, ab);

  uint32_t io[bn254_builder__Scalar_LIMBS];
  #pragma unroll
  for(int i=0;i<bn254_builder__Scalar_LIMBS;i++) {
    io[i]=ab[i];
  }
  bn254_builder__Scalar_reduce(io, bn254_builder__Scalar_INV, bn254_builder__Scalar_P.val);

  // Add io to the upper words of ab
  ab[bn254_builder__Scalar_LIMBS] = add_cc(ab[bn254_builder__Scalar_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < bn254_builder__Scalar_LIMBS - 1; j++) {
    ab[j + bn254_builder__Scalar_LIMBS] = addc_cc(ab[j + bn254_builder__Scalar_LIMBS], io[j]);
  }
  ab[2 * bn254_builder__Scalar_LIMBS - 1] = addc(ab[2 * bn254_builder__Scalar_LIMBS - 1], io[bn254_builder__Scalar_LIMBS - 1]);

  bn254_builder__Scalar r;
  #pragma unroll
  for (int i = 0; i < bn254_builder__Scalar_LIMBS; i++) {
    r.val[i] = ab[i + bn254_builder__Scalar_LIMBS];
  }

  if (bn254_builder__Scalar_gte(r, bn254_builder__Scalar_P)) {
    r = bn254_builder__Scalar_sub_(r, bn254_builder__Scalar_P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE bn254_builder__Scalar bn254_builder__Scalar_mul_default(bn254_builder__Scalar a, bn254_builder__Scalar b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  bn254_builder__Scalar_limb t[bn254_builder__Scalar_LIMBS + 2] = {0};
  for(uchar i = 0; i < bn254_builder__Scalar_LIMBS; i++) {
    bn254_builder__Scalar_limb carry = 0;
    for(uchar j = 0; j < bn254_builder__Scalar_LIMBS; j++)
      t[j] = bn254_builder__Scalar_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[bn254_builder__Scalar_LIMBS] = bn254_builder__Scalar_add_with_carry(t[bn254_builder__Scalar_LIMBS], &carry);
    t[bn254_builder__Scalar_LIMBS + 1] = carry;

    carry = 0;
    bn254_builder__Scalar_limb m = bn254_builder__Scalar_INV * t[0];
    bn254_builder__Scalar_mac_with_carry(m, bn254_builder__Scalar_P.val[0], t[0], &carry);
    for(uchar j = 1; j < bn254_builder__Scalar_LIMBS; j++)
      t[j - 1] = bn254_builder__Scalar_mac_with_carry(m, bn254_builder__Scalar_P.val[j], t[j], &carry);

    t[bn254_builder__Scalar_LIMBS - 1] = bn254_builder__Scalar_add_with_carry(t[bn254_builder__Scalar_LIMBS], &carry);
    t[bn254_builder__Scalar_LIMBS] = t[bn254_builder__Scalar_LIMBS + 1] + carry;
  }

  bn254_builder__Scalar result;
  for(uchar i = 0; i < bn254_builder__Scalar_LIMBS; i++) result.val[i] = t[i];

  if(bn254_builder__Scalar_gte(result, bn254_builder__Scalar_P)) result = bn254_builder__Scalar_sub_(result, bn254_builder__Scalar_P);

  return result;
}

#ifdef CUDA
DEVICE bn254_builder__Scalar bn254_builder__Scalar_mul(bn254_builder__Scalar a, bn254_builder__Scalar b) {
  return bn254_builder__Scalar_mul_nvidia(a, b);
}
#else
DEVICE bn254_builder__Scalar bn254_builder__Scalar_mul(bn254_builder__Scalar a, bn254_builder__Scalar b) {
  return bn254_builder__Scalar_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE bn254_builder__Scalar bn254_builder__Scalar_sqr(bn254_builder__Scalar a) {
  return bn254_builder__Scalar_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of bn254_builder__Scalar_add(a, a)
DEVICE bn254_builder__Scalar bn254_builder__Scalar_double(bn254_builder__Scalar a) {
  for(uchar i = bn254_builder__Scalar_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (bn254_builder__Scalar_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(bn254_builder__Scalar_gte(a, bn254_builder__Scalar_P)) a = bn254_builder__Scalar_sub_(a, bn254_builder__Scalar_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE bn254_builder__Scalar bn254_builder__Scalar_pow(bn254_builder__Scalar base, uint exponent) {
  bn254_builder__Scalar res = bn254_builder__Scalar_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = bn254_builder__Scalar_mul(res, base);
    exponent = exponent >> 1;
    base = bn254_builder__Scalar_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE bn254_builder__Scalar bn254_builder__Scalar_pow_lookup(GLOBAL bn254_builder__Scalar *bases, uint exponent) {
  bn254_builder__Scalar res = bn254_builder__Scalar_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = bn254_builder__Scalar_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE bn254_builder__Scalar bn254_builder__Scalar_mont(bn254_builder__Scalar a) {
  return bn254_builder__Scalar_mul(a, bn254_builder__Scalar_R2);
}

DEVICE bn254_builder__Scalar bn254_builder__Scalar_unmont(bn254_builder__Scalar a) {
  bn254_builder__Scalar one = bn254_builder__Scalar_ZERO;
  one.val[0] = 1;
  return bn254_builder__Scalar_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool bn254_builder__Scalar_get_bit(bn254_builder__Scalar l, uint i) {
  return (l.val[bn254_builder__Scalar_LIMBS - 1 - i / bn254_builder__Scalar_LIMB_BITS] >> (bn254_builder__Scalar_LIMB_BITS - 1 - (i % bn254_builder__Scalar_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint bn254_builder__Scalar_get_bits(bn254_builder__Scalar l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= bn254_builder__Scalar_get_bit(l, skip + i);
  }
  return ret;
}




/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
KERNEL void bn254_builder__Scalar_radix_fft(GLOBAL bn254_builder__Scalar* x, // Source buffer
                      GLOBAL bn254_builder__Scalar* y, // Destination buffer
                      GLOBAL bn254_builder__Scalar* pq, // Precalculated twiddle factors
                      GLOBAL bn254_builder__Scalar* omegas, // [omega, omega^2, omega^4, ...]
                      LOCAL bn254_builder__Scalar* u_arg, // Local buffer to store intermediary values
                      uint n, // Number of elements
                      uint lgp, // Log2 of `p` (Read more in the link above)
                      uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                      uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
// CUDA doesn't support local buffers ("shared memory" in CUDA lingo) as function arguments,
// ignore that argument and use the globally defined extern memory instead.
#ifdef CUDA
  // There can only be a single dynamic shared memory item, hence cast it to the type we need.
  bn254_builder__Scalar* u = (bn254_builder__Scalar*)cuda_shared;
#else
  LOCAL bn254_builder__Scalar* u = u_arg;
#endif

  uint lid = GET_LOCAL_ID();
  uint lsize = GET_LOCAL_SIZE();
  uint index = GET_GROUP_ID();
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg; // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  // Compute powers of twiddle
  const bn254_builder__Scalar twiddle = bn254_builder__Scalar_pow_lookup(omegas, (n >> lgp >> deg) * k);
  bn254_builder__Scalar tmp = bn254_builder__Scalar_pow(twiddle, counts);
  for(uint i = counts; i < counte; i++) {
    u[i] = bn254_builder__Scalar_mul(tmp, x[i*t]);
    tmp = bn254_builder__Scalar_mul(tmp, twiddle);
  }
  BARRIER_LOCAL();

  const uint pqshift = max_deg - deg;
  for(uint rnd = 0; rnd < deg; rnd++) {
    const uint bit = counth >> rnd;
    for(uint i = counts >> 1; i < counte >> 1; i++) {
      const uint di = i & (bit - 1);
      const uint i0 = (i << 1) - di;
      const uint i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = bn254_builder__Scalar_add(u[i0], u[i1]);
      u[i1] = bn254_builder__Scalar_sub(tmp, u[i1]);
      if(di != 0) u[i1] = bn254_builder__Scalar_mul(pq[di << rnd << pqshift], u[i1]);
    }

    BARRIER_LOCAL();
  }

  for(uint i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}

/// Multiplies all of the elements by `field`
KERNEL void bn254_builder__Scalar_mul_by_field(GLOBAL bn254_builder__Scalar* elements,
                        uint n,
                        bn254_builder__Scalar field) {
  const uint gid = GET_GLOBAL_ID();
  elements[gid] = bn254_builder__Scalar_mul(elements[gid], field);
}




