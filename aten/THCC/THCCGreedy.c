#include <THC/THC.h>

#include "THCGreedy.h"

#define THCCGreedy_ TH_CONCAT_3(THCC,Real,Greedy)
#define THCGreedy_ TH_CONCAT_3(TH,CReal,Greedy)

extern THCState *state;

void THCCGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                THCudaLongTensor *deg) {
  THCGreedy(state, cluster, row, col, deg);
}

#include "generic/THCCGreedy.c"
#include "THCGenerateAllTypes.h"
