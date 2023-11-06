#include<mc_runtime_api.h>

typedef struct 
{
  alignas(4)float f;
  double d;
}__attribute__((packed)) test_type_mem_violation;

__global__ void trigger_memory_violation(test_type_mem_violation *dst)
{
  atomicAdd(&dst->f,1.23);
  atomicAdd(&dst->d,20);
  dst->f=9.8765;
}

int main()
{
  test_type_mem_violation hd={0};
  test_type_mem_violation *ddd;
  mcMalloc((void**)&ddd,sizeof(test_type_mem_violation));
  mcMemcpy(ddd,&hd,sizeof(test_type_mem_violation),mcMemcpyHostToDevice);
  trigger_memory_violation<<<dim3(1),dim3(1)>>>(ddd);
  mcMemcpy(&hd,ddd,sizeof(test_type_mem_violation),mcMemcpyDeviceToHost);
  mcFree(ddd);
  return 0;
}
