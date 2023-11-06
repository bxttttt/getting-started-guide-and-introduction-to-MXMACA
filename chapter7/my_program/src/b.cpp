//b.cpp:  
#include<mc_runtime_api.h>
__global__ void kernel_b()
{
/* kernel code*/
}
void func_b()
{
	/* launch kernel */
	kernel_b<<<1, 1>>>();
}

//b.h:
extern void func_b();
