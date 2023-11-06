#include <stdio.h>
#include <mc_runtime_api.h>

__global__ void waveReduce() {
    int laneId = threadIdx.x & 0x3f;
    // Seed starting value as inverse lane ID
    int value = 63 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=1; i<64; i*=2)
		value += __shfl_xor_sync(0xffffffffffffffff, value, i, 64);

    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    waveReduce<<< 1, 64 >>>();
    mcDeviceSynchronize();
    return 0;
} 
