#include<mc_runtime_api.h>

int main( void ) {
    mcDeviceProp_t prop;
    
    int count;
    mcGetDeviceCount( &count );
    for (int i=0; i< count; i++) {
        mcGetDeviceProperties( &prop, i );
        printf( " --- General Information for device %d ---\n", i );
        printf( "Name: %s\n", prop.name );
        printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate: %d\n", prop.clockRate );
        printf( "Device copy overlap: " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
        printf( "Kernel execition timeout : " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
        
        printf( " --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count: %d\n",
                prop.multiProcessorCount );
        printf( "Threads in wave: %d\n", prop.waveSize );
        printf( "Max threads per block: %d\n",
                prop.maxThreadsPerBlock );
        printf( "Max thread dimensions: (%d, %d, %d)\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                prop.maxThreadsDim[2] );
        printf( "Max grid dimensions: (%d, %d, %d)\n",
                prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2] );
        printf( "\n" );
    }
} 
