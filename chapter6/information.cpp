#include<mc_runtime_api.h>

int main( void ) {
    mcDeviceProp_t prop;
    
    int count;
    mcGetDeviceCount( &count );
    for (int i=0; i< count; i++) {
        mcGetDeviceProperties( &prop, i );
        printf( " --- Memory Information for device %d ---\n", i );
        printf( "Total global mem: %ld[bytes]\n", prop.totalGlobalMem );
        printf( "Total constant Mem: %ld[bytes]\n", prop.totalConstMem );
        printf( "Max mem pitch: %ld[bytes]\n", prop.memPitch );
        printf( "Texture alignment: %ld[bytes]\n", prop.textureAlignment );
        printf( "Shared mem per AP: %ld[bytes]\n",prop.sharedMemPerBlock );
        printf( "Registers per AP: %d[bytes]\n", prop.regsPerBlock );
        printf( "\n" );
    }
}
