#pragma warning(disable : 4996)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"source.h"
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <vector>
using namespace std;
void printDeviceProp(const cudaDeviceProp& prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA 初始化
bool InitCUDA()
{
    int count;

    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);

    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;

    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        //打印设备信息
        printDeviceProp(prop);
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}

//容器有盖
void initializeParticlePositionAndVelocity_for2dim(void) {

    int iX, iY;
    int nX, nY;
    double x, y, z;
    int i = 0;
    int flagOfParticleGeneration;

    nX = (int)(1.0 / PARTICLE_DISTANCE) + 5;
    nY = (int)(0.7 / PARTICLE_DISTANCE) + 5;
    for (iX = -4; iX < nX; iX++) {
        for (iY = -4; iY < nY; iY++) {
            x = PARTICLE_DISTANCE * (double)(iX);
            y = PARTICLE_DISTANCE * (double)(iY);
            z = 0.0;
            flagOfParticleGeneration = OFF;

            if (((x > -4.0 * PARTICLE_DISTANCE + EPS) && (x <= 1.00 + 4.0 * PARTICLE_DISTANCE + EPS)) && ((y > 0.0 - 4.0 * PARTICLE_DISTANCE + EPS) && (y <= 0.7 + EPS))) {  /* dummy wall region */
                ParticleType[i] = DUMMY_WALL;
                flagOfParticleGeneration = ON;
            }

            if (((x > -2.0 * PARTICLE_DISTANCE + EPS) && (x <= 1.00 + 2.0 * PARTICLE_DISTANCE + EPS)) && ((y > 0.0 - 2.0 * PARTICLE_DISTANCE + EPS) && (y <= 0.65 + EPS))) { /* wall region */
                ParticleType[i] = WALL;
                flagOfParticleGeneration = ON;
            }

            //      if( ((x>-4.0*PARTICLE_DISTANCE+EPS)&&(x<=1.00+4.0*PARTICLE_DISTANCE+EPS))&&( (y>0.6-2.0*PARTICLE_DISTANCE+EPS )&&(y<=0.6+EPS)) ){  /* wall region */
            //    ParticleType[i]=WALL;
            //    flagOfParticleGeneration = ON;
            //      }

            if (((x > 0.0 + EPS) && (x <= 1.00 + EPS)) && (y > 0.0 + EPS) && (y < 0.6 + EPS)) {  /* empty region */
                flagOfParticleGeneration = OFF;
            }

            if (((x > 0.0 + EPS) && (x <= 1.00 + EPS)) && ((y > 0.0 + EPS) && (y <= 0.4 + EPS))) {  /* fluid region */
                ParticleType[i] = FLUID;
                flagOfParticleGeneration = ON;
            }

            if (flagOfParticleGeneration == ON) {
                PositionX[i] = x; PositionY[i] = y; PositionZ[i] = z;
                i++;
            }
        }
    }
    NumberOfParticles = i;
    for (i = 0; i < NumberOfParticles; i++) { VelocityX[i] = 0.0; VelocityY[i] = 0.0; VelocityZ[i] = 0.0; }

    //ParticleType[719] = 19;
}

void initializeParticlePositionAndVelocity_for3dim(void) {
    int iX, iY, iZ;
    int nX, nY, nZ;
    double x, y, z;
    int i = 0;
    int flagOfParticleGeneration;

    nX = (int)(1.0 / PARTICLE_DISTANCE) + 5;
    nY = (int)(0.7 / PARTICLE_DISTANCE) + 5;
    nZ = (int)(0.3 / PARTICLE_DISTANCE) + 5;
    for (iX = -4; iX < nX; iX++) {
        for (iY = -4; iY < nY; iY++) {
            for (iZ = -4; iZ < nZ; iZ++) {
                x = PARTICLE_DISTANCE * iX;
                y = PARTICLE_DISTANCE * iY;
                z = PARTICLE_DISTANCE * iZ;
                flagOfParticleGeneration = OFF;

                /* dummy wall region */
                if ((((x > -4.0 * PARTICLE_DISTANCE + EPS) && (x <= 1.00 + 4.0 * PARTICLE_DISTANCE + EPS)) && ((y > 0.0 - 4.0 * PARTICLE_DISTANCE + EPS) && (y <= 0.7 + EPS))) && ((z > 0.0 - 4.0 * PARTICLE_DISTANCE + EPS) && (z <= 0.3 + 4.0 * PARTICLE_DISTANCE + EPS))) {
                    ParticleType[i] = DUMMY_WALL;
                    flagOfParticleGeneration = ON;
                }

                /* wall region */
                if ((((x > -2.0 * PARTICLE_DISTANCE + EPS) && (x <= 1.00 + 2.0 * PARTICLE_DISTANCE + EPS)) && ((y > 0.0 - 2.0 * PARTICLE_DISTANCE + EPS) && (y <= 0.6 + EPS))) && ((z > 0.0 - 2.0 * PARTICLE_DISTANCE + EPS) && (z <= 0.3 + 2.0 * PARTICLE_DISTANCE + EPS))) {
                    ParticleType[i] = WALL;
                    flagOfParticleGeneration = ON;
                }

                /* wall region */
                /*if ((((x > -4.0 * PARTICLE_DISTANCE + EPS) && (x <= 1.00 + 4.0 * PARTICLE_DISTANCE + EPS)) && ((y > 0.6 - 2.0 * PARTICLE_DISTANCE + EPS) && (y <= 0.6 + EPS))) && ((z > 0.0 - 4.0 * PARTICLE_DISTANCE + EPS) && (z <= 0.3 + 4.0 * PARTICLE_DISTANCE + EPS))) {
                    ParticleType[i] = WALL;
                    flagOfParticleGeneration = ON;
                }*/

                /* empty region */
                if ((((x > 0.0 + EPS) && (x <= 1.00 + EPS)) && (y > 0.0 + EPS)) && (y < 0.5 + EPS) && ((z > 0.0 + EPS) && (z <= 0.3 + EPS))) {
                    flagOfParticleGeneration = OFF;
                }

                /* fluid region */
                if ((((x > 0.0 + EPS) && (x <= 0.25 + EPS)) && ((y > 0.0 + EPS) && (y < 0.5 + EPS))) && ((z > 0.0 + EPS) && (z <= 0.3 + EPS))) {
                    ParticleType[i] = FLUID;
                    flagOfParticleGeneration = ON;
                }

                if (flagOfParticleGeneration == ON) {
                    PositionX[i] = x;
                    PositionY[i] = y;
                    PositionZ[i] = z;
                    i++;
                }
            }
        }
    }
    NumberOfParticles = i;//2D num=1216 3D num=5652
    for (i = 0; i < NumberOfParticles; i++) { VelocityX[i] = 0.0; VelocityY[i] = 0.0; VelocityZ[i] = 0.0; }
}

void calculateConstantParameter(void) {

    Re_forParticleNumberDensity = RADIUS_FOR_NUMBER_DENSITY;
    Re_forGradient = RADIUS_FOR_GRADIENT;
    Re_forLaplacian = RADIUS_FOR_LAPLACIAN;
    Re2_forParticleNumberDensity = Re_forParticleNumberDensity * Re_forParticleNumberDensity;
    Re2_forGradient = Re_forGradient * Re_forGradient;
    Re2_forLaplacian = Re_forLaplacian * Re_forLaplacian;
    calculateNZeroAndLambda();
    FluidDensity = FLUID_DENSITY;
    collisionDistance = COLLISION_DISTANCE;
    collisionDistance2 = collisionDistance * collisionDistance;
    FileNumber = 0;
    Time = 0.0;
}

void calculateNZeroAndLambda(void) {
    int iX, iY, iZ;
    int iZ_start, iZ_end;
    double xj, yj, zj, distance, distance2;
    double xi, yi, zi;

    if (DIM == 2) {
        iZ_start = 0; iZ_end = 1;
    }
    else {
        iZ_start = -4; iZ_end = 5;
    }

    N0_forParticleNumberDensity = 0.0;
    N0_forGradient = 0.0;
    N0_forLaplacian = 0.0;
    Lambda = 0.0;
    xi = 0.0;  yi = 0.0;  zi = 0.0;

    for (iX = -4; iX < 5; iX++) {
        for (iY = -4; iY < 5; iY++) {
            for (iZ = iZ_start; iZ < iZ_end; iZ++) {
                if (((iX == 0) && (iY == 0)) && (iZ == 0))continue;
                xj = PARTICLE_DISTANCE * (double)(iX);
                yj = PARTICLE_DISTANCE * (double)(iY);
                zj = PARTICLE_DISTANCE * (double)(iZ);
                distance2 = (xj - xi) * (xj - xi) + (yj - yi) * (yj - yi) + (zj - zi) * (zj - zi);
                distance = sqrt(distance2);
                N0_forParticleNumberDensity += weight(distance, Re_forParticleNumberDensity);
                N0_forGradient += weight(distance, Re_forGradient);
                N0_forLaplacian += weight(distance, Re_forLaplacian);
                Lambda += distance2 * weight(distance, Re_forLaplacian);
            }
        }
    }
    Lambda = Lambda / N0_forLaplacian;
}

double weight(double distance, double re) {
    double weightij;

    if (distance >= re) {
        weightij = 0.0;
    }
    else {
        weightij = re / (0.85 * distance + 0.15 * re) - 1.0;
    }
    return weightij;
}
int main()
{
    //CUDA 初始化
    if (!InitCUDA()) {
        return 0;
    }
    //cpu
    printf("\n*** START MPS-SIMULATION ***\n");
    if (DIM == 2) {
        initializeParticlePositionAndVelocity_for2dim();
    }
    else {
        initializeParticlePositionAndVelocity_for3dim();
    }

    calculateConstantParameter();
}
