
#define DIM                  2
#define PARTICLE_DISTANCE    0.025
#define DT                   0.001
#define OUTPUT_INTERVAL      20


/* for three-dimensional simulation */

//#define DIM                  3
//#define PARTICLE_DISTANCE    0.05
//#define DT                   0.003
//#define OUTPUT_INTERVAL      2


#define ARRAY_SIZE           9000
#define FINISH_TIME          2.0
#define KINEMATIC_VISCOSITY  (1.0E-6)
#define FLUID_DENSITY        1000.0
#define GRAVITY_X  0.0
#define GRAVITY_Y  -9.8
#define GRAVITY_Z  0.0
#define RADIUS_FOR_NUMBER_DENSITY  (2.1*PARTICLE_DISTANCE)
#define RADIUS_FOR_GRADIENT        (2.1*PARTICLE_DISTANCE)
#define RADIUS_FOR_LAPLACIAN       (3.1*PARTICLE_DISTANCE)
#define COLLISION_DISTANCE         (0.5*PARTICLE_DISTANCE)
#define THRESHOLD_RATIO_OF_NUMBER_DENSITY  0.97
#define COEFFICIENT_OF_RESTITUTION 0.2
#define COMPRESSIBILITY (0.45E-9)
#define EPS             (0.01 * PARTICLE_DISTANCE)
#define ON              1
#define OFF             0
#define RELAXATION_COEFFICIENT_FOR_PRESSURE 0.2
#define GHOST  -1
#define FLUID   0
#define WALL    2
#define DUMMY_WALL  3
#define GHOST_OR_DUMMY  -1
#define SURFACE_PARTICLE 1
#define INNER_PARTICLE   0
#define DIRICHLET_BOUNDARY_IS_NOT_CONNECTED 0
#define DIRICHLET_BOUNDARY_IS_CONNECTED     1
#define DIRICHLET_BOUNDARY_IS_CHECKED       2


#define PI 3.14159265359
#define T 1  //T为墙壁横摇的周期



 double *AccelerationX;
 double *AccelerationY;
 double *AccelerationZ;							
 int    *ParticleType;
 double *PositionX;
 double *PositionY;
 double *PositionZ;
 double *VelocityX;
 double *VelocityY;
 double *VelocityZ;
 double *Pressure;
 double *ParticleNumberDensity;
 int    *BoundaryCondition;
 double *SourceTerm;
 int    *FlagForCheckingBoundaryCondition;
 double CoefficientMatrix[ARRAY_SIZE * ARRAY_SIZE];//没处理
 double *MinimumPressure;
int    FileNumber;
double Time;




static double r[ARRAY_SIZE];
static double d[ARRAY_SIZE];
static double oldr[ARRAY_SIZE];
//static double product[ARRAY_SIZE];

static double THETA = 0.0;




int    NumberOfParticles;
int    NumberOfNonzeros;



double Re_forParticleNumberDensity, Re2_forParticleNumberDensity;
double Re_forGradient, Re2_forGradient;
double Re_forLaplacian, Re2_forLaplacian;
double N0_forParticleNumberDensity;
double N0_forGradient;
double N0_forLaplacian;
double Lambda;
double collisionDistance, collisionDistance2;
double FluidDensity;

void initializeParticlePositionAndVelocity_for2dim(void);
void initializeParticlePositionAndVelocity_for3dim(void);
void calculateConstantParameter(void);
void calculateNZeroAndLambda(void);
double weight(double distance, double re);