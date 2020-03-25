
/**
   
    Author : Chares Moustakas 
    AEM :    8860 
    E-mail:  cmoustakas@ece.auth.gr
    Course : Parallel & Distributed Systems
    Profs:   
	     Nikolaos Pitsianis , pitsiani@ece.auth.gr
	     Dimitrios Floros   , fcdimitri@auth.gr

**/

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

float *readAdjacency(int N,char*filename);
int minimum(int a,int b);


cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE ;
cusparseDirection_t direction = CUSPARSE_DIRECTION_ROW ;


	

__global__ void hadamardKernel(int*csrRowPtrA,int*csrColIdxA,int* csrRowPtrC,int*csrColIdxC,int*csrRowPtrV,int*csrColIdxV,int N,int rounds,int* lastPtr,int top){

    int offset = rounds*513;
    int uId = threadIdx.x + blockIdx.x*blockDim.x + offset ;
    int columnCnt; 
    int row;
    int tempArray[10];
    int host;
    int limit = top + rounds;
    



    if(uId == 0 ){csrRowPtrV[uId] = 0;return;}
    else if(uId < N+1){
   
	
	for(int i =0;i<10;i++)tempArray[i]=0;	


	row = uId-1;	
	int lastPointA,startPointA;
	int lastPointC,startPointC;

	startPointA = csrRowPtrA[row]-1; 
	startPointC = csrRowPtrC[row]-1; 

	lastPointA = csrRowPtrA[uId]-2; 
	lastPointC = csrRowPtrC[uId]-2; 

	int elementA = csrColIdxA[startPointA];  
	int elementC = csrColIdxC[startPointC];  
	
	if(uId == offset && rounds>0)csrRowPtrV[uId-1] = lastPtr[0];
        csrRowPtrV[uId] = 0;
        columnCnt = 0;

	while(1){		
		if(elementA > csrColIdxC[lastPointC] || elementC > csrColIdxA[lastPointA] || startPointA > lastPointA || startPointC > lastPointC)break;
		else if(elementA>elementC)startPointC++;
		else if(elementA<elementC)startPointA++;
		else if(elementA==elementC){
			startPointC++;
			startPointA++;		
			tempArray[columnCnt] = elementA-1;
			columnCnt++;	
			csrRowPtrV[uId]++;
		}
		elementA = csrColIdxA[startPointA];
		elementC = csrColIdxC[startPointC];			
			
	}
     
   
     }
     host = csrRowPtrV[uId];
     
     __syncthreads();
     if(uId==limit){

	if(uId > N)limit = N;
        
		
	for(int i=offset; i < limit+1; i++)csrRowPtrV[i] += csrRowPtrV[i-1];
	lastPtr[0] = csrRowPtrV[limit]; 		//[-][-] Caution suspect overflow storage
	 
	}
     
     __syncthreads(); 
     if(uId<N+1 && uId > 0){
	for(int i=0;i<host;i++)
             csrColIdxV[csrRowPtrV[uId-1] + i] = tempArray[i];
     	}


} 






int main(int argc,char*argv[]){


   if(argc !=3){ printf("Usage : ./triangs -arg[1] = index -arg[1] = filename.txt \n");return 1;}

    // --- Host side Adjacency dense matrix
   int ind = atoi(argv[1]);
   int N = pow(2,ind);
   //printf("Init Matrix \n {Rows x Columns} = %d x %d \n",N,N);
   char* filename = argv[2];
   float *h_A_dense = (float*)malloc(N * N * sizeof(float));
   h_A_dense = readAdjacency(N,filename);
  
    int rounds = N/513 + 1;
    	


//  Initialize cuSPARSE
    cusparseHandle_t handle;    

    
// SetUp My Descriptor
    cusparseMatDescr_t descr ;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);

// SetUp  Attributes :::
    cudaError_t cudaStat = cudaSuccess ;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS ;
    status = cusparseCreate(&handle);
    assert(status == CUSPARSE_STATUS_SUCCESS);

//Generate Kernel Streams [+]
       
    cudaStream_t *kernel_stream;
    kernel_stream = (cudaStream_t*)malloc(rounds*sizeof(cudaStream_t));

    for(int i = 0;i<rounds;i++){
        cudaStat = cudaStreamCreate(&kernel_stream[i]);
    	assert(cudaStat == cudaSuccess);
    }


// Generate MemCopy Streams [+]

     cudaStream_t cpyStream[6];
     for(int i = 0;i<6;i++){
	cudaStat = cudaStreamCreate(&cpyStream[i]);
	assert(cudaStat == cudaSuccess);
     }



// Setup My Device Matrix
    float *device_dense ;
    cudaMalloc(&device_dense,N*N*sizeof(float));
    cudaMemcpyAsync(device_dense,h_A_dense,N*N*sizeof(float),cudaMemcpyHostToDevice,cpyStream[0]);
    free(h_A_dense);

    int nnz = 0;
    const int lda = N  ;

//Setup Non Zero Elements Per Row OF Matrix - Device Side
    int *device_nnz ;
    cudaMalloc(&device_nnz,N*sizeof(int));
    status  =  cusparseSnnz(handle,direction,N,N,descr,device_dense,lda,device_nnz,&nnz);
    assert(status == CUSPARSE_STATUS_SUCCESS);
	
// Init SPARSE Matrix


    int* deviceRowptr;
    int* deviceColVector;
    float* deviceValVector;

    cudaMalloc(&deviceRowptr,(N+1)*sizeof(int));
    cudaMalloc(&deviceColVector,nnz*sizeof(int));
    cudaMalloc(&deviceValVector,nnz*sizeof(float));

    cusparseSdense2csr(handle,N,N,descr,device_dense,lda,device_nnz,deviceValVector,deviceRowptr,deviceColVector);
    cudaFree(device_dense);



   /**
        Calculation of C =  A * A in csr format 
	[+][+][+][+][+][+][+][+][+][+][+]
   **/
   


   int baseC,nnzC ;
   int *nnzTotalDevHostPtr = (int*)malloc(sizeof(int));
   cusparseSetPointerMode(handle,CUSPARSE_POINTER_MODE_HOST);
   
   int* csrRowPtrC ;
   cudaMalloc(&csrRowPtrC,sizeof(int)*(N+1));

   cusparseMatDescr_t descrC ; 									// SetUp descriptor for the C array
   cusparseCreateMatDescr(&descrC);
   cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ONE);
   
   
   cusparseXcsrgemmNnz(handle,trans,trans,N,N,1, 						// Handling And attributes ..
		       	   	   descr,nnz,deviceRowptr,deviceColVector, 			// Attributes Of Array A 
		       	   	   descr,nnz,deviceRowptr,deviceColVector,descrC, 		// Attributes Of Array B (but yall know B = A :O ) and the descriptor for C
		       	   	   csrRowPtrC,nnzTotalDevHostPtr);
                       										// Output of the function ** Total NonZero Values And Row Pointer Matrix 

    if(NULL != nnzTotalDevHostPtr)
	nnzC = *nnzTotalDevHostPtr ;
    else{
    	cudaMemcpyAsync(&nnz,csrRowPtrC+N,sizeof(int),cudaMemcpyDeviceToHost,cpyStream[1]);
    	cudaMemcpyAsync(&baseC,deviceRowptr,sizeof(int),cudaMemcpyDeviceToHost,cpyStream[2]);
    	nnz = nnz -baseC ;
        									// So now you got the host of non zero elements in your new array .. Lets calculate it then !
    }
    int* csrColIdxC ;
    float* csrValueC ;	
    
    cudaMalloc(&csrColIdxC,sizeof(int)*nnzC);
    cudaMalloc(&csrValueC,sizeof(float)*nnzC);
    
    clock_t start = clock();    

    cusparseScsrgemm(handle,trans,trans,N,N,1,
	                descr,nnz,deviceValVector,deviceRowptr,deviceColVector,
	                descr,nnz,deviceValVector,deviceRowptr,deviceColVector,
	                descrC,csrValueC,csrRowPtrC,csrColIdxC);
   

   
   if(nnzC == 0){
	printf("[-][-][-][-] Number oF Triangles is Zero \n\n");
	clock_t stop = clock();
	float time = (float)(stop-start)/CLOCKS_PER_SEC;
		
	printf("Execution time : %f sec \n",time); 
	return 0;	
    
    }   	

//	Print The Csr In Dense ..


   float* C_Array = (float*)malloc(N*N*sizeof(float));
   float* device_C_Array ;
   cudaMalloc(&device_C_Array,N*N*sizeof(float));
   cusparseScsr2dense(handle,N,N,descrC,csrValueC,csrRowPtrC,csrColIdxC,device_C_Array,N);
   cudaMemcpyAsync(C_Array,device_C_Array,sizeof(float)*N*N,cudaMemcpyDeviceToHost,cpyStream[3]); 
   cudaFree(device_C_Array);
   

   
/**

deviceRowptr     -----------+
deviceColVector             |-----> Vector A  
deviceValVector  -----------+


csrRowPtrC       -----------+
csrColIdxC                  |-----> Vector C = A x A
csrValueC        -----------+

hadamardKernel(float*csrValueA,int*csrRowPtrA,int*csrColIdxA,float* csrValueC,float* csrValueV,int* csrRowPtrC,int* csrRowPtrV,int*csrColIdxC,int* csrColIndxV)

   [+][+][+][+][+][+][+]  So Now I Need To Multiply (Hadamard) Elementwisely C And A. 
   

**/
    
    
    int *d_csrRowPtrV,*d_csrColIndxV,*d_lastPtr;
    
    cudaMalloc(&d_lastPtr,sizeof(int));
    cudaMalloc(&d_csrRowPtrV,(N+1)*sizeof(int));
    int minNnz = minimum(nnzC,nnz);
    cudaMalloc(&d_csrColIndxV,minNnz*sizeof(int));

    
   
    int top;
    if(N>513) top = 512;
    else top = N;
    
    int blocks = 1;
    int threads = top+1;


    
    for(int i = 0;i<rounds;i++){  
        hadamardKernel<<<blocks,threads,0,kernel_stream[i]>>>(deviceRowptr,deviceColVector,
					              csrRowPtrC,csrColIdxC,
						      d_csrRowPtrV,d_csrColIndxV,
						      N,i,d_lastPtr,top);
        
	top += 512 ;
        cudaStreamSynchronize(kernel_stream[i]);
    }
  
   
  
    for(int i =0;i<rounds;i++)cudaStat = cudaStreamDestroy(kernel_stream[i]);
    cudaError_t err = cudaGetLastError();
    if(cudaStat != cudaSuccess)printf("[-][-] Error : %s \n",cudaGetErrorString(err));
    

    int*h_csrRowPtrV = (int*)malloc((N+1)*sizeof(int));
    int *h_csrColIdxV = (int*)malloc(minNnz*sizeof(int));
    cudaMemcpyAsync(h_csrColIdxV,d_csrColIndxV,minNnz*sizeof(int),cudaMemcpyDeviceToHost,cpyStream[4]);
    cudaMemcpyAsync(h_csrRowPtrV,d_csrRowPtrV,(N+1)*sizeof(int),cudaMemcpyDeviceToHost,cpyStream[5]);
    
    //cudaStreamSynchronize(cpyStream[5]);
    


//Pretty Interesting Line, Uncomment If you are Enemy Of Performance


//  cudaDeviceReset();





    
  /**
	[+] Number Of Triangles Calculation
  **/
   
   long unsigned int host_cnt,host,col;
   long long unsigned int  row ;
    long long int sum=0;
    
    for(long int i = 0;i<N;i++){

	 host_cnt = 0;	
	 host = h_csrRowPtrV[i+1]-h_csrRowPtrV[i];
	 while(host_cnt<host){
	       row =N*h_csrColIdxV[h_csrRowPtrV[i]+host_cnt];
	       col=i;
	       sum += C_Array[row + col];
	       host_cnt++;
	
	}
     }
	     
	      
    long int numOfT = (int)(sum/6);
  


    clock_t stop = clock();
    float time = (float)(stop-start)/CLOCKS_PER_SEC;	
     
    cudaDeviceReset();

    printf("\n[+][+][+][+][+] Number Of Triangles : %ld \n",numOfT);
    printf("Execution time : %f sec \n",time);


    return 0;

}



int minimum(int a,int b){
   if(a<b)return a;
   else  return b;
}



float *readAdjacency(int N,char* filename){
    FILE *fp;
    float *array,value;
    array = (float*)malloc(N*N*sizeof(float));
    int counter=0;
    fp = fopen(filename,"r");
    if(fp==NULL){return NULL;}
    int i=10,offset = 0;

    while(counter<N*N){
	if(i == EOF){printf("Not big enough txt file");exit(0);}        
	while(offset<N){	
    		i=fscanf(fp,"%f",&value);
    		array[counter+offset] = value;
    	        offset++;
        }
	i = fscanf(fp,"%*[^\n]\n");	
	counter = counter+N;
	offset = 0;
    }
    
    
    if(N<17){
        printf("Array = [");
        for(int k = 0 ;k<counter ;k++){
	    if(!(k%N))printf("\n");        
	    printf("%f ",array[k]);
	
        } 
        printf("] \n");
    }
    
    fclose(fp);
    return array;   
}

