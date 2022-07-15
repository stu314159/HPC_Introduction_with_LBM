#ifndef CUDAPART_H
#define CUDAPART_H

class CudaPart
{
	public:
	CudaPart(int numEle,int rank, int size);
	~CudaPart();
	void print_result();//prints just the first three and last 3 numbers
     	void increment_on_gpu();

	private:
	
	void initialize();
	int numEle;
	int rank;
	int size;
	float * data = NULL;
	float * d_data = NULL;
};



#endif
