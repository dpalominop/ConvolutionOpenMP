#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <omp.h>

#define dim_kernel 5
#define dim_image 2000
#define size_kernel dim_kernel*dim_kernel
#define li_image dim_kernel/2
#define ls_image dim_image-li_image


#define length(x) (sizeof(x)/sizeof(x[0]))

using namespace std;

int delete_memory(int** matrix, int n);
int reserve_memory(int** matrix, int n);

int convolucion(int** kernel, int** image, int** result, int thread_count);
int show_data(int** result, int n);
int generate_data(int** matrix, int n);

int main(int argc, char** argv) {

  int thread_count;
  bool show=false;

  if (argc == 2){
    thread_count=strtol(argv[1], NULL, 10);
  }else{
	  if (argc==3){
		  thread_count=strtol(argv[1], NULL, 10);
		  show=true;
	  }else{
		  return 0;
	  }
  }

  int** kernel;
  int** image;
  int** result;

  kernel = new int*[dim_kernel];
  for (int i = 0; i < dim_kernel; i++) {
    kernel[i] = new int[dim_kernel];
  }

  image = new int*[dim_image];
  for (int i = 0; i <dim_image; i++) {
    image[i] = new int[dim_image];
  }

  result = new int*[dim_image];
  for (int i = 0; i <dim_image; i++) {
    result[i] = new int[dim_image];
  }

  double elapsed = 10;
  for(int i=0;i<100;i++){
	  generate_data(kernel, dim_kernel);
	  generate_data(image, dim_image);

	  //omp_set_nested (1);
	  double time_init = omp_get_wtime();
	  convolucion(kernel, image, result, thread_count);
	  double time_final = omp_get_wtime();

	  double n_elapsed = time_final-time_init;
	  if(n_elapsed < elapsed) elapsed = n_elapsed;
  }
  cout<<"Minimal Total Time: "<< elapsed << endl;

  if (show){
	  show_data(kernel, dim_kernel);
	  show_data(image, dim_image);
	  show_data(result, dim_image);
  }

  for(int i = 0; i < dim_kernel; i++){
    delete[] kernel[i];
  }

  delete[] kernel;

  for(int i = 0; i < dim_image; i++){
    delete[] image[i];
  }

  delete[] image;

  for(int i = 0; i < dim_image; i++){
    delete[] result[i];
  }

  delete[] result;

  return 0;
}

int convolucion(int** kernel, int** image, int** result, int thread_count)
{
	//Cuadrado central
	#pragma omp parallel for collapse(2) num_threads(thread_count) shared(kernel, image, result, thread_count)
	for (int i = 2; i < ls_image; i++){
	  for (int j = 2; j < ls_image; j++){
		  int acumulador = 0;
		  int* krow;
		  int* irow;

		  /*for (int m = 0; m < 2; m++){
			  krow = kernel[m];
			  irow = image[i + (m - 2)];
			  acumulador += krow[0]*irow[j + -2];
			  acumulador += krow[1]*irow[j + -1];
			  acumulador += krow[2]*irow[j + 0];
			  acumulador += krow[3]*irow[j + 1];
			  acumulador += krow[4]*irow[j + 2];
		  }

		  for (int m = 2; m < 4; m++){
			  krow = kernel[m];
			  irow = image[i + (m - 2)];
			  acumulador += krow[0]*irow[j + -2];
			  acumulador += krow[1]*irow[j + -1];
			  acumulador += krow[2]*irow[j + 0];
			  acumulador += krow[3]*irow[j + 1];
			  acumulador += krow[4]*irow[j + 2];
		  }*/
		  krow = kernel[0];
		  irow = image[i-2];
		  acumulador += krow[0]*irow[j + -2];
		  acumulador += krow[1]*irow[j + -1];
		  acumulador += krow[2]*irow[j + 0];
		  acumulador += krow[3]*irow[j + 1];
		  acumulador += krow[4]*irow[j + 2];

		  krow = kernel[1];
		  irow = image[i-1];
		  acumulador += krow[0]*irow[j + -2];
		  acumulador += krow[1]*irow[j + -1];
		  acumulador += krow[2]*irow[j + 0];
		  acumulador += krow[3]*irow[j + 1];
		  acumulador += krow[4]*irow[j + 2];

		  krow = kernel[2];
		  irow = image[i];
		  acumulador += krow[0]*irow[j + -2];
		  acumulador += krow[1]*irow[j + -1];
		  acumulador += krow[2]*irow[j + 0];
		  acumulador += krow[3]*irow[j + 1];
		  acumulador += krow[4]*irow[j + 2];

		  krow = kernel[3];
		  irow = image[i+1];
		  acumulador += krow[0]*irow[j + -2];
		  acumulador += krow[1]*irow[j + -1];
		  acumulador += krow[2]*irow[j + 0];
		  acumulador += krow[3]*irow[j + 1];
		  acumulador += krow[4]*irow[j + 2];

		  krow = kernel[4];
		  irow = image[i+2];
		  acumulador += krow[0]*irow[j + -2];
		  acumulador += krow[1]*irow[j + -1];
		  acumulador += krow[2]*irow[j + 0];
		  acumulador += krow[3]*irow[j + 1];
		  acumulador += krow[4]*irow[j + 2];

		  result[i][j] = acumulador/25;
	  }
	}


	//Cálculo de los bordes sin ezquinas
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			int* krow;
			int* irow;
			for (int i=2; i<ls_image; i++){
			  for (int j=0; j<2; j++){
				  int acumulador = 0;

				  for (int m = 0; m < 5; m++){
					  krow = kernel[m];
					  irow = image[i + (m - 2)];

					  if ((j-2) >= 0){
						  acumulador += krow[0]*irow[j-2];
					  }
					  if ((j-1) >= 0){
						  acumulador += krow[1]*irow[j-1];
					  }
					  acumulador += krow[2]*irow[j];
					  acumulador += krow[3]*irow[j+1];
					  acumulador += krow[4]*irow[j+2];
				  }
				  result[i][j] = acumulador/(15+5*j);
			  }
			}
		}

		#pragma omp section
		{
			for (int i=2; i<ls_image; i++){
			  for (int j=ls_image; j<dim_image; j++){
				  int acumulador = 0;

				  for (int m = 0; m < 5; m++){
					  int* krow = kernel[m];
					  int* irow = image[i + (m - 2)];

					  acumulador += krow[0]*irow[j-2];
					  acumulador += krow[1]*irow[j-1];
					  acumulador += krow[2]*irow[j];
					  if ((j+1) < dim_image){
						  acumulador += krow[3]*irow[j+1];
					  }
					  if ((j+2) < dim_image){
						  acumulador += krow[4]*irow[j+2];
					  }
				  }
				  result[i][j] = acumulador/(15+5*(dim_image-1-j));
			  }
			}
		}

		#pragma omp section
		{
			for (int i=0; i<2; i++){
			  for (int j=2; j<ls_image; j++){
				  int ii;
				  int acumulador = 0;

				  for (int m = 0; m < 5; m++){
					  int* krow = kernel[m];
					  ii = i + (m - 2);
					  int* irow = image[ii];
					  if (ii >= 0){
						  for (int n = 0; n < 5; n++){
							  acumulador += krow[n]*irow[j + (n - 2)];
						  }
					  }
				  }
				  result[i][j] = acumulador/(15+5*i);
			  }
			}
		}

		#pragma omp section
		{
			for (int i=ls_image; i<dim_image; i++){
			  for (int j=2; j<ls_image; j++){
				  int ii;
				  int acumulador = 0;

				  for (int m = 0; m < 5; m++){
					  int* krow = kernel[m];
					  ii = i + (m - 2);
					  int* irow = image[ii];
					  if (ii < dim_image){
						  for (int n = 0; n < 5; n++){
							  acumulador += krow[n]*irow[j + (n - 2)];
						  }
					  }
				  }
				  result[i][j] = acumulador/(15+5*(dim_image-1-i));
			  }
			}
		}
	}

	//Cálculo de ezquinas
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			for (int i=0; i<2; i++){
			  for (int j=0; j<2; j++){
				  int ii,jj;
				  int acumulador = 0;
				  int num = 0;

				  for (int m = 0; m < 5; m++){
					  int* krow = kernel[m];
					  ii = i + (m - 2);
					  int* irow = image[ii];
					  for (int n = 0; n < 5; n++){
						  jj = j + (n - 2);
						  if (ii >= 0 && jj >= 0){
							  acumulador += krow[n]*irow[jj];
							  num++;
						  }
					  }
				  }
				  result[i][j] = acumulador/num;
			  }
			}
		}
		#pragma omp section
		{
			for (int i=ls_image; i<dim_image; i++){
			  for (int j=0; j<2; j++){
				  int ii,jj;
				  int acumulador = 0;
				  int num = 0;

				  for (int m = 0; m < 5; m++){
					  int* krow = kernel[m];
					  ii = i + (m - 2);
					  int* irow = image[ii];
					  for (int n = 0; n < 5; n++){
						  jj = j + (n - 2);
						  if (ii < dim_image && jj >= 0){
							  acumulador += krow[n]*irow[jj];
							  num++;
						  }
					  }
				  }
				  result[i][j] = acumulador/num;
			  }
			}
		}
		#pragma omp section
		{
			for (int i=0; i<2; i++){
			  for (int j=ls_image; j<dim_image; j++){
				  int ii,jj;
				  int acumulador = 0;
				  int num = 0;

				  for (int m = 0; m < 5; m++){
					  int* krow = kernel[m];
					  ii = i + (m - 2);
					  int* irow = image[ii];
					  for (int n = 0; n < 5; n++){
						  jj = j + (n - 2);
						  if (ii >= 0 && jj < dim_image){
							  acumulador += krow[n]*irow[jj];
							  num++;
						  }
					  }
				  }
				  result[i][j] = acumulador/num;
			  }
			}
		}
		#pragma omp section
		{
			for (int i=ls_image; i<dim_image; i++){
			  for (int j=ls_image; j<dim_image; j++){
				  int ii,jj;
				  int acumulador = 0;
				  int num = 0;

				  for (int m = 0; m < 5; m++){
					  int* krow = kernel[m];
					  ii = i + (m - 2);
					  int* irow = image[ii];
					  for (int n = 0; n < 5; n++){
						  jj = j + (n - 2);
						  if (ii < dim_image && jj < dim_image){
							  acumulador += krow[n]*irow[jj];
							  num++;
						  }
					  }
				  }
				  result[i][j] = acumulador/num;
			  }
			}
		}
	}

	return 0;
}

int reserve_memory(int** matrix, int n){

  matrix = new int* [n];
  for (int i = 0; i < n; i++) {
    matrix[i] = new int[n];
  }

  return 0;
}

int delete_memory(int** matrix, int n){
  for(int i = 0; i < n; i++){
    delete[] matrix[i];
  }

  delete[] matrix;

  return 0;
}

int generate_data(int** matrix, int n){
  srand(time(NULL));

  for(int i=0; i<n ; i++){
    for(int j=0; j<n ; j++){
      matrix[i][j] = rand()%3;
    }
  }

  return 0;
}

int show_data(int** result, int n){
	  for(int i = 0; i < n; i++)
	  {
	      for(int j = 0; j < n; j++)
	      {
	          cout<<" "<<result[i][j];
	      }
	      cout<<endl;
	  }

  return 0;
}

