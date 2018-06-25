# ConvolutionOpenMP
Cálculo de la convolución usando OpenMP

## Compilación

g++ -Wall -fopenmp -o convolucion convolucion.cpp


## Ejecución
```
./convolucion threads print
```

* **threads**: Número de hilos para ejecutar.
* **print(Opcional)**: Etiqueta para indicar que se impriman las matrices del kernel, imagen y resultado.

## IMPORTANTE:
Si se va a imprimir las matrices, usar una imagen de tamaño reducido para apreciar correctamente los valores:
```
#define dim_image 2000
```
