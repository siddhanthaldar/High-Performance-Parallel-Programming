Instructions to run code:

openmp.cpp --> g++ -fopenmp openmp.cpp -o openmp  --> ./openmp  (OpenMP code for 4 and 16 cores)
serial.cpp --> g++ -fopenmp serial.cpp -o serial  --> ./serial	(Serial Code)
compare.cpp --> g++ -fopenmp compare.cpp -o compare --> ./compare  (Code to compare performance of Serial code and the codes using 4 and 16 cores)

Time Logs :
serial.txt   -> time log for prime number identification for serial code
4core.txt   -> time log for prime number identification for OpenMP code using 4 cores
16core.txt   -> time log for prime number identification for OpenMP code using 4 cores

Performance comparison is given in "performance.txt"

