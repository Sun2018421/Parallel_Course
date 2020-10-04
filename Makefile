make:
	gcc -O3 -g -fopenmp -lm main.c -o sdnn
	gcc -O3 -fopenmp -lm main_autothread.c -o sdnn_autothread
