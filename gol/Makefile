# When gol-par.c has been created, uncomment the next line:
all: gol-seq gol-par gol-par-bonus-blocking gol-par-bonus-non-blocking

gol-seq: gol-seq.c
	gcc -Wall -O3 -o gol-seq gol-seq.c -lm

gol-par: gol-par.c
	mpicc -Wall -O3 -o gol-par gol-par.c StopWatch/StopWatch.c -lm

gol-par-bonus-blocking: gol-par-bonus.c
	mpicc -Wall -O3 -o gol-par-bonus-blocking gol-par-bonus.c StopWatch/StopWatch.c -lm

gol-par-bonus-non-blocking: gol-par-bonus.c
	mpicc -Wall -O3 -o gol-par-bonus-non-blocking -D NON_BLOCKING gol-par-bonus.c StopWatch/StopWatch.c -lm

clean:
	rm -f *.o gol-seq gol-par gol-par-bonus-blocking gol-par-bonus-non-blocking *~ *core
