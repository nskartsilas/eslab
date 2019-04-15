# eslab
Embedded Systems Lab

Platform-specific implementation (in C) of a Mean Shift Tracking application on a multi-core heterogeneous computing platform. Migration of the application, designed for a desktop computer, to an ARM® Cortex™-A8 featuring development board (Beagleboard), to meet specific throughput requirements.

The minimum requirement is to have an ARM+DSP+NEON optimized implementation of the Mean-Shift Algorithm, which runs at least 4x faster than the baseline version (the start and stop timers are marked in the main.cpp). Baseline version is the solution running on ARM alone with -O3 optimization level. The speedup is calculated as follows:  Speedup = (Execution Time Baseline) / (Execution Time Optimized)

Folders content
Original_Codes: Contains the provided sample program, showing the Mean Shift algorithm and how to pass messages between GPP (ARM core) and DSP
Final_Code_Implementation: Contains the three versions of the algorithm 
