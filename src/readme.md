Source code for CGZero.

More info at /markdown folder

g++ src/NNSampler.cpp -std=c++17 -o NNSampler
g++ src/CGZero.cpp -std=c++17 -march=native -O2 -mfma -mf16c -mavx -mavx2 -o CGZero -lpthread