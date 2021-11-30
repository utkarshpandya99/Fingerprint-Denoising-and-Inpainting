# Fingerprint-Denoising-and-Inpainting

Data:

Arch_1_O_v1.png
Arch_1_O_v1.pgm

In Serial folder : 

Fingerprint.c
fingerprintDenoisingAndInpaintingSerial.c


In Parallel folder :

Fingerprint_naive_parallel.cu



Decription:

The fingerprint.c is the initial file in which the trials were done to obtain results.

The fingerprintDenoisingAndInpaintingSerial.c is the final commented code file with every for loop inculcated as function 
so that, after gprof, a good analysis can be obtained as to which function take more time in execution.

The normalized image is the one where contrast is raised through histogram analysis
The original image jpg file obtained from FCV database 2004 is attached.
The pgm file is attached in order to run it through the code.
The output will also be a pgm file.

The current output obtained is also attached. The accuracy of it is not upto the mark since many function we had
converted were from the numpy and scipy library and we have built it from scratch in C and are not using the libraries here.
So, the only function we are having problem in the accuracy is the image rotation function. Hence, the inaccurate output.

The Gprof analysis is also attached here as Gprof_results.txt as a reference.

The serial code in fingerprintDenoisingAndInpaintingSerial.c is commented extensively so that every part of the code
is explained in as much as possible in simple words.

The Fingerprint_naive_parallel.cu is the naive parallel implementation in CUDA C. The functions that are selected for
parallelization are gaborconvolution, ndconvolutionsincos and ndconvolutionorient due to their high time consumption 
evident in Gprof.

Regards. Thanks for reading.
