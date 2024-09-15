CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../helper_lib
LDFLAGS  := ../helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

solution: ../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../helper_lib/helper_lib.a: 
	cd ../helper_lib; make

run: solution
	./solution Dataset/input_images/blocks.ppm Dataset/output_images/3x3/kernel.raw Dataset/output_images/3x3/blurred_blocks.ppm
	./solution Dataset/input_images/blocks.ppm Dataset/output_images/5x5/kernel.raw Dataset/output_images/5x5/blurred_blocks.ppm
	./solution Dataset/input_images/blocks.ppm Dataset/output_images/7x7/kernel.raw Dataset/output_images/7x7/blurred_blocks.ppm
	
	./solution Dataset/input_images/lena.ppm Dataset/output_images/3x3/kernel.raw Dataset/output_images/3x3/blurred_lena.ppm
	./solution Dataset/input_images/lena.ppm Dataset/output_images/5x5/kernel.raw Dataset/output_images/5x5/blurred_lena.ppm
	./solution Dataset/input_images/lena.ppm Dataset/output_images/7x7/kernel.raw Dataset/output_images/7x7/blurred_lena.ppm

	./solution Dataset/input_images/prague.ppm Dataset/output_images/3x3/kernel.raw Dataset/output_images/3x3/blurred_prague.ppm
	./solution Dataset/input_images/prague.ppm Dataset/output_images/5x5/kernel.raw Dataset/output_images/5x5/blurred_prague.ppm
	./solution Dataset/input_images/prague.ppm Dataset/output_images/7x7/kernel.raw Dataset/output_images/7x7/blurred_prague.ppm

clean: 
	rm -f solution
	cd ../helper_lib; make clean