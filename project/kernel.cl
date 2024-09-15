#define TILE_SIZE 16

__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth, int imageChannels)
{
    //@@ Insert code to implement matrix multiplication here
    __local float inputData_pr[TILE_SIZE][TILE_SIZE];
    int i_pr = get_local_id(0);
    int j_pr = get_local_id(1);
    int i = (get_group_id(0) * TILE_SIZE + i_pr);
    int j = (get_group_id(1) * TILE_SIZE + j_pr);
    int maskRadius = (maskWidth / 2);
    for (int k = 0; k < imageChannels; k++)
    {
        if ((i < height) && (j < width))
        {
            inputData_pr[i_pr][j_pr] = inputData[(i * width + j) * imageChannels + k];
        }
        else
        {
            inputData_pr[i_pr][j_pr] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sum = 0;
        for (int y = -maskRadius; y <= maskRadius; y++)
        {
            for (int x = -maskRadius; x <= maskRadius; x++)
            {
                int xOffset_pr = (j_pr + x);
                int yOffset_pr = (i_pr + y);
                if (((yOffset_pr >= 0) && (yOffset_pr < TILE_SIZE)) && ((xOffset_pr >= 0) && (xOffset_pr < TILE_SIZE)))
                {
                    float imagePixel = inputData_pr[yOffset_pr][xOffset_pr];
                    float maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];
                    sum += (imagePixel * maskValue);
                }
                else
                {
                    int x_offset = (j + x);
                    int y_offset = (i + y);
                    if (((y_offset >= 0) && (y_offset < height)) && ((x_offset >= 0) && (x_offset < width)))
                    {
                        float imagePixel = inputData[(y_offset * width + x_offset) * imageChannels + k];
                        float maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];
                        sum += (imagePixel * maskValue);
                    }
                }
            }
        }
        if (sum < 0)
        {
            sum = 0;
        }
        else if (sum > 1)
        {
            sum = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((i < height) && (j < width))
        {
            outputData[(i * width + j) * imageChannels + k] = sum;
        }
    }
}