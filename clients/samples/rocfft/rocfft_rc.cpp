// Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <boost/program_options.hpp>
#include <complex>
#include <iostream>
#include <numeric>
#include <functional>
#include <vector>


#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <rocfft.h>

namespace po = boost::program_options;

// Kernel for initializing 1D input data on the GPU.
__global__ void initrdata(double* x, const int Nx)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < Nx)
    {
        x[idx] = idx;
    }
}

// Kernel for initializing 2D input data on the GPU.
__global__ void initrdata(double* x, const int Nx, const int Ny)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < Nx && idy < Ny)
    {
        x[idx + idy * Nx] = idx + idy;
    }
}

// Kernel for initializing 3D input data on the GPU.
__global__ void initrdata(double* x, const int Nx, const int Ny, const int Nz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if(idx < Nx && idy < Ny && idz < Nz)
    {
        x[idx + idy * Nx + idz * Nx * Ny] = idx + idy + idz;
    }
}

// Increment the index (column-major) for looping over arbitrary dimensional loops with
// dimensions length.
template <class T1, class T2>
bool increment_colmajor(std::vector<T1>& index, const std::vector<T2>& length)
{
    for(int idim = 0; idim < length.size(); ++idim)
    {
        if(index[idim] < length[idim])
        {
            if(++index[idim] == length[idim])
            {
                index[idim] = 0;
                continue;
            }
            break;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

// Output a formatted general-dimensional array with given length and stride in batches
// separated by dist.
template <class Toutput, class T1, class T2>
void printoutput(const std::vector<Toutput>& output,
                 const std::vector<T1>       length,
                 const std::vector<T2>       stride,
                 const size_t                nbatch,
                 const size_t                dist)
{
    for(size_t b = 0; b < nbatch; b++)
    {
        std::vector<int> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            const int i = std::inner_product(index.begin(), index.end(), stride.begin(), b * dist);
            std::cout << i << ": ";            
            std::cout << output[i] << " ";
            for(int i = 0; i < index.size(); ++i)
            {
                if(index[i] == (length[i] - 1))
                {
                    std::cout << "\n";
                }
                else
                {
                    break;
                }
            }
        } while(increment_colmajor(index, length));
        std::cout << std::endl;
    }
}

// Helper function for determining grid dimensions
template <typename Tint1, typename Tint2>
Tint1 ceildiv(const Tint1 nominator, const Tint2 denominator)
{
    return (nominator + denominator - 1) / denominator;
}

int main(int argc, char* argv[])
{
    std::cout << "rocfft double-precision real/complex transform\n";

    // Length of transform:
    std::vector<size_t> length = {8};
    
    // Gpu device id:
    int deviceId = 0;
    
    // Command-line options:
    po::options_description desc("rocfft sample command line options");
    desc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the rocfft library")
        ("device", po::value<int>(&deviceId)->default_value(0),
         "Select a specific device id")
        ("outofplace,o", "Perform an out-of-place transform")
        ("inverse,i", "Perform an inverse transform")
        ("length",  po::value<std::vector<size_t>>(&length)->multitoken(), "Lengths.");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    // Placeness for the transform
    const rocfft_result_placement place
        = vm.count("outofplace") ? rocfft_placement_notinplace : rocfft_placement_inplace;

    // Direction of transform
    const rocfft_transform_type direction
        = vm.count("inverse") ? rocfft_transform_type_real_forward
        : rocfft_transform_type_real_inverse;

    // Set the device:
    hipSetDevice(deviceId);

    // Determine problem and buffer size:
    const size_t real_size = std::accumulate(length.begin(), length.end(),
                                             1, std::multiplies<size_t>());
    const size_t real_bytes = real_size * sizeof(double);
    std::vector<size_t> complex_length = length;
    complex_length[0] = length[0] / 2 + 1;
    const size_t complex_size = std::accumulate(complex_length.begin(), complex_length.end(),
                                                1, std::multiplies<size_t>());
    const size_t complex_bytes = real_size * sizeof(std::complex<double>);

    const size_t isize = (direction == rocfft_transform_type_real_forward)
        ? real_size : complex_size;
    const size_t ibytes =  (direction == rocfft_transform_type_real_forward)
        ? real_bytes : complex_bytes;
    const size_t osize = (direction == rocfft_transform_type_real_forward)
        ? complex_size : real_size;
    const size_t obytes = (direction == rocfft_transform_type_real_forward
         ? complex_bytes : real_bytes);
    
    // Create HIP device object and copy data to device
    void* gpu_in = NULL;
    hipMalloc(&gpu_in, ibytes);

    // Inititalize the data on the device
    hipError_t hip_status = hipSuccess;
    switch(length.size())
    {
    case 1:
    {
        const dim3 blockdim(256);
        const dim3 griddim(ceildiv(length[0], blockdim.x));
        hipLaunchKernelGGL(initrdata, blockdim, griddim, 0, 0, (double*)gpu_in,
                           length[0]);
        break;
    }
    case 2:
    {
        const dim3 blockdim(32, 32);
        const dim3 griddim(ceildiv(length[0], blockdim.x), ceildiv(length[1], blockdim.y));
        hipLaunchKernelGGL(initrdata, blockdim, griddim, 0, 0, (double*)gpu_in,
                           length[0], length[1]);
        break;
    }
    case 3:
    {
        const dim3 blockdim(32, 32, 32);
        const dim3 griddim(ceildiv(length[0], blockdim.x),
                           ceildiv(length[1], blockdim.y),
                           ceildiv(length[2], blockdim.z));
        hipLaunchKernelGGL(initrdata, blockdim, griddim, 0, 0, (double*)gpu_in,
                           length[0], length[1], length[2]);
        break;
    }
    default:
        std::cout << "invalid dimension!\n";
        exit(1);
    }
    hipDeviceSynchronize();
    hip_status = hipGetLastError();
    assert(hip_status == hipSuccess);


    std::cout << "input:\n";
    std::vector<double> idata(isize);
    hipMemcpy(idata.data(), gpu_in, ibytes, hipMemcpyDefault);
    // We need to calculate the normal input stride (column-major) for outputting
    // arbitrary-dimensional arrays:
    std::vector<size_t> istride = {1};
    for(int i = 1; i < length.size(); ++i)
    {
        istride.push_back(length[i - 1] * istride[i - 1]);
    }
    printoutput(idata, length, istride, 1, isize);

    // rocfft_status can be used to capture API status info
    rocfft_status rc = rocfft_status_success;
    
    // Create the plan
    rocfft_plan_description gpu_description = NULL;
    rocfft_plan gpu_plan = NULL;
    rc = rocfft_plan_create(&gpu_plan,
                            place,
                            direction,
                            rocfft_precision_double,
                            length.size(), // Dimension
                            length.data(), // lengths
                            1, // Number of transforms
                            gpu_description); // Description
    assert(rc == rocfft_status_success);

    // Get the execution info for the fft plan (in particular, work memory requirements):
    rocfft_execution_info planinfo = NULL;
    rc                     = rocfft_execution_info_create(&planinfo);
    assert(rc == rocfft_status_success);
    size_t workbuffersize = 0;
    rc            = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    assert(rc == rocfft_status_success);

    // If the transform requires work memory, allocate a work buffer:
    void* wbuffer = NULL;
    if(workbuffersize > 0)
    {
        hip_status = hipMalloc(&wbuffer, workbuffersize);
        assert(hip_status == hipSuccess);
        rc = rocfft_execution_info_set_work_buffer(planinfo, wbuffer, workbuffersize);
        assert(rc == rocfft_status_success);
    }

    // If the transform is out-of-place, allocate the output buffer as well:
    void* gpu_out = place == rocfft_placement_notinplace ? gpu_in : NULL;
    if(place != rocfft_placement_notinplace)
    {
        hip_status = hipMalloc(&gpu_out, obytes);
        assert(hip_status == hipSuccess);
    }

    // Execute the GPU transform:
    rc = rocfft_execute(gpu_plan, // plan
                                (void**)&gpu_in, // in_buffer
                                (void**)&gpu_out, // out_buffer
                                planinfo); // execution info

    // Get the output from the device and print to cout:
    std::cout << "output:\n";
    std::vector<std::complex<double>> odata(isize);
    hipMemcpy(odata.data(), gpu_out, obytes, hipMemcpyDeviceToHost);
    std::vector<size_t> ostride = {1};
    for(int i = 1; i < complex_length.size(); ++i)
    {
        ostride.push_back(complex_length[i - 1] * ostride[i - 1]);
    }
    printoutput(odata, complex_length, ostride, 1, osize);

    
    // Clean up: free GPU memory:
    hipFree(gpu_in);
    if(place!= rocfft_placement_notinplace)
    {
        hipFree(gpu_out);
    }
    if(wbuffer != NULL)
    {
        hipFree(wbuffer);
    }

    // Clean up: destroy plans:
    rocfft_execution_info_destroy(planinfo);
    rocfft_plan_description_destroy(gpu_description);
    rocfft_plan_destroy(gpu_plan);

    return 0;
}
