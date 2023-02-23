<p align="center">
 <a href="https://github.com/Goubermouche/gputil">
    <img src="https://github.com/Goubermouche/gputil/blob/da49f416453a22116993233e623ad0c0acfedfff/misc/header.png" width="800" alt="gputil logo">
   </a>
</p>

**gputil** is a simple utility library for [C++ 20](https://cplusplus.com/) and [CUDA](https://developer.nvidia.com/cuda-toolkit) interoperation, it provides basic containers you know and love from the standart library and a few of its own. Note that while I have tried to prioritize performance and all-around correctess, there may be, and probably are, certain performance problems/slight unoptimizations. If you find one of these please feel free to create an issue and I'll try to fix it as soon as possible. 

## Getting up and running
<ins>**1. Downloading CUDA**</ins>   
Download the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and follow the setup instructions for your system. The project runs on CUDA toolkit version 12.0

<ins>**Linking**</ins>
After you've downloaded the CUDA toolkit you'll have to link agains some cuda libraries - specifically `cuda.lib` and `nvrtc.lib` - go into **Properties > Linker > Input > Additional Dependencies** and add the two .lib files. After linking against the two libraries go into **Properties > Linker > General > Additional Library Directories** and include the include directory of our CUDA libraries (default is `$(CUDA_PATH)\lib\x64`). Finally, go into **Properties > C/C++ > General > Additional Include Directories** and include CUDA headers (default is `$(CUDA_PATH)\include`).

*Note that gputil is not, in any way, shape, or form, associated with Nvidia.*
