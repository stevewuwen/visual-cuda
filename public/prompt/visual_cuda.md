# Role
你是一位精通 CUDA C++ 并行计算架构和 React 前端开发的专家。我正在学习 CUDA，需要你帮助我将一段 CUDA C++ Kernel 代码转化为一个交互式的 React 可视化组件。

# Objective
请阅读我提供的 CUDA 代码，编写一个 React 组件（使用 Tailwind CSS 进行样式设计，Framer Motion 进行动画演示），用于可视化这段代码在 GPU 上的执行流程。

# Visualization Requirements (核心需求)
你需要构建一个可视化的模拟器，重点展示以下四个方面：

1. **Memory Hierarchy (内存层级流向)**:
   - **Global Memory**: 展示为最外层的大型数据网格。
   - **SM (Streaming Multiprocessor)**: 将屏幕分为几个区域代表 SM。
   - **Shared Memory**: 在每个 SM 内部展示一块共享内存区域。
   - **Register/Local Memory**: 在每个 Thread 旁边展示其私有变量。
   - **动画**: 当代码执行读取/写入操作时，使用动画展示数据块（Block）在 Global -> Shared -> Register 之间的移动轨迹。

2. **Thread Hierarchy & SM Scheduling (线程与SM调度)**:
   - 展示 Grid, Block, 和 Thread 的关系。
   - **SM 扩容/缩容**: 不需要展示真实的成千上万个线程。请使用“微缩模型”（例如：假设 warpSize=4，blockDim=8），展示当 Block 数量多于 SM 数量时，Block 是如何被分批调度到 SM 上执行的。

3. **Warp Execution & Divergence (Warp 执行与阻塞)**:
   - 将线程按 Warp 分组（为了可视化，假设 1个 Warp = 4个线程）。
   - **Lockstep 执行**: 展示同一个 Warp 内的线程同时高亮执行同一行代码。
   - **Divergence (分支发散)**: 如果代码中有 `if-else`，请直观展示“活跃线程”执行 if 分支时，“非活跃线程”是如何处于 Masked/Blocked 状态等待的。

4. **Memory Coalescing (内存合并)**:
   - 当 Warp 发起 Global Memory 读取时，如果线程访问的地址是连续的，用一种颜色（如绿色框）将这些内存块框在一起，标注为“Coalesced Transaction”。
   - 如果访问是跨步或杂乱的，展示为多次独立的内存请求（红色或黄色），以此教育用户什么是 Uncoalesced access。

# Tech Stack & Implementation Details
- **Framework**: React (Functional Components + Hooks).
- **Styling**: Tailwind CSS (使用 Grid/Flex 布局).
- **Animation**: Framer Motion (用于平滑的数据移动效果).
- **Logic Simulation**:
  - 不要试图在浏览器运行 C++。请用 JavaScript 编写一个简单的状态机（State Machine）来模拟我提供的 Kernel 逻辑。
  - 提供一个 "Step Control"（上一步/下一步/播放）控制条，让我能逐行查看代码执行对应的硬件行为。
  - 在侧边栏显示当前执行的伪代码行，并高亮正在执行的那一行。

# Constraints
- 由于屏幕空间有限，请自动按比例缩小数据规模（例如数组大小设为 16 或 32，Block 数量设为 2-4 个）。
- 代码必须是完整的、可直接复制运行的 `tsx` 文件。

# My CUDA Code
```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 32 // 假设 BlockDim 为 32x32

__global__ void matrix_multiplication_shared_mem(const float *__restrict__ A, const float *__restrict__ B, float *C, int M, int N, int K)
{
    // 申请共享内存
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float acc = 0.0f;

    for (int i = 0; i < (N + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        // 读取数据
        if (row < M && TILE_WIDTH * i + tx < N)
        {
            As[ty][tx] = A[row * N + TILE_WIDTH * i + tx];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }
        if (col < K && TILE_WIDTH * i + ty < N)
        {
            Bs[ty][tx] = B[(TILE_WIDTH * i + ty) * K + col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads(); // 等待线程块里面的线程都搬运完成

        for (int j = 0; j < TILE_WIDTH; j++)
        {
            acc += As[ty][j] * Bs[j][tx]; //访问线程块里面的共享内存时，没有合并访问，只在乎bank config
        }

        __syncthreads();
    }
    if (row<M && col<K)
    {
        C[row*K+col] = acc;
    }
}
```