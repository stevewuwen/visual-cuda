## Role
你是一位世界级的 CUDA C++ 并行计算专家，同时也是一位精通 React、TypeScript 和数据可视化的前端架构师。我正在学习 CUDA，需要通过交互式可视化来理解 Kernel 代码的底层行为。

## Task
请阅读我提供的 CUDA Kernel 代码，编写一个**完整的、单文件的 React 组件 (`.tsx`)**。该组件需要是一个独立的“微型模拟器”，用于在浏览器中动态演示这段代码在 GPU 架构上的执行过程。

## Technical Stack
- **Framework**: React (Functional Components + Hooks).
- **Language**: TypeScript (必须定义清晰的 Interface).
- **Styling**: Tailwind CSS (用于布局和组件样式).
- **Animation**: Framer Motion (用于 `layoutId` 和平滑的数据移动).

## Architecture Requirement: "Trace-Based Simulation" (关键架构)
**不要**尝试在渲染循环中实时计算逻辑。请采用 **"预计算执行轨迹 (Pre-computed Execution Trace)"** 模式：
1.  **Step Generator**: 编写一个纯函数 `generateExecutionTrace()`。它接受输入数据（如数组），模拟 CUDA 线程逻辑，并返回一个 `Step[]` 数组。
2.  **Step Interface**: 每个 `Step` 对象应包含：
    - `activeLineId`: 当前高亮的代码行号。
    - `threads`: 一个对象数组，描述所有线程在当前步骤的状态（Global Memory 值, Shared Memory 值, Register 值, 活跃状态 `isActive`, 掩码状态 `isMasked`）。
    - `memoryHighlights`: 用于标记 Global Memory 读写操作（如 'coalesced', 'uncoalesced'）。
    - `description`: 一句文本，解释当前发生了什么（例如："Warp 0 is reading global memory", "Thread 1 is masked due to divergence"）。
3.  **Visualizer**: React 组件只需维护一个 `currentStepIndex` 状态，根据 `trace[currentStepIndex]` 渲染画面。

## Visualization Requirements

### 1. Layout
创建一个三栏布局：
- **左侧 (Code Panel)**: 显示 CUDA C++ 代码，高亮当前 `activeLineId`。
- **中间 (GPU View)**: 核心可视化区域。
    - **Global Memory**: 顶部展示输入和输出的数组。
    - **SM (Streaming Multiprocessor)**: 中间容器，包含 Block。
    - **Shared Memory**: SM 内部的一块独立区域。
    - **Threads (Warp View)**: 底部展示线程。请将线程按 **Warp (设为 4 个线程一组)** 分组展示。
- **底部 (Controls)**: 播放/暂停、上一步/下一步、重置按钮、以及当前的 `description` 文本。

### 2. Behavior Simulation
- **Memory Hierarchy**: 当发生 `g_idata[idx]` 读取时，使用 Framer Motion 的 `layoutId` 产生从 Global Memory 到 Register 的位移这个动画。
- **Memory Coalescing**:
    - 分析代码中的内存访问模式。
    - 如果 Warp 内线程访问连续地址，在 Global Memory 上画一个绿色的框包裹这些数据块，标注 "Coalesced"。
    - 如果访问不连续，用红色独立高亮，标注 "Uncoalesced"。

## Constraints & Assumptions
- **Scale**: 自动缩小规模。
- **Simplification**: 忽略复杂的 Grid 结构，只关注 **1 个 Block** 内部的执行。
- **Completeness**: 代码必须包含所有必要的 import、类型定义和样式，确保我可以复制到 react-app 中直接运行。

## My CUDA Code
```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// 宏定义块大小
// TS (Tile Size): 每个 Block 计算 64x64 的 C
// WPT (Work Per Thread): 每个线程计算 4x4 的 C
// TS_K: K 维度(你的代码里是 N 维度)的分块大小，设为 8 或 16
#define TS 64
#define WPT 4
#define TS_K 16 

// 优化后的 Kernel
__global__ void matrix_multiplication_optimized(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    // 每个 Block 处理 C 中 TS x TS (64x64) 的区域
    // 线程块维度: dim3(TS/WPT, TS/WPT) -> (16, 16) -> 256 个线程
    // 1. 声明共享内存
    // As: 存储 A 的切片 [TS][TS_K] -> [64][16]
    // Bs: 存储 B 的切片 [TS_K][TS] -> [16][64]
    __shared__ float As[TS][TS_K];
    __shared__ float Bs[TS_K][TS];

    // 2. 声明寄存器
    // accum: 累加器，每个线程负责计算 4x4 = 16 个元素
    float accum[WPT][WPT] = {0.0f};
    
    // reg_A, reg_B: 用于在内循环中缓存从 SMEM 读取的值
    float reg_A[WPT];
    float reg_B[WPT];

    // 线程 ID 和 Block ID
    int tx = threadIdx.x; // range 0-15
    int ty = threadIdx.y; // range 0-15
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // 当前线程负责的 C 矩阵起始坐标 (C 的分块左上角 + 线程偏移)
    // 每个线程覆盖 WPT(4) 个像素宽/高
    int row_c = by * TS + ty * WPT; 
    int col_c = bx * TS + tx * WPT;

    // 3. 循环遍历 N 维度 (步长 TS_K = 16)
    for (int t = 0; t < N; t += TS_K) {
        
        // --- 加载数据到 Shared Memory (协作加载) ---
        // 我们有 256 个线程。
        // 需要加载 A 的 Tile: 64行 * 16列 = 1024 元素。每个线程加载 1024/256 = 4 个元素。
        // 需要加载 B 的 Tile: 16行 * 64列 = 1024 元素。每个线程加载 4 个元素。
        
        // 加载 As (A 的子块): 
        // 这里的逻辑是将 256 个线程映射到 64x16 的区域
        // 我们使用 float4 向量化加载来极致优化带宽
        
        // 计算当前线程加载 As 的位置
        // 将 16x16 的线程块视为 256 个线性线程
        int tid = ty * (TS / WPT) + tx; // 0 ~ 255
        
        // 映射到 As[64][16]: 每一行 16 个元素，如果是 float4 就是 4 个 float4
        // 256 个线程，每个加载 1 个 float4 (4个float)，正好 1024 个 float
        // As 的行索引
        int load_a_row = tid / (TS_K / 4); 
        int load_a_col = (tid % (TS_K / 4)) * 4;
        
        // 从全局内存 A 加载到 As
        // 全局索引: A[(by * TS + load_a_row) * N + (t + load_a_col)]
        // 注意边界检查省略了，假设维度对其
        if (by * TS + load_a_row < M && t + load_a_col < N) {
             // 使用 float4 指针强转进行向量加载
             float4 tmp = reinterpret_cast<const float4*>(&A[(by * TS + load_a_row) * N + (t + load_a_col)])[0];
             As[load_a_row][load_a_col + 0] = tmp.x;
             As[load_a_row][load_a_col + 1] = tmp.y;
             As[load_a_row][load_a_col + 2] = tmp.z;
             As[load_a_row][load_a_col + 3] = tmp.w;
        }

        // 加载 Bs (B 的子块): [16][64]
        // 同样用 tid 映射。每行 64 个元素 = 16 个 float4。
        // 总共 16 行。总 float4 数 = 16 * 16 = 256。正好每个线程取 1 个 float4。
        int load_b_row = tid / (TS / 4);
        int load_b_col = (tid % (TS / 4)) * 4;

        if (t + load_b_row < N && bx * TS + load_b_col < K) {
             float4 tmp = reinterpret_cast<const float4*>(&B[(t + load_b_row) * K + (bx * TS + load_b_col)])[0];
             Bs[load_b_row][load_b_col + 0] = tmp.x;
             Bs[load_b_row][load_b_col + 1] = tmp.y;
             Bs[load_b_row][load_b_col + 2] = tmp.z;
             Bs[load_b_row][load_b_col + 3] = tmp.w;
        }

        __syncthreads(); // 等待数据加载完成

        // --- 在寄存器上进行计算 ---
        // 遍历 Shared Memory 中的 TS_K (16) 维度
        #pragma unroll
        for (int k = 0; k < TS_K; ++k) {
            
            // 1. 将所需的 As 和 Bs 数据预加载到寄存器
            // 每个线程计算 4x4，需要 As 的一列 4 个值，Bs 的一行 4 个值
            for (int i = 0; i < WPT; ++i) {
                reg_A[i] = As[ty * WPT + i][k];
                reg_B[i] = Bs[k][tx * WPT + i];
            }

            // 2. 外积计算 (Outer Product)
            // 计算 4x4 的结果，复用 reg_A 和 reg_B
            for (int row = 0; row < WPT; ++row) {
                for (int col = 0; col < WPT; ++col) {
                    accum[row][col] += reg_A[row] * reg_B[col];
                }
            }
        }
        
        __syncthreads(); // 等待计算完成，准备加载下一块
    }

    // 4. 写回结果到全局内存
    // 每个线程写回 4x4 个点
    for (int row = 0; row < WPT; ++row) {
        for (int col = 0; col < WPT; ++col) {
            int global_row = row_c + row;
            int global_col = col_c + col;
            
            if (global_row < M && global_col < K) {
                C[global_row * K + global_col] = accum[row][col];
            }
        }
    }
}
```