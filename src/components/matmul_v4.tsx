import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, StepForward, StepBack, RotateCcw, Cpu, Layers, Grid3X3, ArrowRight } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- Utils ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Constants & Configuration ---
// SCALED DOWN PARAMETERS for Visualization
// Real World: TS=64, WPT=4, TS_K=16, Threads=256
// Simulation: TS=4,  WPT=2, TS_K=4,  Threads=4 (2x2 Block)
const TS = 4;
const TS_K = 4;
const WPT = 2;
const M = 4; // Matrix Height
const N = 4; // Matrix Shared Dimension
const K = 4; // Matrix Width

// Source Code for Display
const CUDA_CODE = `__global__ void matrix_mul_opt(...) {
  // 1. Setup Shared Mem & Regs
  __shared__ float As[TS][TS_K]; __shared__ float Bs[TS_K][TS];
  float accum[WPT][WPT] = {0}; float reg_A[WPT], reg_B[WPT];
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row_c = by * TS + ty * WPT; int col_c = bx * TS + tx * WPT;
  // 2. Loop over Tiles
  for (int t = 0; t < N; t += TS_K) {
    // 3. Load A to Shared (Vectorized)
    int tid = ty * (TS/WPT) + tx;
    int load_a_row = tid; 
    // float4 load simulated here
    As[load_a_row][0..3] = A_glob[...];
    // 4. Load B to Shared (Vectorized)
    int load_b_row = tid;
    Bs[load_b_row][0..3] = B_glob[...];
    __syncthreads();
    // 5. Compute (Register Blocking)
    #pragma unroll
    for (int k = 0; k < TS_K; ++k) {
      for (int i=0; i<WPT; ++i) {
         reg_A[i] = As[ty*WPT + i][k];
         reg_B[i] = Bs[k][tx*WPT + i];
      }
      for (int r=0; r<WPT; ++r) {
        for (int c=0; c<WPT; ++c) {
           accum[r][c] += reg_A[r] * reg_B[c];
        }
      }
    }
    __syncthreads();
  }
  // 6. Write to Global Memory
  C[row_c..][col_c..] = accum[..][..];
}`;

// Line Mapping for Highlighting
const LINE_MAP = {
  SETUP: 2,
  LOOP_T: 13,
  LOAD_A: 18, // Simplified mapping
  LOAD_B: 23,
  SYNC_1: 25,
  COMPUTE_LOAD_REG: 29,
  COMPUTE_MATH: 34,
  SYNC_2: 39,
  STORE_C: 43
};

// --- Types ---

interface ThreadState {
  id: number;
  tx: number;
  ty: number;
  regs: {
    accum: number[][]; // WPT x WPT
    reg_A: number[];   // WPT
    reg_B: number[];   // WPT
  };
  isActive: boolean;
  status: string;
}

interface MemoryHighlight {
  type: 'read' | 'write';
  target: 'global_A' | 'global_B' | 'global_C' | 'shared_As' | 'shared_Bs';
  indices: { r: number, c: number }[]; // Coordinates highlighted
  coalesced?: boolean;
}

interface Step {
  id: number;
  activeLineId: number;
  description: string;
  globalA: number[][];
  globalB: number[][];
  globalC: number[][];
  sharedAs: number[][];
  sharedBs: number[][];
  threads: ThreadState[];
  highlights: MemoryHighlight[];
}

// --- Logic Generator ---

const generateExecutionTrace = (): Step[] => {
  const steps: Step[] = [];
  let stepCount = 0;

  // Initialize Data
  // Matrix A (4x4): Identity-ish with values
  const matA = Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) => (r === c ? 1 : 0) + r * 0.1));
  // Matrix B (4x4): All 1s and 2s
  const matB = Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) => c + 1));
  // Matrix C (4x4): Empty
  const matC = Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, () => 0));

  // Shared Mem
  const As = Array.from({ length: 4 }, () => Array(4).fill(0));
  const Bs = Array.from({ length: 4 }, () => Array(4).fill(0));

  // Threads (2x2 Block -> 4 Threads)
  // Grid Dim: TS/WPT = 4/2 = 2. So Block is 2x2.
  const threads: ThreadState[] = [];
  for (let ty = 0; ty < 2; ty++) {
    for (let tx = 0; tx < 2; tx++) {
      threads.push({
        id: ty * 2 + tx,
        tx, ty,
        isActive: true,
        status: 'Idle',
        regs: {
          accum: [[0, 0], [0, 0]],
          reg_A: [0, 0],
          reg_B: [0, 0]
        }
      });
    }
  }

  const pushStep = (line: number, desc: string, highlights: MemoryHighlight[] = []) => {
    steps.push({
      id: stepCount++,
      activeLineId: line,
      description: desc,
      globalA: JSON.parse(JSON.stringify(matA)),
      globalB: JSON.parse(JSON.stringify(matB)),
      globalC: JSON.parse(JSON.stringify(matC)),
      sharedAs: JSON.parse(JSON.stringify(As)),
      sharedBs: JSON.parse(JSON.stringify(Bs)),
      threads: JSON.parse(JSON.stringify(threads)),
      highlights
    });
  };

  // --- Step 1: Setup ---
  pushStep(LINE_MAP.SETUP, "Initializing Block (2x2 threads). Each thread handles a 2x2 sub-block of C.", []);

  // --- Step 2: Loop T (Only 1 tile in this scaled version) ---
  pushStep(LINE_MAP.LOOP_T, "Looping over K-dimension (Tile 0).", []);

  // --- Step 3: Load A -> Shared ---
  // Simulation: Threads map linearly to rows of As.
  // Thread 0 -> As[0], Thread 1 -> As[1], etc.
  const highlightsA: MemoryHighlight[] = [];
  const writesAs: MemoryHighlight[] = [];
  
  threads.forEach(t => {
    const tid = t.id; // 0..3
    const row = tid; // Each thread loads 1 row (simulating float4 loading 4 floats)
    // Read Global
    const indices = [0, 1, 2, 3].map(c => ({ r: row, c }));
    highlightsA.push({ type: 'read', target: 'global_A', indices, coalesced: true });
    // Write Shared
    writesAs.push({ type: 'write', target: 'shared_As', indices });
    
    // Logic Update
    for(let c=0; c<4; c++) As[row][c] = matA[row][c];
    t.status = `Loading As row ${row}`;
  });

  pushStep(LINE_MAP.LOAD_A, 
    "Cooperative Load A: Each thread loads one 'float4' (a full row) into Shared Memory As. Access is Coalesced.", 
    [...highlightsA, ...writesAs]
  );

  // --- Step 4: Load B -> Shared ---
  const highlightsB: MemoryHighlight[] = [];
  const writesBs: MemoryHighlight[] = [];
  threads.forEach(t => {
    const tid = t.id;
    const row = tid; 
    const indices = [0, 1, 2, 3].map(c => ({ r: row, c }));
    highlightsB.push({ type: 'read', target: 'global_B', indices, coalesced: true });
    writesBs.push({ type: 'write', target: 'shared_Bs', indices });

    for(let c=0; c<4; c++) Bs[row][c] = matB[row][c];
    t.status = `Loading Bs row ${row}`;
  });

  pushStep(LINE_MAP.LOAD_B, 
    "Cooperative Load B: Similarly, threads load B tile into Shared Memory Bs.", 
    [...highlightsB, ...writesBs]
  );

  // --- Step 5: Sync ---
  threads.forEach(t => t.status = "Waiting");
  pushStep(LINE_MAP.SYNC_1, "__syncthreads(): Ensuring all data is loaded in Shared Memory before computing.", []);

  // --- Step 6: Compute Loop ---
  // Iterate k (0..TS_K-1)
  for (let k = 0; k < TS_K; k++) {
    // 6a: Prefetch to Registers
    const readAs: MemoryHighlight[] = [];
    const readBs: MemoryHighlight[] = [];
    
    threads.forEach(t => {
       // Each thread needs As[ty*WPT + i][k] and Bs[k][tx*WPT + i]
       // In scaled logic: ty (0..1), WPT=2. 
       // Thread(0,0) needs As[0][k], As[1][k] and Bs[k][0], Bs[k][1]
       t.status = `Comp k=${k}`;
       
       const asIndices = [];
       const bsIndices = [];
       
       for(let i=0; i<WPT; i++) {
         const rowA = t.ty * WPT + i;
         const colB = t.tx * WPT + i;
         t.regs.reg_A[i] = As[rowA][k];
         t.regs.reg_B[i] = Bs[k][colB];
         asIndices.push({r: rowA, c: k});
         bsIndices.push({r: k, c: colB});
       }
       readAs.push({ type: 'read', target: 'shared_As', indices: asIndices });
       readBs.push({ type: 'read', target: 'shared_Bs', indices: bsIndices });
    });

    pushStep(LINE_MAP.COMPUTE_LOAD_REG, 
      `Inner Loop k=${k}: Loading slices from Shared Memory into Registers (reg_A, reg_B).`, 
      [...readAs, ...readBs]
    );

    // 6b: Math (Outer Product)
    threads.forEach(t => {
      for(let r=0; r<WPT; r++) {
        for(let c=0; c<WPT; c++) {
          t.regs.accum[r][c] += t.regs.reg_A[r] * t.regs.reg_B[c];
        }
      }
      t.status = `FFMA`;
    });
    pushStep(LINE_MAP.COMPUTE_MATH, 
      `Outer Product: Accumulating results in Registers (WPT x WPT tile per thread).`, 
      []
    );
  }

  pushStep(LINE_MAP.SYNC_2, "__syncthreads(): Preparing for next tile (or finish).", []);

  // --- Step 7: Store C ---
  const writeC: MemoryHighlight[] = [];
  threads.forEach(t => {
     t.status = "Storing";
     const indices = [];
     for(let r=0; r<WPT; r++) {
       for(let c=0; c<WPT; c++) {
         const gr = t.ty * WPT + r;
         const gc = t.tx * WPT + c;
         matC[gr][gc] = t.regs.accum[r][c];
         indices.push({r: gr, c: gc});
       }
     }
     writeC.push({type: 'write', target: 'global_C', indices});
  });

  pushStep(LINE_MAP.STORE_C, 
    "Write Back: Storing computed register values to Global Memory C.", 
    writeC
  );

  return steps;
};

// --- Components ---

const MatrixGrid = ({ data, title, highlights, name }: { data: number[][], title: string, highlights: MemoryHighlight[], name: string }) => {
  return (
    <div className="flex flex-col items-center">
      <h3 className="text-xs font-bold text-gray-400 mb-1">{title}</h3>
      <div className="grid grid-cols-4 gap-1 p-1 bg-slate-800 rounded border border-slate-700">
        {data.map((row, r) => 
          row.map((val, c) => {
            // Check highlights
            const hl = highlights.find(h => 
              (h.target === name) && h.indices.some(idx => idx.r === r && idx.c === c)
            );
            
            let bgClass = "bg-slate-900";
            let borderClass = "border-transparent";
            
            if (hl) {
              if (hl.type === 'read') {
                 bgClass = hl.coalesced ? "bg-green-900/50" : "bg-blue-900/50";
                 borderClass = hl.coalesced ? "border-green-500" : "border-blue-400";
              } else {
                 bgClass = "bg-purple-900/50";
                 borderClass = "border-purple-400";
              }
            }

            return (
              <motion.div 
                key={`${r}-${c}`}
                layoutId={`${name}-${r}-${c}`}
                className={cn(
                  "w-8 h-8 flex items-center justify-center text-[10px] font-mono rounded border transition-colors duration-300",
                  bgClass, borderClass
                )}
              >
                {val.toFixed(0)}
              </motion.div>
            );
          })
        )}
      </div>
    </div>
  );
};

const ThreadView = ({ thread, isActive }: { thread: ThreadState, isActive: boolean }) => {
  return (
    <div className={cn(
      "p-2 rounded border text-xs font-mono transition-all duration-300 w-full",
      isActive ? "bg-orange-900/20 border-orange-500/50" : "bg-slate-800 border-slate-700 opacity-50"
    )}>
      <div className="flex justify-between mb-1 text-orange-400 font-bold">
        <span>T{thread.id} ({thread.tx},{thread.ty})</span>
        <span>{thread.status}</span>
      </div>
      
      {/* Registers */}
      <div className="grid grid-cols-2 gap-x-2 gap-y-1">
        <div className="col-span-2 flex gap-1 items-center">
           <span className="text-gray-500">RegA:</span>
           <div className="flex gap-0.5">
             {thread.regs.reg_A.map((v,i) => <div key={i} className="bg-slate-900 px-1 rounded">{v.toFixed(0)}</div>)}
           </div>
        </div>
        <div className="col-span-2 flex gap-1 items-center">
           <span className="text-gray-500">RegB:</span>
           <div className="flex gap-0.5">
             {thread.regs.reg_B.map((v,i) => <div key={i} className="bg-slate-900 px-1 rounded">{v.toFixed(0)}</div>)}
           </div>
        </div>
        
        <div className="col-span-2 mt-1">
          <span className="text-gray-500 block mb-0.5">Accum (Result):</span>
          <div className="grid grid-cols-2 gap-0.5 w-max">
             {thread.regs.accum.map((row, i) => 
               row.map((v, j) => (
                 <div key={`${i}-${j}`} className="w-6 h-6 bg-orange-500/20 flex items-center justify-center rounded text-orange-200">
                   {v.toFixed(0)}
                 </div>
               ))
             )}
          </div>
        </div>
      </div>
    </div>
  );
};

const CodePanel = ({ activeLine }: { activeLine: number }) => {
  const lines = CUDA_CODE.split('\n');
  return (
    <div className="h-full overflow-hidden bg-slate-950 text-slate-300 font-mono text-xs p-4 rounded-lg shadow-inner border border-slate-800">
      <h3 className="text-sm font-bold text-blue-400 mb-2 flex items-center gap-2">
        <Cpu size={14} /> matrix_multiplication.cu
      </h3>
      <div className="overflow-y-auto h-[500px] custom-scrollbar">
        {lines.map((line, idx) => {
           const lineNum = idx + 1;
           const isActive = activeLine === lineNum;
           return (
             <div key={idx} id={`line-${lineNum}`} className={cn(
               "flex gap-3 px-2 py-0.5 rounded transition-colors",
               isActive ? "bg-blue-900/30 text-white font-bold" : "opacity-60"
             )}>
               <span className="w-6 text-right text-slate-600 select-none">{lineNum}</span>
               <pre>{line}</pre>
             </div>
           );
        })}
      </div>
    </div>
  );
};

// --- Main Visualizer Component ---

export default function CudaOptimizedKernelVisualizer() {
  const [trace, setTrace] = useState<Step[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    setTrace(generateExecutionTrace());
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && currentStep < trace.length - 1) {
      interval = setInterval(() => setCurrentStep(c => c + 1), 1500);
    } else if (currentStep >= trace.length - 1) {
      setIsPlaying(false);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentStep, trace.length]);

  if (trace.length === 0) return <div className="text-white">Generating simulation...</div>;

  const step = trace[currentStep];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 flex flex-col gap-6 font-sans">
      {/* Header */}
      <header className="flex justify-between items-end border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
            CUDA Kernel Visualization: Tiled Matrix Mul
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Visualizing Cooperative Loading, Shared Memory, and Register Blocking.
          </p>
        </div>
        <div className="text-right text-xs text-slate-500">
          <div className="flex gap-4">
             <span className="flex items-center gap-1"><span className="w-2 h-2 bg-blue-500 rounded-full"></span> Global Read</span>
             <span className="flex items-center gap-1"><span className="w-2 h-2 bg-green-500 rounded-full"></span> Coalesced</span>
             <span className="flex items-center gap-1"><span className="w-2 h-2 bg-purple-500 rounded-full"></span> Write</span>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left: Code */}
        <div className="lg:col-span-4 h-full">
           <CodePanel activeLine={step.activeLineId} />
        </div>

        {/* Center: GPU View */}
        <div className="lg:col-span-8 flex flex-col gap-4">
          
          {/* Top: Global Memory */}
          <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800 relative">
             <div className="absolute top-2 left-2 text-xs font-bold text-slate-500 uppercase tracking-wider">Global Memory (DRAM)</div>
             <div className="flex justify-center gap-12 mt-4">
                <MatrixGrid name="global_A" title="Matrix A (MxN)" data={step.globalA} highlights={step.highlights} />
                <MatrixGrid name="global_B" title="Matrix B (NxK)" data={step.globalB} highlights={step.highlights} />
                <MatrixGrid name="global_C" title="Matrix C (MxK)" data={step.globalC} highlights={step.highlights} />
             </div>
          </div>

          {/* Arrow */}
          <div className="flex justify-center -my-2 opacity-30 animate-pulse">
            <ArrowRight className="rotate-90" />
          </div>

          {/* Middle: SM Container */}
          <div className="flex-1 bg-slate-900 p-4 rounded-xl border-2 border-slate-700 shadow-xl relative overflow-hidden">
             <div className="absolute top-0 right-0 p-2 bg-slate-800 rounded-bl text-xs font-bold text-green-400 border-b border-l border-slate-700">
               Streaming Multiprocessor (SM)
             </div>

             <div className="flex flex-col h-full gap-6">
               
               {/* Shared Memory */}
               <div className="flex justify-center gap-8 items-start p-4 bg-slate-950/50 rounded-lg border border-slate-800 border-dashed">
                 <div className="text-xs text-green-500 font-bold absolute mt-[-24px] bg-slate-900 px-2">Shared Memory (L1)</div>
                 <MatrixGrid name="shared_As" title="Tile As (4x4)" data={step.sharedAs} highlights={step.highlights} />
                 <MatrixGrid name="shared_Bs" title="Tile Bs (4x4)" data={step.sharedBs} highlights={step.highlights} />
               </div>

               {/* Threads (Warp) */}
               <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-auto">
                 {step.threads.map(t => (
                   <ThreadView key={t.id} thread={t} isActive={t.isActive} />
                 ))}
               </div>
             </div>
          </div>
        </div>

      </div>

      {/* Footer: Controls */}
      <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 flex items-center gap-6 sticky bottom-0 z-10 shadow-2xl">
        <div className="flex gap-2">
           <button onClick={() => setCurrentStep(0)} className="p-2 hover:bg-slate-800 rounded text-slate-400"><RotateCcw size={20}/></button>
           <button onClick={() => setCurrentStep(Math.max(0, currentStep - 1))} className="p-2 hover:bg-slate-800 rounded text-slate-400"><StepBack size={20}/></button>
           <button 
             onClick={() => setIsPlaying(!isPlaying)} 
             className="p-2 bg-blue-600 hover:bg-blue-500 rounded text-white shadow-lg shadow-blue-900/50 transition-all"
           >
             {isPlaying ? <Pause size={20} fill="currentColor"/> : <Play size={20} fill="currentColor"/>}
           </button>
           <button onClick={() => setCurrentStep(Math.min(trace.length-1, currentStep + 1))} className="p-2 hover:bg-slate-800 rounded text-slate-400"><StepForward size={20}/></button>
        </div>
        
        <div className="h-8 w-[1px] bg-slate-700 mx-2"></div>

        <div className="flex-1">
          <div className="flex justify-between items-center mb-1">
             <span className="text-xs font-bold text-slate-500">STEP {currentStep + 1} / {trace.length}</span>
             <div className="w-32 h-1 bg-slate-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-500 transition-all duration-300" 
                  style={{ width: `${((currentStep + 1) / trace.length) * 100}%`}}
                ></div>
             </div>
          </div>
          <AnimatePresence mode='wait'>
            <motion.p 
              key={currentStep}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="text-lg font-medium text-blue-100"
            >
              {step.description}
            </motion.p>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

export const metadata = {
  name: '矩阵乘法（第四优化版）',
  description: '',
  image: '',
};