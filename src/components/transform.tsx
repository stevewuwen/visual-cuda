import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, SkipForward, SkipBack, RefreshCw, Cpu, Layers, Database, AlertTriangle, ArrowRight } from 'lucide-react';

// --- Constants & Types ---

const ROWS = 6;
const COLS = 6;
const BLOCK_DIM_X = 4;
const BLOCK_DIM_Y = 4;
const WARP_SIZE = 4; // Simulated warp size for visualization clarity

// Grid size needed to cover 6x6 with 4x4 blocks -> 2x2 blocks
const GRID_DIM_X = 2;
const GRID_DIM_Y = 2;

type ThreadState = 'idle' | 'calculating' | 'active' | 'masked' | 'reading' | 'writing';

interface Thread {
  id: number;
  localId: number; // Index within block
  blockId: { x: number; y: number };
  threadIdx: { x: number; y: number };
  globalRow: number;
  globalCol: number;
  val: number | null; // Register value
  state: ThreadState;
}

interface Step {
  line: number;
  description: string;
  activeWarpId: string | null; // e.g., "block0-warp0"
  memoryAccess?: {
    type: 'read' | 'write';
    indices: number[];
    isCoalesced: boolean;
  };
  threads: Record<string, Thread>; // Key: "bx-by-tx-ty"
}

// The Source Code
const CUDA_CODE = `__global__ void matrix_transpose(float* in, float* out) {
  // 1. Calculate Global Coordinates
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // 2. Boundary Check (Divergence source)
  if (row < rows && col < cols) {
      
      // 3. Global Memory Load (Coalesced?)
      float val = input[row * cols + col];

      // 4. Global Memory Store (Uncoalesced!)
      output[col * rows + row] = val;
  }
}`;

const LINES = [
  { num: 1, text: "int col = blockIdx.x * blockDim.x + threadIdx.x;" },
  { num: 2, text: "int row = blockIdx.y * blockDim.y + threadIdx.y;" },
  { num: 3, text: "if (row < rows && col < cols) {" },
  { num: 4, text: "    float val = input[row * cols + col];" },
  { num: 5, text: "    output[col * rows + row] = val;" },
  { num: 6, text: "}" },
];

// --- Simulation Logic ---

const generateSteps = (): Step[] => {
  const steps: Step[] = [];
  
  // Initialize all threads
  const initialThreads: Record<string, Thread> = {};
  
  for (let by = 0; by < GRID_DIM_Y; by++) {
    for (let bx = 0; bx < GRID_DIM_X; bx++) {
      for (let ty = 0; ty < BLOCK_DIM_Y; ty++) {
        for (let tx = 0; tx < BLOCK_DIM_X; tx++) {
          const key = `${bx}-${by}-${tx}-${ty}`;
          const col = bx * BLOCK_DIM_X + tx;
          const row = by * BLOCK_DIM_Y + ty;
          initialThreads[key] = {
            id: row * COLS + col, // Conceptual ID
            localId: ty * BLOCK_DIM_X + tx,
            blockId: { x: bx, y: by },
            threadIdx: { x: tx, y: ty },
            globalRow: row,
            globalCol: col,
            val: null,
            state: 'idle'
          };
        }
      }
    }
  }

  // Helper to deep copy threads
  const copyThreads = (current: Record<string, Thread>) => JSON.parse(JSON.stringify(current));

  // --- Step 0: Init ---
  steps.push({
    line: -1,
    description: "Kernel Launch: Grid(2,2), Block(4,4). Waiting to start.",
    activeWarpId: null,
    threads: copyThreads(initialThreads)
  });

  // Iterate through execution per Warp (Simulating SM scheduling)
  // For visualization, we will serialize Warp execution to show the "Step-by-step" nature clearer,
  // even though in reality warps run in parallel across SMs.
  
  const blocks = [
    { bx: 0, by: 0 }, { bx: 1, by: 0 }, 
    { bx: 0, by: 1 }, { bx: 1, by: 1 }
  ];

  blocks.forEach(block => {
    // Break block into Warps
    const threadsInBlock = [];
    for(let ty=0; ty<BLOCK_DIM_Y; ty++) {
        for(let tx=0; tx<BLOCK_DIM_X; tx++) {
            threadsInBlock.push({tx, ty});
        }
    }

    // Process each Warp (4 threads per warp)
    for (let w = 0; w < threadsInBlock.length; w += WARP_SIZE) {
        const warpThreads = threadsInBlock.slice(w, w + WARP_SIZE);
        const warpId = `b${block.bx}${block.by}-w${w/WARP_SIZE}`;
        
        let currentThreads = steps.length > 0 ? copyThreads(steps[steps.length-1].threads) : copyThreads(initialThreads);

        // Reset previous states to idle/done for visual clarity
        Object.keys(currentThreads).forEach(k => {
           if(currentThreads[k].state === 'reading' || currentThreads[k].state === 'writing') {
               currentThreads[k].state = 'idle'; // visually settle down
           }
        });

        // 1. Calculate Indices
        warpThreads.forEach(({tx, ty}) => {
            const key = `${block.bx}-${block.by}-${tx}-${ty}`;
            currentThreads[key].state = 'calculating';
        });

        steps.push({
            line: 1, 
            description: `[Block (${block.bx},${block.by}) Warp ${w/WARP_SIZE}] Computing Global Indices (row, col).`,
            activeWarpId: warpId,
            threads: copyThreads(currentThreads)
        });

        // 2. Check Bounds (Divergence)
        const activeIndices: number[] = [];
        const readIndices: number[] = [];
        const writeIndices: number[] = [];

        warpThreads.forEach(({tx, ty}) => {
            const key = `${block.bx}-${block.by}-${tx}-${ty}`;
            const t = currentThreads[key];
            if (t.globalRow < ROWS && t.globalCol < COLS) {
                t.state = 'active';
                activeIndices.push(1);
                readIndices.push(t.globalRow * COLS + t.globalCol);
                writeIndices.push(t.globalCol * ROWS + t.globalRow);
            } else {
                t.state = 'masked'; // Divergence!
            }
        });

        steps.push({
            line: 3, 
            description: `[Block (${block.bx},${block.by}) Warp ${w/WARP_SIZE}] Boundary Check. ${WARP_SIZE - activeIndices.length} threads masked (Divergence).`,
            activeWarpId: warpId,
            threads: copyThreads(currentThreads)
        });

        // If whole warp is masked, skip memory ops
        if (activeIndices.length > 0) {
            // 3. Load (Coalesced Check)
            warpThreads.forEach(({tx, ty}) => {
                const key = `${block.bx}-${block.by}-${tx}-${ty}`;
                if (currentThreads[key].state === 'active') {
                    currentThreads[key].state = 'reading';
                    currentThreads[key].val = (currentThreads[key].globalRow * COLS) + currentThreads[key].globalCol; // Dummy value = index
                }
            });

            // Check if readIndices are sequential
            let isCoalescedRead = true;
            for(let i=0; i<readIndices.length-1; i++) {
                if(readIndices[i+1] !== readIndices[i] + 1) isCoalescedRead = false;
            }

            steps.push({
                line: 4, 
                description: `[Block (${block.bx},${block.by}) Warp ${w/WARP_SIZE}] Reading Global Mem. Pattern: ${isCoalescedRead ? 'Sequential (Coalesced ✅)' : 'Strided (Uncoalesced ❌)'}`,
                activeWarpId: warpId,
                memoryAccess: {
                    type: 'read',
                    indices: readIndices,
                    isCoalesced: isCoalescedRead
                },
                threads: copyThreads(currentThreads)
            });

            // 4. Store (Uncoalesced Check)
            // Transition from reading to writing state
            warpThreads.forEach(({tx, ty}) => {
                const key = `${block.bx}-${block.by}-${tx}-${ty}`;
                if (currentThreads[key].state === 'reading') {
                    currentThreads[key].state = 'writing';
                }
            });
            
             // Check if writeIndices are sequential
             let isCoalescedWrite = true;
             // Sort to check strictly for sequential chunks, but for transpose it's strided
             // For matrix transpose, writes are usually stride = Row Width
             if (writeIndices.length > 1) {
                 const stride = writeIndices[1] - writeIndices[0];
                 if (stride !== 1) isCoalescedWrite = false;
             }

            steps.push({
                line: 5, 
                description: `[Block (${block.bx},${block.by}) Warp ${w/WARP_SIZE}] Writing Global Mem. Pattern: ${isCoalescedWrite ? 'Sequential (Coalesced ✅)' : 'Strided/Scattered (Uncoalesced ❌)'}`,
                activeWarpId: warpId,
                memoryAccess: {
                    type: 'write',
                    indices: writeIndices,
                    isCoalesced: isCoalescedWrite
                },
                threads: copyThreads(currentThreads)
            });
        }
    }
  });

  steps.push({
      line: 6,
      description: "Kernel execution complete.",
      activeWarpId: null,
      threads: copyThreads(steps[steps.length-1].threads)
  });

  return steps;
};


// --- Components ---

const MemoryGrid = ({ type, data, highlightIndices, isCoalesced, rows, cols }: any) => {
  return (
    <div className="bg-slate-900 p-4 rounded-xl border border-slate-700 shadow-xl">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
           <Database size={16} /> {type === 'read' ? 'Global Memory (Input)' : 'Global Memory (Output)'}
        </h3>
        {highlightIndices.length > 0 && (
          <span className={`text-xs px-2 py-1 rounded font-mono ${isCoalesced ? 'bg-green-900 text-green-300 border border-green-700' : 'bg-red-900 text-red-300 border border-red-700'}`}>
            {isCoalesced ? 'COALESCED TRAMSACTION' : 'UNCOALESCED TRANSACTION'}
          </span>
        )}
      </div>
      
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {Array.from({ length: rows * cols }).map((_, idx) => {
          const isHighlighted = highlightIndices.includes(idx);
          
          // Determine color based on highlight
          let bgClass = "bg-slate-800";
          if (isHighlighted) {
            bgClass = isCoalesced ? "bg-green-500 shadow-[0_0_15px_rgba(34,197,94,0.6)]" : "bg-red-500 shadow-[0_0_15px_rgba(239,68,68,0.6)]";
          } else if (idx >= 36) { // Visual filler for 8x8 grid bounds logic but 6x6 memory
             bgClass = "bg-black opacity-30";
          }

          return (
            <motion.div
              key={idx}
              initial={false}
              animate={{
                scale: isHighlighted ? 1.1 : 1,
                opacity: (idx >= rows * cols) ? 0.2 : 1 // Dim out of bounds if any
              }}
              className={`aspect-square rounded-sm flex items-center justify-center text-[10px] text-slate-400 font-mono relative ${bgClass} border border-slate-700/50`}
            >
              {idx}
            </motion.div>
          );
        })}
      </div>
      {/* Memory Bus Visual */}
      <div className="mt-2 h-1 w-full bg-slate-700 rounded overflow-hidden">
          {highlightIndices.length > 0 && (
              <motion.div 
                layoutId="bus-signal"
                className={`h-full w-full ${isCoalesced ? 'bg-green-500' : 'bg-red-500'}`}
                initial={{ x: '-100%' }}
                animate={{ x: '100%' }}
                transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
              />
          )}
      </div>
    </div>
  );
};

const Thread = ({ thread, isActiveWarp }: { thread: Thread, isActiveWarp: boolean }) => {
  const getStatusColor = () => {
    if (thread.state === 'masked') return 'bg-slate-700 opacity-50 border-slate-600';
    if (thread.state === 'reading') return 'bg-blue-600 border-blue-400 text-white shadow-lg shadow-blue-500/50';
    if (thread.state === 'writing') return 'bg-orange-600 border-orange-400 text-white shadow-lg shadow-orange-500/50';
    if (thread.state === 'active') return 'bg-green-600 border-green-400 text-white';
    if (thread.state === 'calculating') return 'bg-yellow-600 border-yellow-400 text-white';
    return 'bg-slate-800 border-slate-700 text-slate-500';
  };

  return (
    <div className={`relative w-8 h-10 rounded text-[9px] flex flex-col items-center justify-center border transition-all duration-200 ${getStatusColor()} ${isActiveWarp && thread.state !== 'masked' && thread.state !== 'idle' ? 'ring-2 ring-white scale-105 z-10' : ''}`}>
       <div className="absolute -top-2 text-[8px] bg-slate-900 px-1 rounded text-slate-400">T{thread.threadIdx.y},{thread.threadIdx.x}</div>
       <div className="font-mono mt-1">
         {thread.state === 'masked' ? 'MASK' : (thread.val !== null ? `R:${thread.val}` : 'R:--')}
       </div>
       
       {/* Visualize data arriving in register */}
       {thread.state === 'reading' && (
           <motion.div 
             initial={{ y: -20, opacity: 0 }}
             animate={{ y: 0, opacity: 1 }}
             className="absolute w-full h-full border-2 border-blue-300 rounded animate-ping"
           />
       )}
    </div>
  );
};

const Warp = ({ threads, isActiveWarp }: { threads: Thread[], isActiveWarp: boolean }) => {
  return (
    <div className={`flex gap-1 p-1 rounded ${isActiveWarp ? 'bg-slate-700/50 ring-1 ring-slate-500' : ''}`}>
       {threads.map((t) => (
         <Thread key={t.localId} thread={t} isActiveWarp={isActiveWarp} />
       ))}
    </div>
  );
};

const SM = ({ id, threads, activeWarpId }: { id: number, threads: Record<string, Thread>, activeWarpId: string | null }) => {
  // Group threads by Block and Warp for display
  // We filter threads that belong to blocks assigned to this SM
  // For sim: SM0 gets Block (0,0) and (1,0). SM1 gets Block (0,1) and (1,1).
  
  const assignedBlocks = id === 0 ? [{x:0,y:0}, {x:1,y:0}] : [{x:0,y:1}, {x:1,y:1}];

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 flex flex-col gap-4 min-w-[300px]">
      <div className="flex items-center gap-2 border-b border-slate-700 pb-2">
        <Cpu className="text-indigo-400" size={20} />
        <h3 className="font-bold text-slate-200">SM {id} (Streaming Multiprocessor)</h3>
      </div>

      <div className="space-y-4">
        {assignedBlocks.map((blk) => (
            <div key={`${blk.x}-${blk.y}`} className="bg-slate-950/50 p-2 rounded border border-slate-800">
                <div className="text-xs text-indigo-300 font-mono mb-2 flex justify-between">
                    <span>Block ({blk.x}, {blk.y})</span>
                    <span className="text-slate-600">Shared Mem: 0B</span>
                </div>
                <div className="flex flex-wrap gap-2 justify-center">
                    {/* Render Warps. We have 4 warps per block (16 threads / 4 warp size) */}
                    {Array.from({length: (BLOCK_DIM_X*BLOCK_DIM_Y)/WARP_SIZE}).map((_, wIdx) => {
                         const warpId = `b${blk.x}${blk.y}-w${wIdx}`;
                         const isActive = activeWarpId === warpId;
                         
                         // Gather threads for this warp
                         const warpThreads = [];
                         for(let i=0; i<WARP_SIZE; i++) {
                             const flatIdx = wIdx * WARP_SIZE + i;
                             const tx = flatIdx % BLOCK_DIM_X;
                             const ty = Math.floor(flatIdx / BLOCK_DIM_X);
                             const key = `${blk.x}-${blk.y}-${tx}-${ty}`;
                             if(threads[key]) warpThreads.push(threads[key]);
                         }

                         return (
                             <Warp key={warpId} threads={warpThreads} isActiveWarp={isActive} />
                         );
                    })}
                </div>
            </div>
        ))}
      </div>
    </div>
  );
};

const CodePanel = ({ step }: { step: Step }) => {
  return (
    <div className="bg-[#1e1e1e] rounded-xl overflow-hidden shadow-2xl border border-slate-700 flex flex-col h-full font-mono text-xs md:text-sm">
      <div className="bg-[#252526] px-4 py-2 border-b border-slate-700 flex items-center gap-2">
        <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500"/>
            <div className="w-3 h-3 rounded-full bg-yellow-500"/>
            <div className="w-3 h-3 rounded-full bg-green-500"/>
        </div>
        <span className="text-slate-400 ml-2">kernel.cu</span>
      </div>
      <div className="p-4 overflow-auto flex-1 relative">
        {LINES.map((line) => {
          const isActive = step.line === line.num;
          return (
            <div key={line.num} className={`relative pl-4 pr-2 py-1 ${isActive ? 'bg-indigo-900/30' : ''}`}>
              {isActive && (
                <motion.div 
                  layoutId="activeLine"
                  className="absolute left-0 top-0 w-1 h-full bg-indigo-500"
                />
              )}
              <span className="text-slate-600 select-none mr-4 w-4 inline-block text-right">{line.num}</span>
              <span className={isActive ? 'text-indigo-200 font-bold' : 'text-slate-300'}>
                {line.text}
              </span>
            </div>
          );
        })}
      </div>
      {/* Legend */}
      <div className="bg-[#252526] p-3 text-[10px] grid grid-cols-2 gap-2 text-slate-400 border-t border-slate-700">
         <div className="flex items-center gap-1"><div className="w-2 h-2 bg-green-600 rounded"></div> Active</div>
         <div className="flex items-center gap-1"><div className="w-2 h-2 bg-slate-700 border border-slate-600 rounded"></div> Masked (Divergence)</div>
         <div className="flex items-center gap-1"><div className="w-2 h-2 bg-blue-600 rounded"></div> Reading (Global Mem)</div>
         <div className="flex items-center gap-1"><div className="w-2 h-2 bg-orange-600 rounded"></div> Writing (Global Mem)</div>
      </div>
    </div>
  );
};

// --- Main App Component ---

const CudaVisualizer = () => {
  const steps = useMemo(() => generateSteps(), []);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const currentStep = steps[currentStepIdx];

  // Auto-play logic
  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentStepIdx((prev) => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 800); // 800ms per step
    }
    return () => clearInterval(interval);
  }, [isPlaying, steps.length]);

  const handleNext = () => setCurrentStepIdx(prev => Math.min(prev + 1, steps.length - 1));
  const handlePrev = () => setCurrentStepIdx(prev => Math.max(prev - 1, 0));
  const handleReset = () => {
      setIsPlaying(false);
      setCurrentStepIdx(0);
  }

  // Derived state for visualization
  const memHighlight = currentStep.memoryAccess ? currentStep.memoryAccess.indices : [];
  const isCoalesced = currentStep.memoryAccess ? currentStep.memoryAccess.isCoalesced : true;
  const memType = currentStep.memoryAccess ? currentStep.memoryAccess.type : 'read';

  return (
    <div className="min-h-screen bg-[#0f1117] text-slate-200 font-sans p-4 md:p-6 flex flex-col gap-6">
      
      {/* Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-800 pb-4">
        <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-indigo-500 bg-clip-text text-transparent">
            CUDA Kernel Visualization
            </h1>
            <p className="text-slate-500 text-sm mt-1">Simulating Naive Matrix Transpose (6x6 Matrix on 2 SMs)</p>
        </div>

        <div className="flex items-center gap-2 bg-slate-900 p-2 rounded-lg border border-slate-800">
            <button onClick={handleReset} className="p-2 hover:bg-slate-800 rounded text-slate-400 hover:text-white" title="Reset"><RefreshCw size={18}/></button>
            <button onClick={handlePrev} className="p-2 hover:bg-slate-800 rounded text-slate-400 hover:text-white" disabled={currentStepIdx===0}><SkipBack size={18}/></button>
            <button 
                onClick={() => setIsPlaying(!isPlaying)} 
                className={`flex items-center gap-2 px-4 py-2 rounded font-bold transition-colors ${isPlaying ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20' : 'bg-green-500/10 text-green-400 hover:bg-green-500/20'}`}
            >
                {isPlaying ? <><Pause size={18}/> Pause</> : <><Play size={18}/> Play</>}
            </button>
            <button onClick={handleNext} className="p-2 hover:bg-slate-800 rounded text-slate-400 hover:text-white" disabled={currentStepIdx===steps.length-1}><SkipForward size={18}/></button>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        
        {/* Left Column: Code & Status */}
        <div className="lg:col-span-4 flex flex-col gap-4">
            <div className="h-[400px]">
                <CodePanel step={currentStep} />
            </div>
            
            <div className="bg-slate-900 border border-slate-700 p-4 rounded-xl flex-1">
                <h3 className="text-indigo-400 font-bold mb-2 flex items-center gap-2">
                    <AlertTriangle size={16}/> Execution State
                </h3>
                <div className="text-sm font-mono space-y-2">
                    <p className="text-slate-400">Step: <span className="text-white">{currentStepIdx} / {steps.length - 1}</span></p>
                    <p className="text-slate-300 border-l-2 border-indigo-500 pl-3 py-1 bg-slate-800/50 rounded-r">
                        {currentStep.description}
                    </p>
                    {currentStep.memoryAccess && (
                         <div className={`mt-4 p-3 rounded border ${currentStep.memoryAccess.isCoalesced ? 'bg-green-900/20 border-green-800' : 'bg-red-900/20 border-red-800'}`}>
                             <p className={`font-bold ${currentStep.memoryAccess.isCoalesced ? 'text-green-400' : 'text-red-400'}`}>
                                {currentStep.memoryAccess.isCoalesced ? 'Coalesced Access' : 'Uncoalesced Access'}
                             </p>
                             <p className="text-xs text-slate-400 mt-1">
                                {currentStep.memoryAccess.isCoalesced 
                                    ? "Threads in Warp accessing contiguous memory addresses. One transaction serves all." 
                                    : "Threads accessing strided addresses. Requires multiple memory transactions."}
                             </p>
                         </div>
                    )}
                </div>
            </div>
        </div>

        {/* Right Column: Hardware Sim */}
        <div className="lg:col-span-8 flex flex-col gap-6">
            
            {/* Global Memory Section */}
            <div className="grid grid-cols-2 gap-6">
                <MemoryGrid 
                    type="read" 
                    rows={ROWS} 
                    cols={COLS} 
                    data={[]} 
                    highlightIndices={memType === 'read' ? memHighlight : []}
                    isCoalesced={isCoalesced}
                />
                <MemoryGrid 
                    type="write" 
                    rows={ROWS} 
                    cols={COLS} 
                    data={[]} 
                    highlightIndices={memType === 'write' ? memHighlight : []}
                    isCoalesced={isCoalesced}
                />
            </div>

            {/* Memory Bus Arrow */}
            <div className="flex justify-center items-center gap-4 text-slate-600">
                <ArrowRight size={24} className={memType === 'read' && memHighlight.length ? 'text-blue-500 animate-pulse' : ''} />
                <span className="text-xs uppercase tracking-widest font-bold">Bus</span>
                <ArrowRight size={24} className={memType === 'write' && memHighlight.length ? 'text-orange-500 animate-pulse' : ''} />
            </div>

            {/* SMs Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <SM id={0} threads={currentStep.threads} activeWarpId={currentStep.activeWarpId} />
                <SM id={1} threads={currentStep.threads} activeWarpId={currentStep.activeWarpId} />
            </div>
        </div>

      </div>
    </div>
  );
};

export default CudaVisualizer;