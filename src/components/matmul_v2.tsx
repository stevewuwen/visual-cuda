import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, ChevronRight, ChevronLeft, RotateCcw, Cpu, Layers, Database, Code, ArrowRight } from 'lucide-react';

// --- Types & Constants ---

const TILE_WIDTH = 2;
const MATRIX_SIZE = 4; // 4x4 Matrices
// Visual simplification: We assume BlockDim = 2x2 to match Tile Width
// We focus on Block(0,0) which calculates the top-left 2x2 of C.

// Simplified Data for 4x4 Matrices
// Matrix A: Row-based pattern to make it easy to track
const MATRIX_A = [
  1, 1, 2, 2,
  1, 1, 2, 2,
  3, 3, 4, 4,
  3, 3, 4, 4
];

// Matrix B: Identity-like to make math obvious
const MATRIX_B = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
];

// Initial C
const MATRIX_C_INIT = Array(16).fill(0);

// Code for display
const CUDA_CODE = `__global__ void matrixMulShared(float *A, float *B, float *C, int N) {
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = blockIdx.y * TILE_WIDTH + ty;
  int col = blockIdx.x * TILE_WIDTH + tx;
  float acc = 0.0f;

  for (int i = 0; i < N / TILE_WIDTH; i++) {
    // Load Global -> Shared (Coalesced)
    As[ty][tx] = A[row * N + (i * TILE_WIDTH + tx)];
    Bs[ty][tx] = B[(i * TILE_WIDTH + ty) * N + col];
    
    __syncthreads(); // Wait for load

    // Compute from Shared (Fast)
    for (int k = 0; k < TILE_WIDTH; k++) {
       acc += As[ty][k] * Bs[k][tx];
    }
    
    __syncthreads(); // Wait for compute
  }
  
  if (row < N && col < N) {
    C[row * N + col] = acc;
  }
}`;

type StepType = {
  id: number;
  lineHighlight: number[]; // Lines of code to highlight
  description: string;
  phase: 'INIT' | 'LOAD_A' | 'LOAD_B' | 'SYNC' | 'COMPUTE' | 'WRITE';
  activeTile: number; // 0 or 1 (since 4/2 = 2 tiles)
  globalHighlightA?: number[]; // Indices in Global Mem A to highlight
  globalHighlightB?: number[]; // Indices in Global Mem B to highlight
  sharedHighlightAs?: number[]; // Indices in Shared Mem As to highlight/fill
  sharedHighlightBs?: number[]; // Indices in Shared Mem Bs to highlight/fill
  computeHighlight?: { k: number }; // For showing dot product calculation
  accUpdate?: boolean; // Whether to update accumulator visualization
  barrier?: boolean;
};

// --- Simulation Logic (The "Script") ---
// Simulating Block(0,0) handling top-left 2x2 of C
const generateSteps = (): StepType[] => {
  const steps: StepType[] = [];
  let id = 0;

  // 1. Init
  steps.push({
    id: id++, lineHighlight: [5, 6, 7], description: "Initialization: Threads calculate global row/col indices. Registers are set to 0.", phase: 'INIT', activeTile: -1
  });

  // Loop over tiles (i = 0, i = 1)
  for (let tileIdx = 0; tileIdx < MATRIX_SIZE / TILE_WIDTH; tileIdx++) {
    const baseOffset = tileIdx * TILE_WIDTH;

    // Loop Start
    steps.push({
      id: id++, lineHighlight: [9], description: `Loop i=${tileIdx}: Processing Tile #${tileIdx}. The Block moves its focus to a new chunk of A and B.`, phase: 'INIT', activeTile: tileIdx
    });

    // --- Load A ---
    // In our 2x2 block, all threads load.
    // Row 0 threads (Warp 0) load A[0][base + 0] and A[0][base + 1] -> Coalesced
    const loadAIndices = [
      (0 * MATRIX_SIZE) + (baseOffset + 0), (0 * MATRIX_SIZE) + (baseOffset + 1),
      (1 * MATRIX_SIZE) + (baseOffset + 0), (1 * MATRIX_SIZE) + (baseOffset + 1)
    ];
    
    steps.push({
      id: id++, lineHighlight: [11], 
      description: "Load A -> Shared 'As'. Notice the GREEN BOX: Threads in a Warp define a continuous range of addresses. This is a Coalesced Memory Access (High Bandwidth).",
      phase: 'LOAD_A', activeTile: tileIdx,
      globalHighlightA: loadAIndices,
      sharedHighlightAs: [0, 1, 2, 3]
    });

    // --- Load B ---
    const loadBIndices = [
        (baseOffset + 0) * MATRIX_SIZE + 0, (baseOffset + 0) * MATRIX_SIZE + 1,
        (baseOffset + 1) * MATRIX_SIZE + 0, (baseOffset + 1) * MATRIX_SIZE + 1
    ];

    steps.push({
      id: id++, lineHighlight: [12], 
      description: "Load B -> Shared 'Bs'. Threads load their corresponding values. Also coalesced in this optimized tiling pattern.",
      phase: 'LOAD_B', activeTile: tileIdx,
      globalHighlightB: loadBIndices,
      sharedHighlightBs: [0, 1, 2, 3]
    });

    // --- Sync 1 ---
    steps.push({
      id: id++, lineHighlight: [14], description: "__syncthreads(): BARRIER. All threads pause here to ensure Shared Memory is fully populated before any calculation starts.", phase: 'SYNC', activeTile: tileIdx, barrier: true
    });

    // --- Compute Loop (k=0..TILE_WIDTH) ---
    for(let k=0; k<TILE_WIDTH; k++) {
        steps.push({
            id: id++, lineHighlight: [17, 18], 
            description: `Compute (k=${k}): Threads read As[*][${k}] and Bs[${k}][*] from Shared Memory (L1 Speed) and accumulate into Register 'acc'.`,
            phase: 'COMPUTE', activeTile: tileIdx,
            computeHighlight: { k: k },
            accUpdate: true
        });
    }

    // --- Sync 2 ---
    steps.push({
      id: id++, lineHighlight: [21], description: "__syncthreads(): BARRIER. Wait for all threads to finish using the current tile data before overwriting it in the next loop.", phase: 'SYNC', activeTile: tileIdx, barrier: true
    });
  }

  // Write C
  steps.push({
    id: id++, lineHighlight: [24, 25], description: "Write Back: Threads write their final 'acc' register value to Global Memory C.", phase: 'WRITE', activeTile: -1
  });

  return steps;
};

const STEPS = generateSteps();

// --- Components ---

const MatrixCell = ({ value, label, highlight, colorClass, isHeader = false, dim = false }: any) => (
  <motion.div 
    layout
    initial={{ scale: 0.8, opacity: 0 }}
    animate={{ 
        scale: highlight ? 1.1 : 1, 
        opacity: dim ? 0.3 : 1, 
        borderColor: highlight ? '#facc15' : 'transparent',
        backgroundColor: highlight ? 'rgba(250, 204, 21, 0.2)' : ''
    }}
    className={`
      w-8 h-8 md:w-9 md:h-9 flex items-center justify-center border rounded text-xs font-mono relative transition-colors duration-300
      ${isHeader ? 'bg-transparent border-none text-gray-500' : `${colorClass} border-slate-700`}
      ${highlight ? 'ring-2 ring-yellow-400 z-10 shadow-lg !border-yellow-400' : ''}
    `}
  >
    {value}
    {label && <span className="absolute -top-3 -left-2 text-[8px] text-gray-500">{label}</span>}
  </motion.div>
);

export const metadata = {
  name: '矩阵乘法',
  description: '',
  image: '',
};

export default function CudaVisualizer() {
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [accumulators, setAccumulators] = useState([0,0,0,0]);
  const [sharedAs, setSharedAs] = useState([0,0,0,0]);
  const [sharedBs, setSharedBs] = useState([0,0,0,0]);
  const [matrixC, setMatrixC] = useState([...MATRIX_C_INIT]);

  const step = STEPS[currentStepIdx];
  const timerRef = useRef<number | null>(null);

  // --- Animation Loop ---
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setCurrentStepIdx(prev => {
          if (prev < STEPS.length - 1) return prev + 1;
          setIsPlaying(false);
          return prev;
        });
      }, 1500); // 1.5s per step
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isPlaying]);

  // --- Logic Applicator ---
  useEffect(() => {
    const s = STEPS[currentStepIdx];

    // Reset logic if at start
    if (currentStepIdx === 0) {
        setAccumulators([0,0,0,0]);
        setSharedAs([0,0,0,0]);
        setSharedBs([0,0,0,0]);
        setMatrixC([...MATRIX_C_INIT]);
        return;
    }

    if (s.phase === 'LOAD_A' && s.globalHighlightA) {
        const newAs = [...sharedAs];
        s.globalHighlightA.forEach((gIdx, i) => newAs[i] = MATRIX_A[gIdx]);
        setSharedAs(newAs);
    }

    if (s.phase === 'LOAD_B' && s.globalHighlightB) {
        const newBs = [...sharedBs];
        s.globalHighlightB.forEach((gIdx, i) => newBs[i] = MATRIX_B[gIdx]);
        setSharedBs(newBs);
    }

    if (s.phase === 'COMPUTE' && s.computeHighlight) {
        const k = s.computeHighlight.k;
        const newAcc = [...accumulators];
        // Hardcoded dot product logic for 2x2 block
        // acc[row][col] += As[row][k] * Bs[k][col]
        // Flattened: row=0,1 col=0,1
        // T0 (0,0): As[0][k] * Bs[k][0]
        newAcc[0] += sharedAs[0*2 + k] * sharedBs[k*2 + 0];
        // T1 (0,1): As[0][k] * Bs[k][1]
        newAcc[1] += sharedAs[0*2 + k] * sharedBs[k*2 + 1];
        // T2 (1,0): As[1][k] * Bs[k][0]
        newAcc[2] += sharedAs[1*2 + k] * sharedBs[k*2 + 0];
        // T3 (1,1): As[1][k] * Bs[k][1]
        newAcc[3] += sharedAs[1*2 + k] * sharedBs[k*2 + 1];
        setAccumulators(newAcc);
    }

    if (s.phase === 'WRITE') {
        const newC = [...matrixC];
        newC[0] = accumulators[0];
        newC[1] = accumulators[1];
        newC[4] = accumulators[2];
        newC[5] = accumulators[3];
        setMatrixC(newC);
    }

  }, [currentStepIdx]);

  // --- Handlers ---
  const handleNext = () => setCurrentStepIdx(prev => Math.min(prev + 1, STEPS.length - 1));
  const handlePrev = () => setCurrentStepIdx(prev => Math.max(prev - 1, 0));
  const handleReset = () => { setIsPlaying(false); setCurrentStepIdx(0); };

  // --- Render Helpers ---
  const renderCoalescingBox = (active: boolean, offsetTop: number, offsetLeft: number) => {
      if(!active) return null;
      return (
        <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute border-2 border-green-400 bg-green-400/20 pointer-events-none z-20 flex items-start justify-center"
            style={{
                top: offsetTop, left: offsetLeft, 
                width: '80px', height: '80px', // Covers 2x2 cells
                borderRadius: '8px'
            }}
        >
            <span className="bg-green-600 text-white text-[9px] px-1 rounded shadow -mt-2">Coalesced</span>
        </motion.div>
      )
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden selection:bg-blue-500/30">
      
      {/* Header */}
      <header className="h-14 border-b border-slate-800 flex items-center px-6 bg-slate-900 shadow-sm z-30 shrink-0">
        <Cpu className="w-5 h-5 text-green-400 mr-2" />
        <h1 className="text-base font-bold tracking-wide text-slate-200">CUDA Visualizer: <span className="text-blue-400">Tiled Matrix Multiplication</span></h1>
        
        {/* Legend */}
        <div className="ml-auto hidden md:flex items-center space-x-4 text-xs text-slate-400">
            <div className="flex items-center"><div className="w-2 h-2 rounded-full bg-blue-500 mr-1.5"></div>Global Mem</div>
            <div className="flex items-center"><div className="w-2 h-2 rounded-full bg-orange-500 mr-1.5"></div>Shared Mem</div>
            <div className="flex items-center"><div className="w-2 h-2 rounded-full bg-slate-600 mr-1.5"></div>Registers</div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* Left: Global Memory */}
        <div className="w-72 lg:w-80 flex flex-col border-r border-slate-800 bg-slate-900/50 overflow-y-auto">
            <div className="p-4 space-y-8">
                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center mb-2">
                    <Database className="w-3 h-3 mr-1"/> Global DRAM
                </div>

                {/* Matrix A */}
                <div className="relative">
                    <div className="text-xs text-blue-400 mb-2 font-mono">Matrix A [4x4]</div>
                    <div className="grid grid-cols-4 gap-1 p-2 bg-slate-900 rounded border border-slate-800 w-max">
                        {MATRIX_A.map((val, idx) => (
                            <MatrixCell 
                                key={`A-${idx}`} value={val} 
                                colorClass="bg-blue-900/30 text-blue-200"
                                highlight={step.globalHighlightA?.includes(idx)}
                                dim={step.phase === 'LOAD_A' && !step.globalHighlightA?.includes(idx)}
                            />
                        ))}
                    </div>
                    {/* Coalescing Box A */}
                    {step.phase === 'LOAD_A' && step.activeTile !== -1 && renderCoalescingBox(true, 30, 8 + (step.activeTile * 80))}
                </div>

                {/* Matrix B */}
                <div className="relative">
                    <div className="text-xs text-purple-400 mb-2 font-mono">Matrix B [4x4]</div>
                    <div className="grid grid-cols-4 gap-1 p-2 bg-slate-900 rounded border border-slate-800 w-max">
                        {MATRIX_B.map((val, idx) => (
                            <MatrixCell 
                                key={`B-${idx}`} value={val} 
                                colorClass="bg-purple-900/30 text-purple-200"
                                highlight={step.globalHighlightB?.includes(idx)}
                                dim={step.phase === 'LOAD_B' && !step.globalHighlightB?.includes(idx)}
                            />
                        ))}
                    </div>
                     {/* Coalescing Box B */}
                     {step.phase === 'LOAD_B' && step.activeTile !== -1 && renderCoalescingBox(true, 30 + (step.activeTile * 80), 8)}
                </div>

                {/* Matrix C */}
                <div className="relative">
                    <div className="text-xs text-green-400 mb-2 font-mono">Matrix C (Result)</div>
                    <div className="grid grid-cols-4 gap-1 p-2 bg-slate-900 rounded border border-slate-800 w-max">
                        {matrixC.map((val, idx) => (
                            <MatrixCell 
                                key={`C-${idx}`} value={val.toFixed(0)} 
                                colorClass="bg-green-900/30 text-green-200"
                                highlight={step.phase === 'WRITE' && [0,1,4,5].includes(idx)}
                            />
                        ))}
                    </div>
                    {step.phase === 'WRITE' && (
                        <motion.div 
                        style={{
                            top: 30
                        }}
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                            className="absolute top-2 left-2 w-[80px] h-[80px] border-2 border-green-400 bg-green-400/10 pointer-events-none rounded"
                        />
                    )}
                </div>
            </div>
        </div>

        {/* Center: SM Simulator */}
        <div className="flex-1 bg-slate-950 relative flex flex-col">
            {/* Background Grid Decoration */}
            <div className="absolute inset-0 opacity-[0.03] bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>

            <div className="absolute top-4 right-4 text-xs font-mono text-slate-600 border border-slate-800 px-2 py-1 rounded bg-slate-900">
                Focus: Block (0, 0)
            </div>

            <div className="flex-1 flex flex-col items-center justify-center p-6 space-y-10 z-10">
                
                {/* L1 Shared Memory */}
                <div className="relative p-6 bg-orange-500/5 border border-orange-500/20 rounded-xl backdrop-blur-sm shadow-2xl w-full max-w-md">
                    <div className="absolute -top-3 left-4 bg-slate-950 px-2 text-xs font-bold text-orange-400 flex items-center border border-orange-500/30 rounded">
                        <Layers className="w-3 h-3 mr-1"/> Shared Memory (L1)
                    </div>
                    
                    <div className="flex justify-around items-center">
                        <div className="text-center">
                            <div className="text-[10px] text-slate-500 mb-2">As [2x2]</div>
                            <div className="grid grid-cols-2 gap-2">
                                {sharedAs.map((val, idx) => {
                                                                         const r = Math.floor(idx/2);
                                                                         const c = idx%2;
                                                                        const isHighlighted = step.phase === 'COMPUTE' && step.computeHighlight && r === Math.floor(idx/2) && c === step.computeHighlight.k; // Simplified visual
                                    
                                    return (
                                        <MatrixCell 
                                            key={`As-${idx}`} value={val} 
                                            colorClass="bg-orange-500/20 text-orange-100"
                                            highlight={step.sharedHighlightAs?.includes(idx) || isHighlighted}
                                        />
                                    )
                                })}
                            </div>
                        </div>

                        <div className="text-slate-600"><ArrowRight size={20}/></div>

                        <div className="text-center">
                            <div className="text-[10px] text-slate-500 mb-2">Bs [2x2]</div>
                            <div className="grid grid-cols-2 gap-2">
                                {sharedBs.map((val, idx) => {
                                     const r = Math.floor(idx/2);
                                     // Highlight Row in Bs? No, we highlight based on k loop
                                     // We need Bs[k][tx]. 
                                     const isHighlighted = step.phase === 'COMPUTE' && step.computeHighlight && r === step.computeHighlight.k;
                                     return (
                                        <MatrixCell 
                                            key={`Bs-${idx}`} value={val} 
                                            colorClass="bg-orange-500/20 text-orange-100"
                                            highlight={step.sharedHighlightBs?.includes(idx) || isHighlighted}
                                        />
                                     )
                                })}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Barrier Indicator */}
                <div className="h-6 flex items-center justify-center w-full">
                    <AnimatePresence>
                    {step.barrier && (
                        <motion.div 
                            initial={{ width: 0, opacity: 0 }} 
                            animate={{ width: "60%", opacity: 1 }} 
                            exit={{ width: 0, opacity: 0 }}
                            className="h-0.5 bg-yellow-400 shadow-[0_0_15px_rgba(250,204,21,0.6)] rounded-full flex items-center justify-center"
                        >
                            <span className="text-[9px] text-yellow-400 bg-slate-950 px-2 -mt-4">__syncthreads()</span>
                        </motion.div>
                    )}
                    </AnimatePresence>
                </div>

                {/* Threads */}
                <div className="w-full max-w-md">
                    <div className="flex items-center justify-between mb-3">
                         <div className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center">
                            <Cpu className="w-3 h-3 mr-1"/> Active Block Threads
                        </div>
                    </div>
                   
                    <div className="grid grid-cols-2 gap-4">
                        {[0, 1, 2, 3].map((tIdx) => {
                            const tx = tIdx % 2;
                            const ty = Math.floor(tIdx / 2);
                            const isComputing = step.phase === 'COMPUTE';
                            
                            return (
                                <motion.div 
                                    key={tIdx}
                                    layout
                                    className={`
                                        p-3 rounded-lg border transition-all duration-300 relative overflow-hidden
                                        ${isComputing ? 'bg-slate-800/80 border-green-500/50 shadow-[0_0_20px_rgba(34,197,94,0.1)]' : 'bg-slate-900 border-slate-800'}
                                    `}
                                >
                                    <div className="flex justify-between items-center mb-2 border-b border-slate-700/50 pb-1">
                                        <span className="text-[10px] font-mono text-slate-400">Thread({tx},{ty})</span>
                                        <span className={`text-[9px] px-1.5 rounded-full ${tIdx < 2 ? 'bg-indigo-500/20 text-indigo-300' : 'bg-pink-500/20 text-pink-300'}`}>
                                            Warp {tIdx < 2 ? 0 : 1}
                                        </span>
                                    </div>
                                    
                                    <div className="flex justify-between items-center">
                                        <span className="text-[10px] text-slate-500 font-mono">reg[acc]</span>
                                        <motion.span 
                                            key={accumulators[tIdx]}
                                            initial={{ scale: 1.2, color: '#4ade80' }}
                                            animate={{ scale: 1, color: step.accUpdate ? '#4ade80' : '#cbd5e1' }}
                                            className="text-sm font-bold font-mono"
                                        >
                                            {accumulators[tIdx].toFixed(0)}
                                        </motion.span>
                                    </div>

                                    {/* Compute Micro-Visualization */}
                                    {isComputing && (
                                        <motion.div 
                                            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                                            className="absolute inset-0 bg-green-500/5 pointer-events-none flex items-center justify-center"
                                        >
                                            <div className="text-[8px] text-green-300 font-mono bg-slate-900/90 px-1 border border-green-500/30 rounded">
                                                MAC
                                            </div>
                                        </motion.div>
                                    )}
                                </motion.div>
                            )
                        })}
                    </div>
                </div>

            </div>
        </div>

        {/* Right: Code & Controls */}
        <div className="w-80 md:w-96 flex flex-col border-l border-slate-800 bg-slate-900">
            
            {/* Control Bar */}
            <div className="p-4 bg-slate-900 border-b border-slate-800 flex justify-center space-x-3 shrink-0">
                <button onClick={handleReset} className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 transition-colors" title="Reset"><RotateCcw size={16}/></button>
                <button onClick={handlePrev} className="p-2 rounded-lg hover:bg-slate-800 text-slate-200 transition-colors" title="Step Back"><ChevronLeft size={20}/></button>
                <button onClick={() => setIsPlaying(!isPlaying)} className={`p-2 w-12 flex justify-center rounded-lg shadow-lg transition-all ${isPlaying ? 'bg-red-500/20 text-red-400 border border-red-500/50' : 'bg-blue-500 text-white hover:bg-blue-400'}`}>
                    {isPlaying ? <Pause size={20}/> : <Play size={20}/>}
                </button>
                <button onClick={handleNext} className="p-2 rounded-lg hover:bg-slate-800 text-slate-200 transition-colors" title="Step Forward"><ChevronRight size={20}/></button>
            </div>

            {/* Explainer */}
            <div className="p-5 bg-slate-800/40 border-b border-slate-800 shrink-0 min-h-[140px]">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-bold text-blue-400 uppercase tracking-widest bg-blue-500/10 px-2 py-0.5 rounded">
                        Phase: {step.phase}
                    </span>
                    <span className="text-[10px] text-slate-500">Step {currentStepIdx + 1}/{STEPS.length}</span>
                </div>
                <p className="text-sm text-slate-300 leading-relaxed font-light">
                    {step.description}
                </p>
            </div>

            {/* Code View */}
            <div className="flex-1 overflow-auto bg-[#0d1117] p-4 font-mono text-[10px] sm:text-xs relative">
                <div className="absolute top-2 right-4 text-slate-600"><Code size={14}/></div>
                {CUDA_CODE.split('\n').map((line, i) => {
                    const isHighlighted = step.lineHighlight.includes(i);
                    return (
                        <div 
                            key={i} 
                            className={`flex px-1 py-0.5 transition-colors duration-200 ${isHighlighted ? 'bg-blue-500/20' : ''}`}
                        >
                            <span className={`w-6 text-right mr-3 select-none ${isHighlighted ? 'text-blue-400 font-bold' : 'text-slate-600'}`}>{i + 1}</span>
                            <span className={`${isHighlighted ? 'text-slate-100' : 'text-slate-500'}`}>
                                {line}
                            </span>
                        </div>
                    )
                })}
            </div>
        </div>

      </div>
    </div>
  );
}