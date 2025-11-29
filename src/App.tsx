import { Routes, Route, Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import './App.css';

// This interface defines the shape of our visualization modules
interface Visualization {
  path: string;
  name: string;
  description: string;
  image: string;
  Component: React.ComponentType;
}

function Home({ visualizations }: { visualizations: Visualization[] }) {
  return (
    <div className="min-h-screen bg-[#0f1117] text-slate-200 font-sans p-4 md:p-6 flex flex-col gap-6">
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-indigo-500 bg-clip-text text-transparent">
            CUDA Operations Visualization
          </h1>
          <p className="text-slate-500 text-sm mt-1">Select an operation to visualize:</p>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {visualizations.map(vis => (
          <Link key={vis.path} to={vis.path} className="group block bg-slate-800 hover:bg-slate-700/80 p-6 rounded-lg transition-all duration-200 border border-slate-700 hover:border-indigo-500">
            <img src={vis.image} alt={`${vis.name} visualization`} className="mb-4 rounded-lg object-cover" />
            <h2 className="text-xl font-bold text-slate-100 group-hover:text-indigo-400">{vis.name}</h2>
            <p className="text-slate-400 mt-2">{vis.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [visualizations, setVisualizations] = useState<Visualization[]>([]);

  useEffect(() => {
    // Vite's glob import to get all visualization components
    const modules = import.meta.glob<{
      default: React.ComponentType;
      metadata: { name: string; description: string; image: string; };
    }>('./components/**/*.tsx', { eager: true });

    const loadedVisualizations = Object.entries(modules).map(([path, module]) => {
      // Create a URL-friendly path from the file path
      const routePath = `/${path.split('/').pop()?.replace('.tsx', '')}`;
      return {
        path: routePath,
        name: module.metadata.name,
        description: module.metadata.description,
        image: module.metadata.image,
        Component: module.default,
      };
    });

    setVisualizations(loadedVisualizations);
  }, []);

  return (
    <Routes>
      <Route path="/" element={<Home visualizations={visualizations} />} />
      {visualizations.map(({ path, Component }) => (
        <Route key={path} path={path} element={<Component />} />
      ))}
    </Routes>
  );
}

export default App;
