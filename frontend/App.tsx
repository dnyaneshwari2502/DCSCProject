import React, { useState } from 'react';
import { PredictionResponse, AnalysisState, ScoreLevel } from './types';
import { rewriteHeadline } from './services/geminiService';
import { 
  ChartBarIcon, 
  ExclamationTriangleIcon, 
  CheckCircleIcon, 
  SparklesIcon,
  ArrowPathIcon,
  ShieldCheckIcon,
  FireIcon,
  UserGroupIcon
} from '@heroicons/react/24/solid';

// --- Helper Components ---

const ProgressBar = ({ score, label, colorClass }: { score: number; label: string; colorClass: string }) => (
  <div className="mb-5">
    <div className="flex justify-between mb-2">
      <span className="text-sm font-medium text-slate-300 tracking-wide">{label}</span>
      <span className="text-sm font-bold text-white">{(score * 100).toFixed(1)}%</span>
    </div>
    <div className="w-full bg-slate-700/50 rounded-full h-3 backdrop-blur-sm border border-slate-600/30 overflow-hidden">
      <div 
        className={`h-full rounded-full transition-all duration-1000 ease-out shadow-lg ${colorClass}`} 
        style={{ width: `${score * 100}%` }}
      ></div>
    </div>
  </div>
);

const StatCard = ({ title, value, subtext, icon: Icon, color, gradient }: { title: string; value: string; subtext: string; icon: any; color: string; gradient: string }) => (
  <div className="relative overflow-hidden bg-slate-800/40 backdrop-blur-md border border-white/10 rounded-2xl p-6 flex flex-col items-start shadow-xl group hover:border-white/20 transition-all duration-300">
    <div className={`absolute top-0 right-0 w-24 h-24 bg-gradient-to-br ${gradient} opacity-10 rounded-bl-full -mr-4 -mt-4 transition-opacity group-hover:opacity-20`}></div>
    
    <div className={`p-3 rounded-xl ${color} bg-opacity-20 mb-4 ring-1 ring-white/10`}>
      <Icon className={`w-6 h-6 ${color.replace('bg-', 'text-')}`} />
    </div>
    <h3 className="text-slate-400 text-xs font-bold uppercase tracking-widest mb-2">{title}</h3>
    <p className="text-2xl font-bold text-white mb-1 tracking-tight">{value}</p>
    <p className="text-slate-500 text-sm font-medium">{subtext}</p>
  </div>
);

// --- Main Component ---

export default function App() {
  const [inputText, setInputText] = useState('');
  const [analysis, setAnalysis] = useState<AnalysisState>({
    isLoading: false,
    error: null,
    data: null,
  });
  
  const [isRewriting, setIsRewriting] = useState(false);
  const [rewrittenHeadline, setRewrittenHeadline] = useState<string | null>(null);

  const getScoreLevel = (score: number): ScoreLevel => {
    if (score < 0.3) return ScoreLevel.SAFE;
    if (score < 0.7) return ScoreLevel.CAUTION;
    return ScoreLevel.DANGER;
  };

  const getColorForScore = (score: number) => {
    const level = getScoreLevel(score);
    switch (level) {
      case ScoreLevel.SAFE: return 'bg-emerald-500 shadow-emerald-500/50';
      case ScoreLevel.CAUTION: return 'bg-amber-500 shadow-amber-500/50';
      case ScoreLevel.DANGER: return 'bg-rose-500 shadow-rose-500/50';
    }
  };

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;

    setAnalysis({ isLoading: true, error: null, data: null });
    setRewrittenHeadline(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText })
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const result: PredictionResponse = await response.json();
      
      // Add artificial delay for smoother UX if response is too fast
      if (Date.now() % 2 === 0) await new Promise(r => setTimeout(r, 500));

      setAnalysis({
        isLoading: false,
        error: null,
        data: result
      });

    } catch (err: any) {
      setAnalysis({
        isLoading: false,
        error: "Could not connect to the analysis model. Ensure your local backend is running.",
        data: null
      });
    }
  };

  const handleRewrite = async () => {
    if (!inputText) return;
    setIsRewriting(true);
    try {
      const newHeadline = await rewriteHeadline(inputText);
      setRewrittenHeadline(newHeadline);
    } catch (e) {
      console.error(e);
    } finally {
      setIsRewriting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-[#0f172a] to-black text-slate-100 flex flex-col font-sans">
      
      {/* Navbar / Brand */}
      <nav className="w-full p-6 flex justify-center border-b border-white/5 bg-black/20 backdrop-blur-sm">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <ShieldCheckIcon className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight text-white">Click-a-Bait</span>
        </div>
      </nav>

      <main className="flex-grow flex flex-col items-center p-4 md:p-10 max-w-5xl mx-auto w-full">
        
        {/* Hero Section */}
        <header className="text-center mb-12 mt-4 animate-float">
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight mb-6 text-transparent bg-clip-text bg-gradient-to-r from-white via-slate-200 to-slate-400 drop-shadow-sm">
            Is it <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">Clickbait?</span>
          </h1>
          <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
            Use NLP model to detect exaggerated claims, and emotional manipulation in headlines instantly.
          </p>
        </header>

        {/* Input Card */}
        <div className="w-full max-w-3xl relative group z-10">
          <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
          <div className="relative bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-2xl p-6 md:p-8 shadow-2xl">
            
            <div className="mb-4 flex items-center justify-between">
              <label htmlFor="headline" className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                Analyze Headline
              </label>
            </div>

            <textarea
              id="headline"
              rows={3}
              className="w-full bg-black/30 border border-slate-700/50 rounded-xl p-4 text-xl text-white placeholder-slate-500 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all outline-none resize-none shadow-inner"
              placeholder="Paste a headline here (e.g., 'You Won't Believe This Trick!')"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleAnalyze())}
            />
            
            <div className="mt-6 flex justify-end">
              <button
                onClick={handleAnalyze}
                disabled={analysis.isLoading || !inputText}
                className={`
                  relative overflow-hidden group/btn flex items-center px-8 py-4 rounded-xl font-bold text-white transition-all transform hover:-translate-y-1
                  ${analysis.isLoading || !inputText 
                    ? 'bg-slate-800 cursor-not-allowed opacity-50' 
                    : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:shadow-lg hover:shadow-indigo-500/25 active:scale-95'}
                `}
              >
                {analysis.isLoading ? (
                  <>
                    <ArrowPathIcon className="w-5 h-5 mr-3 animate-spin" />
                    Running Analysis...
                  </>
                ) : (
                  <>
                    <ChartBarIcon className="w-5 h-5 mr-2 group-hover/btn:scale-110 transition-transform" />
                    Detect Clickbait
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {analysis.error && (
          <div className="mt-8 w-full max-w-3xl bg-rose-500/10 border border-rose-500/20 rounded-xl p-4 flex items-center text-rose-300 animate-pulse">
            <ExclamationTriangleIcon className="w-6 h-6 mr-3 flex-shrink-0" />
            <p className="font-medium">{analysis.error}</p>
          </div>
        )}

        {/* Results Display */}
        {analysis.data && (
          <div className="mt-10 w-full max-w-3xl animate-fade-in-up">
            
            {/* Top Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <StatCard 
                title="Verdict"
                value={analysis.data.clickbait_prediction}
                subtext={`Confidence Score: ${(analysis.data.clickbait_score * 100).toFixed(1)}%`}
                icon={analysis.data.clickbait_score > 0.5 ? ExclamationTriangleIcon : CheckCircleIcon}
                color={analysis.data.clickbait_score > 0.5 ? 'bg-rose-500' : 'bg-emerald-500'}
                gradient={analysis.data.clickbait_score > 0.5 ? 'from-rose-500 to-orange-500' : 'from-emerald-500 to-teal-500'}
              />
               <StatCard 
                title="Emotion"
                value={analysis.data.emotion_prediction}
                subtext={`Intensity Score: ${(analysis.data.emotion_score * 100).toFixed(1)}%`}
                icon={FireIcon}
                color="bg-amber-500"
                gradient="from-amber-500 to-yellow-500"
              />
            </div>

            {/* Deep Analysis Card */}
            <div className="bg-slate-800/40 backdrop-blur-md border border-white/10 rounded-2xl p-8 shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500"></div>
              
              <h3 className="text-xl font-bold text-white mb-8 flex items-center">
                <ChartBarIcon className="w-5 h-5 mr-2 text-indigo-400" />
                Metrics
              </h3>
              
              <ProgressBar 
                label="Confidence Score" 
                score={analysis.data.clickbait_score} 
                colorClass={getColorForScore(analysis.data.clickbait_score)} 
              />
              
              <ProgressBar 
                label="Emotional Intensity" 
                score={analysis.data.emotion_score} 
                colorClass="bg-amber-500 shadow-amber-500/50" 
              />

              {/* Gemini Rewrite Section */}
              {analysis.data.clickbait_score > 0.5 && (
                <div className="mt-8 bg-indigo-500/10 border border-indigo-500/20 rounded-xl p-6 relative overflow-hidden group">
                   <div className="absolute inset-0 bg-gradient-to-r from-indigo-600/5 to-purple-600/5 group-hover:opacity-100 opacity-0 transition-opacity duration-500"></div>
                   
                   {rewrittenHeadline && (
                     <div className="mt-5 p-4 bg-slate-950/50 rounded-lg border border-indigo-500/30 animate-fade-in">
                        <span className="text-[10px] text-indigo-400 uppercase font-bold tracking-widest mb-1 block">AI Suggestion</span>
                        <p className="text-lg text-emerald-400 font-medium leading-relaxed">{rewrittenHeadline}</p>
                     </div>
                   )}
                </div>
              )}

              <div className="mt-8 pt-6 border-t border-white/5 flex justify-between items-center text-xs text-slate-500 font-mono">
                <span className="flex items-center"><div className="w-2 h-2 rounded-full bg-emerald-500 mr-2"></div> System Online</span>
                
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Author Footer */}
      <footer className="w-full py-8 bg-black/20 border-t border-white/5 backdrop-blur-lg mt-auto">
        <div className="max-w-5xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-4">
           <div className="flex items-center space-x-2 opacity-70">
              <ShieldCheckIcon className="w-4 h-4 text-slate-400" />
              <span className="text-sm text-slate-400">Cick-a-Bait</span>
           </div>
           
           <div className="flex flex-col items-center md:items-end">
             <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider font-semibold">Developed By</p>
             <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 group cursor-pointer">
                  <span className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">Jayan Agarwal</span>
                </div>
                <div className="h-4 w-px bg-slate-700"></div>
                <div className="flex items-center space-x-2 group cursor-pointer">
                   <span className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">Dnyaneshwari Rakshe</span>
                </div>
             </div>
           </div>
        </div>
      </footer>
    </div>
  );
}