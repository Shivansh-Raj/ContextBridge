import { useState } from 'react'

// Reflects the new LangGraph pipeline stages
const PIPELINE_STAGES = [
  { id: 1, label: 'Chunking', model: 'rule-based', detail: 'Splitting conversation into semantic chunks' },
  { id: 2, label: 'Classify + Memory loop', model: 'Groq · Llama / Instant', detail: 'Per-chunk classification and incremental memory updates' },
  { id: 3, label: 'Output generation', model: 'Groq · Llama 3.3 70B', detail: 'Formatting 3 memory documents' },
]

const OUTPUT_TABS = [
  { id: 'intent',      label: 'Intent & Context',  file: '01_project_intent.md',      color: 'blue',   detailedOnly: false },
  { id: 'code',        label: 'Code Understanding', file: '02_code_understanding.md',  color: 'purple', detailedOnly: false },
  { id: 'structure',   label: 'Project Structure',  file: '03_project_structure.md',   color: 'emerald',detailedOnly: false },
  { id: 'code_detail', label: 'Detailed Code',      file: '04_detailed_code.md',       color: 'amber',  detailedOnly: true  },
]

const TAB_COLORS = {
  blue:    { active: 'border-blue-500 text-blue-300 bg-blue-950/30',       inactive: 'border-gray-700 text-gray-500 hover:text-gray-300' },
  purple:  { active: 'border-purple-500 text-purple-300 bg-purple-950/30', inactive: 'border-gray-700 text-gray-500 hover:text-gray-300' },
  emerald: { active: 'border-emerald-500 text-emerald-300 bg-emerald-950/30', inactive: 'border-gray-700 text-gray-500 hover:text-gray-300' },
  amber:   { active: 'border-amber-500 text-amber-300 bg-amber-950/30',    inactive: 'border-gray-700 text-gray-500 hover:text-gray-300' },
}

function StageIndicator({ currentStage }) {
  return (
    <div className="w-full max-w-md space-y-3">
      {PIPELINE_STAGES.map(stage => {
        const isDone = currentStage > stage.id
        const isActive = currentStage === stage.id
        return (
          <div
            key={stage.id}
            className={`flex items-center gap-4 px-4 py-3 rounded-lg border transition-all duration-500 ${
              isActive  ? 'border-blue-500 bg-blue-950/50'
              : isDone  ? 'border-green-800 bg-green-950/30'
              : 'border-gray-800 bg-gray-900/50'
            }`}
          >
            <div className="w-6 h-6 shrink-0 flex items-center justify-center text-base">
              {isDone    ? <span className="text-green-400">✓</span>
              : isActive ? <span className="text-blue-400 animate-spin inline-block">◌</span>
              :             <span className="text-gray-700">○</span>}
            </div>
            <div className="min-w-0">
              <p className={`text-sm font-medium ${isActive ? 'text-white' : isDone ? 'text-green-400' : 'text-gray-600'}`}>
                Stage {stage.id} — {stage.label}
              </p>
              <p className="text-xs text-gray-600 truncate">{stage.model}</p>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function OutputTabs({ result, outputFiles }) {
  const isDetailed = result.mode === 'detailed'
  const visibleTabs = OUTPUT_TABS.filter(t => !t.detailedOnly || isDetailed)
  const [activeTab, setActiveTab] = useState('intent')
  const [copied, setCopied]       = useState(null)

  async function handleCopy(tabId, text) {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const el = document.createElement('textarea')
      el.value = text
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
    }
    setCopied(tabId)
    setTimeout(() => setCopied(null), 2000)
  }

  const currentTab = visibleTabs.find(t => t.id === activeTab)
  const currentContent = outputFiles[activeTab] || ''

  return (
    <div className="space-y-4">
      {/* Meta bar */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-green-400 text-sm font-medium">✓ Memory documents ready</span>
          <span className="text-xs text-gray-500 border border-gray-800 bg-gray-900 px-2 py-0.5 rounded-md">
            {result.token_count.toLocaleString()} tokens
          </span>
          <span className="text-xs text-gray-500 border border-gray-800 bg-gray-900 px-2 py-0.5 rounded-md">
            {result.chunks_processed} chunks
          </span>
          <span className="text-xs text-gray-500 border border-gray-800 bg-gray-900 px-2 py-0.5 rounded-md">
            {(result.processing_time_ms / 1000).toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-800 flex-wrap">
        {visibleTabs.map(tab => {
          const colors = TAB_COLORS[tab.color]
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-all ${
                isActive ? colors.active : colors.inactive
              }`}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div className="relative">
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
          <span className="text-xs text-gray-600">{currentTab?.file}</span>
          <button
            onClick={() => handleCopy(activeTab, currentContent)}
            className={`text-xs px-3 py-1.5 rounded border transition-all ${
              copied === activeTab
                ? 'bg-green-900/50 border-green-700 text-green-300'
                : 'bg-gray-900/80 border-gray-700 text-gray-300 hover:border-blue-500 hover:text-blue-300'
            }`}
          >
            {copied === activeTab ? '✓ Copied' : 'Copy'}
          </button>
        </div>
        <pre className="bg-gray-900 border border-gray-800 rounded-xl p-5 pt-10 text-xs text-gray-300 leading-relaxed whitespace-pre-wrap overflow-auto max-h-[60vh]">
          {currentContent || '(no content for this section)'}
        </pre>
      </div>

      <p className="text-xs text-gray-600 text-center">
        Paste any of the above into a new AI session — each file is self-contained.
      </p>
    </div>
  )
}

export default function App() {
  const [text, setText]               = useState('')
  const [mode, setMode]               = useState('detailed')
  const [status, setStatus]           = useState('idle')
  const [currentStage, setCurrentStage] = useState(0)
  const [result, setResult]           = useState(null)
  const [error, setError]             = useState('')

  async function handleProcess() {
    if (!text.trim()) return
    setStatus('processing')
    setCurrentStage(1)
    setResult(null)
    setError('')

    const t1 = setTimeout(() => setCurrentStage(2), 1500)
    const t2 = setTimeout(() => setCurrentStage(3), 8000)

    try {
      const resp = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_text: text, mode, token_target: null }),
      })

      clearTimeout(t1)
      clearTimeout(t2)
      setCurrentStage(3)

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }

      const data = await resp.json()
      setResult(data)
      setStatus('done')
    } catch (err) {
      clearTimeout(t1)
      clearTimeout(t2)
      setError(err.message)
      setStatus('error')
    }
  }

  function handleReset() {
    setStatus('idle')
    setCurrentStage(0)
    setResult(null)
    setError('')
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100" style={{ fontFamily: "'JetBrains Mono','Fira Code','Cascadia Code',monospace" }}>
      {/* Header */}
      <header className="border-b border-gray-800/60 px-6 py-4 sticky top-0 bg-gray-950/95 backdrop-blur z-10">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded bg-blue-600 flex items-center justify-center text-white text-xs font-bold">CB</div>
            <div>
              <span className="text-base font-bold text-white tracking-tight">
                Context<span className="text-blue-400">Bridge</span>
              </span>
              <span className="text-gray-600 text-xs ml-2 hidden sm:inline">stateful memory pipeline</span>
            </div>
          </div>
          {status !== 'idle' && (
            <button
              onClick={handleReset}
              className="text-xs text-gray-400 hover:text-white transition-colors border border-gray-700 hover:border-gray-500 px-3 py-1.5 rounded-md"
            >
              ← New session
            </button>
          )}
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10">

        {/* ── INPUT ── */}
        {status === 'idle' && (
          <div className="space-y-7">
            <div className="space-y-2">
              <h2 className="text-2xl font-bold text-white">Continue any AI conversation</h2>
              <p className="text-gray-400 text-sm leading-relaxed max-w-2xl">
                Paste a conversation export. ContextBridge reads it chunk by chunk,
                builds three evolving memory documents — Intent, Code, and Structure —
                and outputs them ready to paste into any new AI session.
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-gray-500 uppercase tracking-widest">Conversation Export</label>
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                placeholder={"Paste your exported conversation here...\n\nSupports: plain text, ChatGPT JSON export, or any AI platform output."}
                className="w-full h-72 bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500/80 focus:ring-1 focus:ring-blue-500/20 resize-y transition-colors"
              />
              <p className="text-xs text-gray-600">
                {text.length > 0
                  ? `${text.length.toLocaleString()} chars · ~${Math.round(text.length / 4).toLocaleString()} tokens`
                  : 'No input'}
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-gray-500 uppercase tracking-widest">Output Mode</label>
              <div className="flex gap-2 max-w-xs">
                {[
                  { value: 'detailed',    label: 'Detailed' },
                  { value: 'summarized',  label: 'Summarized' },
                ].map(m => (
                  <button
                    key={m.value}
                    onClick={() => setMode(m.value)}
                    className={`flex-1 py-2 px-4 rounded-lg text-sm border font-medium transition-all ${
                      mode === m.value
                        ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/30'
                        : 'bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500 hover:text-gray-200'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleProcess}
              disabled={!text.trim()}
              className="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-all text-sm tracking-wide shadow-lg shadow-blue-900/20"
            >
              Process Conversation →
            </button>

            {/* Pipeline diagram */}
            <div className="border border-gray-800 rounded-xl p-5 bg-gray-900/40 space-y-4">
              <p className="text-xs text-gray-500 uppercase tracking-widest">How the memory pipeline works</p>
              <div className="space-y-2 text-xs">
                {[
                  { step: '1', label: 'Chunker',          desc: 'Splits conversation into token-bounded semantic chunks' },
                  { step: '2', label: 'Classifier',       desc: 'Labels each chunk: code_update · architecture · debugging · ...' },
                  { step: '3', label: 'Memory Dispatcher',desc: 'Routes to Intent / Code / Structure updaters based on labels' },
                  { step: '4', label: 'Loop',             desc: 'Repeats classifier → dispatcher for every chunk' },
                  { step: '5', label: 'Output Generator', desc: 'Formats the 3 evolved memory layers into markdown documents' },
                ].map(s => (
                  <div key={s.step} className="flex gap-3">
                    <span className="text-gray-700 w-4 text-right shrink-0">{s.step}</span>
                    <span className="text-blue-400 w-36 shrink-0 font-medium">{s.label}</span>
                    <span className="text-gray-500 hidden sm:block">{s.desc}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-700 pt-1 border-t border-gray-800">
                Memory is patched incrementally — never reset, never duplicated
              </p>
            </div>
          </div>
        )}

        {/* ── PROCESSING ── */}
        {status === 'processing' && (
          <div className="flex flex-col items-center justify-center min-h-80 space-y-10">
            <div className="text-center space-y-2">
              <h2 className="text-xl font-semibold text-white">Building memory documents</h2>
              <p className="text-gray-500 text-sm">Processing conversation chunk by chunk</p>
            </div>
            <StageIndicator currentStage={currentStage} />
            <p className="text-xs text-gray-700">Time varies with conversation length — typically 30 s to 2 min</p>
          </div>
        )}

        {/* ── ERROR ── */}
        {status === 'error' && (
          <div className="flex flex-col items-center justify-center min-h-64 space-y-6">
            <div className="border border-red-800/60 bg-red-950/30 rounded-xl p-6 w-full max-w-lg space-y-3">
              <p className="text-red-400 font-semibold text-sm">Processing failed</p>
              <p className="text-red-300/80 text-sm leading-relaxed">{error}</p>
            </div>
            <button onClick={handleReset} className="text-sm text-blue-400 hover:text-blue-300">← Try again</button>
          </div>
        )}

        {/* ── OUTPUT ── */}
        {status === 'done' && result && (
          <OutputTabs
            result={result}
            outputFiles={result.output_files || {}}
          />
        )}

      </main>
    </div>
  )
}
