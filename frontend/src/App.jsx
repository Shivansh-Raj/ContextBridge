import { useState } from 'react'

const PIPELINE_STAGES = [
  { id: 1, label: 'Segmentation', detail: 'Classifying and cleaning messages', model: 'Groq · Mixtral 8x7B' },
  { id: 2, label: 'Extraction ×4', detail: 'Code · Decisions · Debug · Intent (parallel)', model: 'Groq · Llama 3.3 70B' },
  { id: 3, label: 'Synthesis', detail: 'Producing the continuation package', model: 'Anthropic · Claude Sonnet' },
]

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
              isActive
                ? 'border-blue-500 bg-blue-950/50'
                : isDone
                ? 'border-green-800 bg-green-950/30'
                : 'border-gray-800 bg-gray-900/50'
            }`}
          >
            <div className="w-6 h-6 shrink-0 flex items-center justify-center text-base">
              {isDone ? (
                <span className="text-green-400">✓</span>
              ) : isActive ? (
                <span className="text-blue-400 animate-spin inline-block">◌</span>
              ) : (
                <span className="text-gray-700">○</span>
              )}
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

export default function App() {
  const [text, setText] = useState('')
  const [mode, setMode] = useState('detailed')
  const [status, setStatus] = useState('idle')
  const [currentStage, setCurrentStage] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  async function handleProcess() {
    if (!text.trim()) return
    setStatus('processing')
    setCurrentStage(1)
    setResult(null)
    setError('')
    setCopied(false)

    const t1 = setTimeout(() => setCurrentStage(2), 2500)
    const t2 = setTimeout(() => setCurrentStage(3), 5500)

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

  async function handleCopy() {
    if (!result) return
    try {
      await navigator.clipboard.writeText(result.output)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback for environments without clipboard API
      const el = document.createElement('textarea')
      el.value = result.output
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  function handleReset() {
    setStatus('idle')
    setCurrentStage(0)
    setResult(null)
    setError('')
    setCopied(false)
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100" style={{ fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace" }}>
      {/* Header */}
      <header className="border-b border-gray-800/60 px-6 py-4 sticky top-0 bg-gray-950/95 backdrop-blur z-10">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded bg-blue-600 flex items-center justify-center text-white text-xs font-bold">CB</div>
            <div>
              <span className="text-base font-bold text-white tracking-tight">
                Context<span className="text-blue-400">Bridge</span>
              </span>
              <span className="text-gray-600 text-xs ml-2 hidden sm:inline">AI conversation continuity</span>
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

        {/* ── INPUT SCREEN ── */}
        {status === 'idle' && (
          <div className="space-y-7">
            <div className="space-y-2">
              <h2 className="text-2xl font-bold text-white">Continue any AI conversation</h2>
              <p className="text-gray-400 text-sm leading-relaxed max-w-2xl">
                Paste an exported AI conversation below. ContextBridge runs a 3-stage pipeline — segmentation, parallel extraction, synthesis — and produces a structured package you can paste into any new AI session to pick up exactly where you left off.
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-gray-500 uppercase tracking-widest">
                Conversation Export
              </label>
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                placeholder={"Paste your exported conversation here...\n\nSupports: plain text, ChatGPT JSON export, markdown export, or any AI platform output.\n\nExample format:\nUser: I want to build a REST API...\nAssistant: Sure, let's start with..."}
                className="w-full h-72 bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500/80 focus:ring-1 focus:ring-blue-500/20 resize-y transition-colors"
              />
              <p className="text-xs text-gray-600">
                {text.length > 0 ? `${text.length.toLocaleString()} characters · ~${Math.round(text.length / 4).toLocaleString()} tokens` : 'No input'}
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-gray-500 uppercase tracking-widest">Output Mode</label>
              <div className="flex gap-2 max-w-xs">
                {[
                  { value: 'detailed', label: 'Detailed', desc: 'Maximum continuity — all technical context' },
                  { value: 'summarized', label: 'Summarized', desc: 'Token-efficient — essential facts only' },
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
              <p className="text-xs text-gray-600">
                {mode === 'detailed'
                  ? 'Preserves all decisions, debug history, code structure, and conversation context'
                  : 'Outputs a compact briefing — architecture, decisions, issues, next steps'}
              </p>
            </div>

            <button
              onClick={handleProcess}
              disabled={!text.trim()}
              className="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-all text-sm tracking-wide shadow-lg shadow-blue-900/20"
            >
              Process Conversation →
            </button>

            {/* Pipeline info box */}
            <div className="border border-gray-800 rounded-xl p-5 bg-gray-900/40 space-y-4">
              <p className="text-xs text-gray-500 uppercase tracking-widest">How it works</p>
              <div className="space-y-3">
                {[
                  { n: 1, calls: '1 call', label: 'Segmentation', model: 'Groq · Mixtral 8x7B', desc: 'Classifies messages, removes filler' },
                  { n: 2, calls: '4 calls', label: 'Extraction', model: 'Groq · Llama 3.3 70B', desc: 'Code · Decisions · Debug · Intent — run in parallel' },
                  { n: 3, calls: '1 call', label: 'Synthesis', model: 'Anthropic · Claude Sonnet', desc: 'Merges extractions into the final package' },
                ].map(s => (
                  <div key={s.n} className="flex items-start gap-3 text-xs">
                    <span className="text-gray-700 w-4 shrink-0 text-right">{s.n}</span>
                    <span className="text-gray-600 w-12 shrink-0">{s.calls}</span>
                    <span className="text-blue-400 w-24 shrink-0 font-medium">{s.label}</span>
                    <span className="text-gray-500 hidden sm:block">{s.model} — {s.desc}</span>
                    <span className="text-gray-500 sm:hidden">{s.model}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-700 pt-1 border-t border-gray-800">
                6 total AI calls · Stages 1+2 use Groq free tier · Stage 3 uses Anthropic API
              </p>
            </div>
          </div>
        )}

        {/* ── PROCESSING SCREEN ── */}
        {status === 'processing' && (
          <div className="flex flex-col items-center justify-center min-h-80 space-y-10">
            <div className="text-center space-y-2">
              <h2 className="text-xl font-semibold text-white">Processing your conversation</h2>
              <p className="text-gray-500 text-sm">Running 6 AI calls across 3 stages</p>
            </div>
            <StageIndicator currentStage={currentStage} />
            <p className="text-xs text-gray-700">Typically completes in 8–15 seconds</p>
          </div>
        )}

        {/* ── ERROR SCREEN ── */}
        {status === 'error' && (
          <div className="flex flex-col items-center justify-center min-h-64 space-y-6">
            <div className="border border-red-800/60 bg-red-950/30 rounded-xl p-6 w-full max-w-lg space-y-3">
              <p className="text-red-400 font-semibold text-sm">Processing failed</p>
              <p className="text-red-300/80 text-sm leading-relaxed">{error}</p>
              <p className="text-gray-600 text-xs">
                Check that your API keys are set in the backend .env file and that both Groq and Anthropic APIs are reachable.
              </p>
            </div>
            <button
              onClick={handleReset}
              className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
            >
              ← Try again
            </button>
          </div>
        )}

        {/* ── OUTPUT SCREEN ── */}
        {status === 'done' && result && (
          <div className="space-y-4">
            {/* Result meta bar */}
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-green-400 text-sm font-medium">✓ Package ready</span>
                <span className="text-xs text-gray-500 border border-gray-800 bg-gray-900 px-2 py-0.5 rounded-md">
                  {result.token_count.toLocaleString()} tokens
                </span>
                <span className="text-xs text-gray-500 border border-gray-800 bg-gray-900 px-2 py-0.5 rounded-md capitalize">
                  {result.mode} mode
                </span>
                <span className="text-xs text-gray-500 border border-gray-800 bg-gray-900 px-2 py-0.5 rounded-md">
                  {(result.processing_time_ms / 1000).toFixed(1)}s
                </span>
              </div>
              <button
                onClick={handleCopy}
                className={`text-sm px-4 py-2 rounded-lg border font-medium transition-all ${
                  copied
                    ? 'bg-green-900/50 border-green-700 text-green-300'
                    : 'bg-gray-900 border-gray-700 text-gray-300 hover:border-blue-500 hover:text-blue-300 hover:bg-blue-950/20'
                }`}
              >
                {copied ? '✓ Copied!' : 'Copy to clipboard'}
              </button>
            </div>

            {/* Output content */}
            <div className="relative">
              <pre className="bg-gray-900 border border-gray-800 rounded-xl p-5 text-xs text-gray-300 leading-relaxed whitespace-pre-wrap overflow-auto max-h-[62vh]">
                {result.output}
              </pre>
            </div>

            <div className="flex items-center justify-between">
              <p className="text-xs text-gray-600">
                Paste the above as your opening message in any new AI session.
              </p>
              <button
                onClick={handleReset}
                className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                Process another →
              </button>
            </div>
          </div>
        )}

      </main>
    </div>
  )
}
