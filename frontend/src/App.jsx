import { useState, useRef, useCallback } from 'react'

// ── Constants ─────────────────────────────────────────────────────────────────

const PIPELINE_STAGES = [
  { id: 1, label: 'Chunking',               model: 'rule-based' },
  { id: 2, label: 'Classify + Memory loop', model: 'Groq · Llama / Instant' },
  { id: 3, label: 'Output generation',      model: 'Groq · Llama 3.3 70B' },
]

const OUTPUT_TABS = [
  { id: 'intent',      label: 'Intent & Context',  filename: '01_project_intent.md',      color: 'blue',    detailedOnly: false },
  { id: 'code',        label: 'Code Understanding', filename: '02_code_understanding.md',  color: 'purple',  detailedOnly: false },
  { id: 'structure',   label: 'Project Structure',  filename: '03_project_structure.md',   color: 'emerald', detailedOnly: false },
  { id: 'code_detail', label: 'Detailed Code',      filename: '04_detailed_code.md',       color: 'amber',   detailedOnly: true  },
]

const TAB_COLORS = {
  blue:    { active: 'border-blue-500 text-blue-300 bg-blue-950/30',        inactive: 'border-transparent text-gray-500 hover:text-gray-300' },
  purple:  { active: 'border-purple-500 text-purple-300 bg-purple-950/30',  inactive: 'border-transparent text-gray-500 hover:text-gray-300' },
  emerald: { active: 'border-emerald-500 text-emerald-300 bg-emerald-950/30', inactive: 'border-transparent text-gray-500 hover:text-gray-300' },
  amber:   { active: 'border-amber-500 text-amber-300 bg-amber-950/30',     inactive: 'border-transparent text-gray-500 hover:text-gray-300' },
}

const ACCEPTED_TYPES = { 'text/plain': '.txt', 'text/markdown': '.md', 'application/json': '.json', 'text/html': '.html' }
const ACCEPTED_EXT   = ['.txt', '.md', '.json', '.html']

// ── Download helpers ──────────────────────────────────────────────────────────

function downloadMd(filename, content) {
  const blob = new Blob([content], { type: 'text/markdown' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

async function downloadAllZip(outputFiles) {
  const { default: JSZip } = await import('jszip')
  const zip = new JSZip()
  OUTPUT_TABS.forEach(({ id, filename }) => {
    if (outputFiles[id]) zip.file(filename, outputFiles[id])
  })
  const blob = await zip.generateAsync({ type: 'blob' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = 'contextbridge_output.zip'; a.click()
  URL.revokeObjectURL(url)
}

// ── FileUploadZone ────────────────────────────────────────────────────────────

function FileUploadZone({ uploadedFile, onFile, onClear }) {
  const inputRef   = useRef(null)
  const [dragging, setDragging] = useState(false)

  const readFile = useCallback((file) => {
    if (!file) return
    const ext  = '.' + file.name.split('.').pop().toLowerCase()
    if (!ACCEPTED_EXT.includes(ext)) return
    const type = ext === '.json' ? 'json' : ext === '.html' ? 'html' : 'txt'
    const reader = new FileReader()
    reader.onload = (e) => onFile({ name: file.name, content: e.target.result, type })
    reader.readAsText(file)
  }, [onFile])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    readFile(e.dataTransfer.files[0])
  }, [readFile])

  if (uploadedFile) {
    return (
      <div className="flex items-center gap-3 px-4 py-3 bg-gray-900 border border-emerald-800/60 rounded-xl">
        <span className="text-emerald-400 text-sm">📄</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-emerald-300 truncate">{uploadedFile.name}</p>
          <p className="text-xs text-gray-600">{uploadedFile.type} file · will be combined with pasted text if both are provided</p>
        </div>
        <button onClick={onClear} className="text-xs text-gray-500 hover:text-red-400 transition-colors px-2 py-1 border border-gray-700 rounded">
          Remove
        </button>
      </div>
    )
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`flex flex-col items-center justify-center gap-2 px-4 py-5 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
        dragging ? 'border-blue-500 bg-blue-950/20' : 'border-gray-700 hover:border-gray-500 bg-gray-900/30'
      }`}
    >
      <span className="text-2xl text-gray-600">↑</span>
      <p className="text-sm text-gray-400">Drop a file or <span className="text-blue-400">click to browse</span></p>
      <p className="text-xs text-gray-600">.txt · .md · .json (ChatGPT export) · .html</p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXT.join(',')}
        className="hidden"
        onChange={(e) => readFile(e.target.files[0])}
      />
    </div>
  )
}

// ── StageIndicator ────────────────────────────────────────────────────────────

function StageIndicator({ currentStage }) {
  return (
    <div className="w-full max-w-md space-y-3">
      {PIPELINE_STAGES.map(stage => {
        const isDone   = currentStage > stage.id
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
            <div className="w-6 h-6 shrink-0 flex items-center justify-center">
              {isDone    ? <span className="text-green-400">✓</span>
              : isActive ? <span className="text-blue-400 animate-spin inline-block">◌</span>
              :             <span className="text-gray-700">○</span>}
            </div>
            <div className="min-w-0">
              <p className={`text-sm font-medium ${isActive ? 'text-white' : isDone ? 'text-green-400' : 'text-gray-600'}`}>
                Stage {stage.id} — {stage.label}
              </p>
              <p className="text-xs text-gray-600">{stage.model}</p>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── OutputTabs ────────────────────────────────────────────────────────────────

function OutputTabs({ result, outputFiles }) {
  const isDetailed  = result.mode === 'detailed'
  const visibleTabs = OUTPUT_TABS.filter(t => !t.detailedOnly || isDetailed)
  const [activeTab, setActiveTab]   = useState('intent')
  const [copied, setCopied]         = useState(null)
  const [downloading, setDownloading] = useState(false)

  async function handleCopy(tabId, content) {
    try { await navigator.clipboard.writeText(content) }
    catch {
      const el = document.createElement('textarea')
      el.value = content; document.body.appendChild(el); el.select()
      document.execCommand('copy'); document.body.removeChild(el)
    }
    setCopied(tabId)
    setTimeout(() => setCopied(null), 2000)
  }

  async function handleDownloadAll() {
    setDownloading(true)
    try { await downloadAllZip(outputFiles) }
    finally { setDownloading(false) }
  }

  const currentTab    = visibleTabs.find(t => t.id === activeTab)
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
        <button
          onClick={handleDownloadAll}
          disabled={downloading}
          className="text-sm px-4 py-2 rounded-lg border border-gray-700 text-gray-300 hover:border-blue-500 hover:text-blue-300 transition-all disabled:opacity-50"
        >
          {downloading ? 'Zipping…' : '↓ Download all (.zip)'}
        </button>
      </div>

      {/* Tab row */}
      <div className="flex gap-1 border-b border-gray-800 flex-wrap">
        {visibleTabs.map(tab => {
          const colors   = TAB_COLORS[tab.color]
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

      {/* Content area */}
      <div className="relative">
        {/* Toolbar inside the box */}
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
          <span className="text-xs text-gray-700 hidden sm:block">{currentTab?.filename}</span>
          <button
            onClick={() => downloadMd(currentTab?.filename, currentContent)}
            className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-400 hover:border-gray-500 hover:text-gray-200 transition-all"
            title="Download this file"
          >
            ↓ .md
          </button>
          <button
            onClick={() => handleCopy(activeTab, currentContent)}
            className={`text-xs px-3 py-1 rounded border transition-all ${
              copied === activeTab
                ? 'bg-green-900/50 border-green-700 text-green-300'
                : 'bg-gray-900/80 border-gray-700 text-gray-300 hover:border-blue-500 hover:text-blue-300'
            }`}
          >
            {copied === activeTab ? '✓ Copied' : 'Copy'}
          </button>
        </div>

        <pre className="bg-gray-900 border border-gray-800 rounded-xl p-5 pt-11 text-xs text-gray-300 leading-relaxed whitespace-pre-wrap overflow-auto max-h-[60vh]">
          {currentContent || '(no content for this section)'}
        </pre>
      </div>

      <p className="text-xs text-gray-600 text-center">
        Paste any file into a new AI session as the opening message — each is self-contained.
      </p>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [text, setText]               = useState('')
  const [uploadedFile, setUploadedFile] = useState(null)
  const [mode, setMode]               = useState('detailed')
  const [chunkLevel, setChunkLevel]   = useState('medium')
  const [status, setStatus]           = useState('idle')
  const [currentStage, setCurrentStage] = useState(0)
  const [result, setResult]           = useState(null)
  const [error, setError]             = useState('')

  const hasInput = text.trim() || uploadedFile

  async function handleProcess() {
    if (!hasInput) return
    setStatus('processing'); setCurrentStage(1)
    setResult(null); setError('')

    const t1 = setTimeout(() => setCurrentStage(2), 1500)
    const t2 = setTimeout(() => setCurrentStage(3), 8000)

    try {
      const body = {
        conversation_text: text,
        mode,
        chunk_level: chunkLevel,
        token_target: null,
      }
      if (uploadedFile) {
        body.file_content = uploadedFile.content
        body.file_type    = uploadedFile.type
      }

      const resp = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      clearTimeout(t1); clearTimeout(t2); setCurrentStage(3)

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }

      setResult(await resp.json())
      setStatus('done')
    } catch (err) {
      clearTimeout(t1); clearTimeout(t2)
      setError(err.message); setStatus('error')
    }
  }

  function handleReset() {
    setStatus('idle'); setCurrentStage(0)
    setResult(null);   setError('')
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

        {/* ── INPUT ─────────────────────────────────────────────────────────── */}
        {status === 'idle' && (
          <div className="space-y-6">
            <div className="space-y-1">
              <h2 className="text-2xl font-bold text-white">Continue any AI conversation</h2>
              <p className="text-gray-400 text-sm leading-relaxed max-w-2xl">
                Paste text, upload an export file, or both. ContextBridge reads your
                conversation chunk by chunk, builds structured memory, and outputs ready-to-paste
                documents for a new AI session.
              </p>
            </div>

            {/* Text paste */}
            <div className="space-y-2">
              <label className="text-xs text-gray-500 uppercase tracking-widest">
                Paste conversation text
              </label>
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                placeholder={"Paste exported conversation text here…\n\nSupports any AI platform: plain text, ChatGPT JSON, markdown, HTML export."}
                className="w-full h-52 bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500/80 focus:ring-1 focus:ring-blue-500/20 resize-y transition-colors"
              />
              <p className="text-xs text-gray-600">
                {text.length > 0 ? `${text.length.toLocaleString()} chars · ~${Math.round(text.length / 4).toLocaleString()} tokens` : 'No text input'}
              </p>
            </div>

            {/* File upload */}
            <div className="space-y-2">
              <label className="text-xs text-gray-500 uppercase tracking-widest">
                Or upload an export file
                {text.trim() && uploadedFile && (
                  <span className="ml-2 text-emerald-500 normal-case">· both sources will be combined</span>
                )}
              </label>
              <FileUploadZone
                uploadedFile={uploadedFile}
                onFile={setUploadedFile}
                onClear={() => setUploadedFile(null)}
              />
            </div>

            {/* Mode selector */}
            <div className="space-y-2">
              <label className="text-xs text-gray-500 uppercase tracking-widest">Output mode</label>
              <div className="flex gap-2 max-w-xs">
                {[
                  { value: 'detailed',   label: 'Detailed',   desc: '4 files — includes actual code blocks' },
                  { value: 'summarized', label: 'Summarized', desc: '3 files — no code, faster' },
                ].map(m => (
                  <button
                    key={m.value}
                    onClick={() => setMode(m.value)}
                    className={`flex-1 py-2 px-4 rounded-lg text-sm border font-medium transition-all ${
                      mode === m.value
                        ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/30'
                        : 'bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
              <p className="text-xs text-gray-600">
                {mode === 'detailed'
                  ? 'Captures actual code blocks shown in the conversation alongside semantic descriptions'
                  : 'Faster — produces intent, code semantics, and structure documents only'}
              </p>
            </div>

            {/* Chunk level selector */}
            <div className="space-y-2">
              <label className="text-xs text-gray-500 uppercase tracking-widest">
                Chunking level
              </label>
              <div className="flex gap-2">
                {[
                  {
                    value: 'low',
                    label: 'Low',
                    badge: 'Fast',
                    desc: 'Fewer, larger chunks — quick processing, less granular',
                    badgeColor: 'text-emerald-400',
                  },
                  {
                    value: 'medium',
                    label: 'Medium',
                    badge: 'Balanced',
                    desc: 'Default — good balance of speed and detail',
                    badgeColor: 'text-blue-400',
                  },
                  {
                    value: 'high',
                    label: 'High',
                    badge: 'Thorough',
                    desc: 'Many small chunks — most detail, slowest',
                    badgeColor: 'text-amber-400',
                  },
                ].map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setChunkLevel(opt.value)}
                    className={`flex-1 py-2 px-3 rounded-lg text-sm border transition-all text-left ${
                      chunkLevel === opt.value
                        ? 'border-gray-500 bg-gray-800 text-white'
                        : 'border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600'
                    }`}
                  >
                    <span className="font-medium block">{opt.label}</span>
                    <span className={`text-xs ${opt.badgeColor}`}>{opt.badge}</span>
                  </button>
                ))}
              </div>
              <p className="text-xs text-gray-600">
                {
                  { low:    'Fewer, larger chunks — quick processing, best for short conversations',
                    medium: 'Balanced — recommended for most conversations',
                    high:   'Many small chunks — most thorough, significantly slower on long conversations',
                  }[chunkLevel]
                }
              </p>
            </div>

            <button
              onClick={handleProcess}
              disabled={!hasInput}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-all text-sm tracking-wide"
            >
              Process Conversation →
            </button>

            {/* Pipeline info */}
            <div className="border border-gray-800 rounded-xl p-5 bg-gray-900/40 space-y-3">
              <p className="text-xs text-gray-500 uppercase tracking-widest">Memory pipeline</p>
              <div className="space-y-2 text-xs">
                {[
                  { n: '1', label: 'Chunker',           desc: 'Token-bounded semantic chunks' },
                  { n: '2', label: 'Classifier',        desc: 'Labels each chunk: code_update · architecture · debugging · …' },
                  { n: '3', label: 'Memory Dispatcher', desc: 'Patches Intent / Code / Structure / Code-Detail memory (JSON)' },
                  { n: '4', label: 'Loop',              desc: 'Repeats 2–3 for every chunk; memory auto-summarises when large' },
                  { n: '5', label: 'Output Generator',  desc: 'Formats 3–4 markdown documents from the final memory state' },
                ].map(s => (
                  <div key={s.n} className="flex gap-3">
                    <span className="text-gray-700 w-4 text-right shrink-0">{s.n}</span>
                    <span className="text-blue-400 w-36 shrink-0 font-medium">{s.label}</span>
                    <span className="text-gray-500 hidden sm:block">{s.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── PROCESSING ────────────────────────────────────────────────────── */}
        {status === 'processing' && (
          <div className="flex flex-col items-center justify-center min-h-80 space-y-10">
            <div className="text-center space-y-1">
              <h2 className="text-xl font-semibold text-white">Building memory documents</h2>
              <p className="text-gray-500 text-sm">Processing conversation chunk by chunk</p>
            </div>
            <StageIndicator currentStage={currentStage} />
            <p className="text-xs text-gray-700">Typically 30 s – 2 min depending on conversation length</p>
          </div>
        )}

        {/* ── ERROR ─────────────────────────────────────────────────────────── */}
        {status === 'error' && (
          <div className="flex flex-col items-center justify-center min-h-64 space-y-6">
            <div className="border border-red-800/60 bg-red-950/30 rounded-xl p-6 w-full max-w-lg space-y-3">
              <p className="text-red-400 font-semibold text-sm">Processing failed</p>
              <p className="text-red-300/80 text-sm leading-relaxed">{error}</p>
            </div>
            <button onClick={handleReset} className="text-sm text-blue-400 hover:text-blue-300">← Try again</button>
          </div>
        )}

        {/* ── OUTPUT ────────────────────────────────────────────────────────── */}
        {status === 'done' && result && (
          <OutputTabs result={result} outputFiles={result.output_files || {}} />
        )}

      </main>
    </div>
  )
}
