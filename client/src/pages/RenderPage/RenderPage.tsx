import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { File, FolderOpen, Maximize2, Play } from "lucide-react"
import { Select } from "@/components/ui/select"
import { apiFetch, getBackendUrl } from "@/lib/api"

interface BackendInfo {
  type: string
  installed: boolean
  version: string
}

interface AvailableBackends {
  pytorch: BackendInfo & { accelerator: string }
  ncnn: BackendInfo
  tensorrt: BackendInfo
}

interface ModelInfo {
  id: string
  type: string
  variant: string
  description?: string | null
  backend: { type: string }
}

function deriveOutputPath(inputPath: string): string {
  if (!inputPath) return ""
  const dot = inputPath.lastIndexOf(".")
  if (dot === -1) return `${inputPath}_enhanced`
  return `${inputPath.slice(0, dot)}_enhanced${inputPath.slice(dot)}`
}

export function RenderPage() {
  const [backendUrl, setBackendUrl] = useState("")
  const [backends, setBackends] = useState<AvailableBackends | null>(null)
  const [selectedBackend, setSelectedBackend] = useState("")
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState("")
  const [imgTs, setImgTs] = useState(0)
  const [imgError, setImgError] = useState(false)
  const [rendering, setRendering] = useState(false)
  const [inputFile, setInputFile] = useState("")
  const [outputFile, setOutputFile] = useState("")
  const previewRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getBackendUrl().then(setBackendUrl)
  }, [])

  useEffect(() => {
    if (!backendUrl) return
    fetch(`${backendUrl}/backends/available`)
      .then((r) => r.json())
      .then(setBackends)
      .catch(() => {})
  }, [backendUrl])

  useEffect(() => {
    if (!selectedBackend || !backendUrl) return
    setSelectedModel("")
    fetch(`${backendUrl}/models/interpolate?backend_id=${selectedBackend}`)
      .then((r) => r.json())
      .then((data) => {
        setModels(Array.isArray(data) ? data : [])
      })
      .catch(() => setModels([]))
  }, [selectedBackend, backendUrl])

  useEffect(() => {
    if (!backendUrl) return
    const interval = setInterval(() => setImgTs(Date.now()), 1000)
    return () => clearInterval(interval)
  }, [backendUrl])

  const handleInputFileChange = useCallback((path: string) => {
    setInputFile(path)
    if (!path) {
      setOutputFile("")
      return
    }
    setOutputFile((prev) => (prev && prev.startsWith(path.slice(0, path.lastIndexOf("/")) + "/") ? prev : deriveOutputPath(path)))
  }, [])

  const handleStartRender = useCallback(async () => {
    const model = models.find((m) => m.id === selectedModel)
    if (!model) return

    setRendering(true)
    try {
      await apiFetch("/render/start_render", {
        method: "POST",
        body: JSON.stringify({
          input_video_info: { input_file: inputFile },
          output_video_info: { output_file: outputFile },
          benchmark_mode: false,
          slow_mo_mode: false,
          interpolate_model: {
            type: "interpolate",
            id: model.id,
            backed_type: model.backend.type,
            interpolate_factor: 2,
          },
          upscale_model: null,
          enhancement_models: null,
          start_time: null,
          end_time: null,
        }),
      })
    } catch (e) {
      console.error(e)
    } finally {
      setRendering(false)
    }
  }, [models, selectedModel, inputFile, outputFile])

  const installedBackends = !backends
    ? []
    : [
        ...(backends.pytorch.installed
          ? [{ value: "pytorch", label: `PyTorch (${backends.pytorch.accelerator})` }]
          : []),
        ...(backends.ncnn.installed ? [{ value: "ncnn", label: "NCNN" }] : []),
        ...(backends.tensorrt.installed ? [{ value: "tensorrt", label: "TensorRT" }] : []),
      ]

  const modelOptions = models.map((m) => ({
    value: m.id,
    label: m.description ? `${m.variant} — ${m.description}` : m.variant,
  }))

  const canStart = useMemo(
    () => Boolean(selectedModel && inputFile && outputFile && !rendering),
    [selectedModel, inputFile, outputFile, rendering],
  )

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <div
        ref={previewRef}
        className="relative flex flex-1 items-center justify-center overflow-hidden rounded-lg border border-[#343b47] bg-[#1f232a]"
      >
        {backendUrl && !imgError ? (
          <img
            src={`${backendUrl}/render/latest_image?t=${imgTs}`}
            alt="Render preview"
            className="max-h-full max-w-full object-contain"
            onError={() => setImgError(true)}
            onLoad={() => setImgError(false)}
          />
        ) : (
          <p className="text-sm text-[#838ea2]">No render active</p>
        )}

        <button
          onClick={() => previewRef.current?.requestFullscreen()}
          className="absolute bottom-3 right-3 flex items-center gap-1.5 rounded-md bg-[#2c313c] px-2.5 py-1.5 text-xs text-[#838ea2] transition-colors hover:text-white"
        >
          <Maximize2 size={14} />
          Fullscreen
        </button>
      </div>

      <div className="flex flex-wrap gap-4">
        <div className="flex flex-1 flex-col gap-1.5">
          <label className="text-xs text-[#838ea2]">Input file</label>
          <div className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={inputFile}
              onChange={(e) => handleInputFileChange(e.target.value)}
              placeholder="/path/to/video.mp4"
              className="flex h-9 flex-1 rounded-md border border-[#343b47] bg-[#1f232a] px-3 py-1 text-sm text-white placeholder:text-[#838ea2] focus:outline-none focus:ring-1 focus:ring-[#838ea2]"
            />
            <button
              onClick={() => inputRef.current?.click()}
              className="flex h-9 items-center gap-1.5 rounded-md bg-[#2c313c] px-3 text-sm text-[#838ea2] transition-colors hover:text-white"
            >
              <FolderOpen size={14} />
              Browse
            </button>
          </div>
        </div>

        <div className="flex flex-1 flex-col gap-1.5">
          <label className="text-xs text-[#838ea2]">Output file</label>
          <div className="relative">
            <File
              size={14}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#838ea2]"
            />
            <input
              type="text"
              value={outputFile}
              onChange={(e) => setOutputFile(e.target.value)}
              placeholder="Auto-generated from input"
              className="flex h-9 w-full rounded-md border border-[#343b47] bg-[#1f232a] py-1 pl-8 pr-3 text-sm text-white placeholder:text-[#838ea2] focus:outline-none focus:ring-1 focus:ring-[#838ea2]"
            />
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1.5">
          <label className="text-xs text-[#838ea2]">Backend</label>
          <Select
            options={installedBackends}
            placeholder="Select backend..."
            value={selectedBackend}
            onChange={(e) => setSelectedBackend(e.target.value)}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs text-[#838ea2]">Model</label>
          <Select
            options={modelOptions}
            placeholder="Select model..."
            value={selectedModel}
            disabled={!selectedBackend}
            onChange={(e) => setSelectedModel(e.target.value)}
          />
        </div>

        <button
          onClick={handleStartRender}
          disabled={!canStart}
          className="flex h-9 items-center gap-1.5 rounded-md bg-[#2c313c] px-4 text-sm text-white transition-colors hover:bg-[#343b47] disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Play size={14} />
          {rendering ? "Starting..." : "Start Render"}
        </button>
      </div>
    </div>
  )
}
