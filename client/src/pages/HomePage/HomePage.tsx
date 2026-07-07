import { useEffect, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { apiFetch } from "@/lib/api"
import { Cpu, HardDrive, Monitor, CuboidIcon as CubeIcon } from "lucide-react"
import RVELogo from "/src/assets/logo-v2.svg"

interface GPUInfo {
  name: string
  vendor: string | null
  memory_mb: number | null
  driver_version: string | null
  device_id: string | null
}

interface SystemInfo {
  python_version: string
  app_version: string
  opencv_version: string | null
  pytorch_version: string | null
  cuda_version: string | null
  torch_accelerator: string
  total_memory_gb: number | null
  gpus: GPUInfo[]
}

const appVersion = import.meta.env.VITE_APP_VERSION ?? "0.1.0"

function gpuLabel(sysInfo: SystemInfo | null): string {
  if (!sysInfo?.gpus.length) return sysInfo?.torch_accelerator ?? "..."
  const gpu = sysInfo.gpus[0]
  const mem = gpu.memory_mb ? `${Math.round(gpu.memory_mb / 1024)} GB` : ""
  return `${gpu.name}${mem ? ` (${mem})` : ""}`
}

function InfoCard({ label, value, icon: Icon }: { label: string; value: string; icon: React.ElementType }) {
  return (
    <div className="flex items-center gap-3 rounded-lg bg-[#2c313c] px-3 py-2.5">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-[#343b47]">
        <Icon className="h-4 w-4 text-[#838ea2]" />
      </div>
      <div className="min-w-0">
        <p className="text-[11px] text-[#838ea2]">{label}</p>
        <p className="text-sm font-semibold text-white truncate">{value}</p>
      </div>
    </div>
  )
}

export function HomePage() {
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null)

  useEffect(() => {
    apiFetch<SystemInfo>("/system/info")
      .then(setSysInfo)
      .catch(() => {})
  }, [])

  const softwareInfo = [
    { label: "Python", value: sysInfo?.python_version ?? "...", icon: CubeIcon },
    { label: "OpenCV", value: sysInfo?.opencv_version ?? "Not installed", icon: CubeIcon },
    { label: "PyTorch", value: sysInfo?.pytorch_version ?? "Not installed", icon: CubeIcon },
    { label: "CUDA", value: sysInfo?.cuda_version ?? sysInfo?.torch_accelerator ?? "...", icon: CubeIcon },
  ]

  const systemInfo = [
    { label: "OS", value: navigator.platform, icon: Monitor },
    { label: "CPU", value: `${navigator.hardwareConcurrency ?? 0} cores`, icon: Cpu },
    { label: "Memory", value: sysInfo?.total_memory_gb ? `${sysInfo.total_memory_gb} GB` : "...", icon: HardDrive },
    { label: "GPU", value: gpuLabel(sysInfo), icon: Monitor },
  ]

  return (
    <div className="flex h-full items-center justify-center">
      <div className="flex w-full max-w-2xl flex-col items-center gap-5">
        {/* Header */}
        <div className="flex flex-col items-center text-center">
          <div className="mb-3 flex h-20 w-20 items-center justify-center rounded-2xl bg-[#1f232a] border border-[#343b47]">
            <img src={RVELogo} alt="RVE Logo" className="h-full w-full p-3" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            REAL Video Enhancer
          </h1>
          <div className="mt-1 flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              v{appVersion}
            </Badge>
            <Badge variant="outline" className="text-xs text-[#838ea2] border-[#343b47]">
              Beta
            </Badge>
          </div>
        </div>

        {/* Cards side by side */}
        <div className="flex w-full gap-4">
          {/* Software Information */}
          <Card className="flex-1 bg-[#1f232a] border-[#343b47]">
            <CardHeader className="pb-2 pt-4 px-4">
              <CardTitle className="text-sm text-white">Software</CardTitle>
              <CardDescription className="text-xs text-[#838ea2]">
                Runtime environment
              </CardDescription>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <div className="flex flex-col gap-2">
                {softwareInfo.map((info) => (
                  <InfoCard key={info.label} {...info} />
                ))}
              </div>
            </CardContent>
          </Card>

          {/* System Information */}
          <Card className="flex-1 bg-[#1f232a] border-[#343b47]">
            <CardHeader className="pb-2 pt-4 px-4">
              <CardTitle className="text-sm text-white">System</CardTitle>
              <CardDescription className="text-xs text-[#838ea2]">
                Machine specifications
              </CardDescription>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <div className="flex flex-col gap-2">
                {systemInfo.map((info) => (
                  <InfoCard key={info.label} {...info} />
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-[#838ea2]">
          Built with PyTorch, OpenCV, and Tauri
        </p>
      </div>
    </div>
  )
}
