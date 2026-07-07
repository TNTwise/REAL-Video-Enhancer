import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Cpu, HardDrive, Languages, Monitor, CuboidIcon as CubeIcon } from "lucide-react"
import RVELogo from "/src/assets/logo-v2.svg"

const appVersion = import.meta.env.VITE_APP_VERSION ?? "0.1.0"

const systemInfo = [
  { label: "OS", value: navigator.platform, icon: Monitor },
  { label: "CPU", value: `${navigator.hardwareConcurrency ?? 0} cores`, icon: Cpu },
  { label: "Memory", value: `${(navigator as any).deviceMemory ?? 8} GB`, icon: HardDrive },
  { label: "Language", value: navigator.language, icon: Languages },
]

const softwareInfo = [
  { label: "Python", value: "3.12", icon: CubeIcon },
  { label: "OpenCV", value: "4.10.0", icon: CubeIcon },
  { label: "PyTorch", value: "2.5.1+cu124", icon: CubeIcon },
  { label: "CUDA", value: "Checking...", icon: CubeIcon },
]

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
