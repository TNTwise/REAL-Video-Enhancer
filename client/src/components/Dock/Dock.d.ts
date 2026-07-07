import type { ReactNode } from 'react';

export interface DockItemData {
  icon: ReactNode;
  label: string;
  onClick?: () => void;
  className?: string;
}

export interface DockProps {
  items: DockItemData[];
  className?: string;
  distance?: number;
  panelWidth?: number;
  baseItemSize?: number;
  magnification?: number;
  spring?: { mass: number; stiffness: number; damping: number };
  activeIndex?: number;
}

export default function Dock(props: DockProps): JSX.Element;
