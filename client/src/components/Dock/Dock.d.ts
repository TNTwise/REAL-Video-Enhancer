import type { ReactNode, ComponentPropsWithoutRef } from 'react';

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
  panelHeight?: number;
  baseItemSize?: number;
  dockHeight?: number;
  magnification?: number;
  spring?: { mass: number; stiffness: number; damping: number };
}

export default function Dock(props: DockProps): JSX.Element;
