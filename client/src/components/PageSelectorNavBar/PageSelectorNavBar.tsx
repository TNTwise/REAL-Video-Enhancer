import { type ComponentType, type SVGProps, ReactElement, JSXElementConstructor, ReactNode, ReactPortal } from "react";
import "./PageSelectorNavBar.css";

export function PageSelectorNavBar(props: { children: string | number | bigint | boolean | ReactElement<unknown, string | JSXElementConstructor<any>> | Iterable<ReactNode> | ReactPortal | Promise<string | number | bigint | boolean | ReactPortal | ReactElement<unknown, string | JSXElementConstructor<any>> | Iterable<ReactNode> | null | undefined> | null | undefined; }) {
    return (
        <nav className="navbar">
            <ul className="navbar-nav">
                {props.children}
            </ul>
        </nav>
    );
}



export function PageSelectorNavItem(props: { 
  icon: string;
  label: string; 
  href: string; 
  active: boolean 
}) {
  
  return (
    <li className={`nav-item ${props.active ? 'active' : ''}`}>
      <a href={props.href} className="icon-button">
        <img src={props.icon} />
      </a>
    </li>
  );
}