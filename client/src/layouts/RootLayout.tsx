import { Outlet } from "react-router-dom";
import { Navbar } from "../components/PageSelectorNavBar/PageSelectorNavBar";



export default function RootLayout() {
    return (
        <div>
            <Navbar></Navbar>
            <Outlet />
        </div>
    )
}