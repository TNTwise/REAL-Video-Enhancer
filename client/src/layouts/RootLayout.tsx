import { Outlet } from "react-router-dom";
import { Navbar } from "../components/PageSelectorNavBar/PageSelectorNavBar";
import homeIcon from '/src/assets/icons/home.svg';
import settingsIcon from '/src/assets/icons/settings.svg';
import downloadIcon from '/src/assets/icons/download.svg';


export default function RootLayout() {
    return (
        <div>
            <Navbar></Navbar>
            <Outlet />
        </div>
    )
}