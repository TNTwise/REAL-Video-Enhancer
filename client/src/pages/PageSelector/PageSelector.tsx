import {PageSelectorNavBar, PageSelectorNavItem} from "../../components/PageSelectorNavBar/PageSelectorNavBar";
import homeIcon from '../../assets/icons/home.svg';
import settingsIcon from '../../assets/icons/settings.svg';
import downloadIcon from '../../assets/icons/download.svg';



export function PageSelector() {
    return (<PageSelectorNavBar >
        <PageSelectorNavItem icon={homeIcon} label="homePage" href="/" active={true} />
        <PageSelectorNavItem icon={settingsIcon} label="settingsPage" href="/settings" active={false} />
        <PageSelectorNavItem icon={downloadIcon} label="downloadPage" href="/settings" active={false} />
      </PageSelectorNavBar>
    );
}