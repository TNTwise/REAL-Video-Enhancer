import {PageSelectorNavBar, PageSelectorNavItem} from "../../components/PageSelectorNavBar/PageSelectorNavBar";
import home from '../../assets/icons/home.svg'

export function PageSelector() {
    return (<PageSelectorNavBar >
        <PageSelectorNavItem icon={home} />
        <PageSelectorNavItem icon="s" />
        <PageSelectorNavItem icon="s" />
      </PageSelectorNavBar>
    );
}