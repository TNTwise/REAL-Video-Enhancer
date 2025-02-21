from ..Util import openLink, networkCheck
import requests


class HomeTab:
    def __init__(self, parent):
        self.parent = parent
        self.QConnect()

    def getChangelog(self):
        changeLog = ""
        try:
            response = requests.get(
                "https://api.github.com/repos/tntwise/real-video-enhancer/releases"
            )
            releases = response.json()
            releaseTags = [release["tag_name"] for release in releases]
            releaseBodies = [release["body"].replace(r"\r\n", "") for release in releases]
            for releaseTag, releaseBody in zip(releaseTags, releaseBodies):
                changeLog += "\n# " + releaseTag
                changeLog += "\n" + releaseBody
        except Exception:
            pass
        return changeLog

    def QConnect(self):
        self.parent.githubBtn.clicked.connect(
            lambda: openLink("https://github.com/tntwise/REAL-Video-Enhancer")
        )
        self.parent.kofiBtn.clicked.connect(
            lambda: openLink("https://ko-fi.com/tntwise")
        )
        if networkCheck(
            "https://api.github.com/repos/tntwise/real-video-enhancer/releases"
        ):
            self.parent.changeLogText.setVisible(True)
            changelog = self.getChangelog()
            self.parent.changeLogText.setMarkdown(changelog)
