from platformdirs import PlatformDirs

APP_NAME = "REALVideoEnhancer"
COMPANY = "TNTWISE"
dirs = PlatformDirs(APP_NAME, COMPANY)

CONFIG_PATH = dirs.user_config_path
DEFAULT_VIDEOS_PATH = dirs.user_videos_path

DEFAULT_VIDEOS_PATH.mkdir(exist_ok=True)
CONFIG_PATH.mkdir(exist_ok=True)