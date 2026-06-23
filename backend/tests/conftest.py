from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_settings_proxy():
    mock = MagicMock()
    mock.pytorch_version = "2.12.1"
    mock.torch_accelerator = "cu132"
    mock.torchvision_version = "0.27.1"
    mock.tensorrt_version = "10.16.1.11"
    return mock


@pytest.fixture
def service(mock_settings_proxy):
    from src.logic.services.install_packages_service import InstallPackagesService

    return InstallPackagesService(settings_proxy=mock_settings_proxy)
