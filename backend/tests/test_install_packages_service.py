import asyncio
import sys
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def _mock_process(returncode: int = 0, stdout_lines: list[bytes] | None = None):
    process = AsyncMock()
    process.returncode = returncode
    process.stdout.readline = AsyncMock(side_effect=stdout_lines or [b""])
    process.wait = AsyncMock()
    return process


class TestInstallPackages:
    """Tests for InstallPackagesService."""

    async def test_install_packages_constructs_correct_command(self, service):
        process = _mock_process()
        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as mock_exec:
            await service._install_packages(["foo", "bar"])

        mock_exec.assert_called_once()
        args, kwargs = mock_exec.call_args
        cmd = list(args)

        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "pip"
        assert cmd[3] == "install"
        assert "--isolated" in cmd
        assert "--extra-index-url" in cmd
        assert "https://download.pytorch.org/whl/" in cmd
        assert "--trusted-host" in cmd
        assert "download.pytorch.org" in cmd
        assert cmd[-2:] == ["foo", "bar"]
        assert kwargs.get("stdout") is asyncio.subprocess.PIPE
        assert kwargs.get("stderr") is asyncio.subprocess.STDOUT

    async def test_install_packages_appends_upgrade_flag(self, service):
        process = _mock_process()
        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as mock_exec:
            await service._install_packages(["foo"], upgrade=True)

        cmd = list(mock_exec.call_args[0])
        assert "--upgrade" in cmd
        assert cmd[-1] == "foo"

    async def test_install_packages_without_upgrade_flag(self, service):
        process = _mock_process()
        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as mock_exec:
            await service._install_packages(["foo"], upgrade=False)

        cmd = list(mock_exec.call_args[0])
        assert "--upgrade" not in cmd

    async def test_install_packages_appends_multiple_packages(self, service):
        process = _mock_process()
        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as mock_exec:
            await service._install_packages(["a", "b", "c"])

        cmd = list(mock_exec.call_args[0])
        assert cmd[-3:] == ["a", "b", "c"]

    async def test_install_packages_logs_info_on_success(self, service, caplog):
        caplog.set_level("INFO")
        process = _mock_process(returncode=0)
        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ):
            await service._install_packages(["foo"])

        assert "Successfully installed: foo" in caplog.text

    async def test_install_packages_logs_error_on_failure(self, service, caplog):
        caplog.set_level("ERROR")
        process = _mock_process(returncode=1, stdout_lines=[b"some error\n", b""])
        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ):
            await service._install_packages(["foo"])

        assert "Installation failed:\nsome error" in caplog.text

    async def test_install_base_packages_calls_install_with_correct_packages(
        self, service
    ):
        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_base_packages()

        mock_install.assert_called_once_with(
            ["requests", "opencv-python-headless", "numpy", "typing_extensioons"]
        )

    async def test_install_torch_packages_calls_install_with_correct_packages(
        self, service, mock_settings_proxy
    ):
        mock_settings_proxy.pytorch_version = "2.0.0"
        mock_settings_proxy.torch_accelerator = "cu118"
        mock_settings_proxy.torchvision_version = "0.15.0"

        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_torch_packages()

        mock_install.assert_called_once_with(
            ["torch==2.0.0+cu118", "torchvision==0.15.0+cu118"],
            upgrade=False,
        )

    async def test_install_ncnn_packages_calls_install_with_correct_packages(
        self, service
    ):
        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_ncnn_packages()

        mock_install.assert_called_once_with(
            ["ncnn", "rife-ncnn-vulkan-python-tntwise", "upscale_ncnn_py"],
            upgrade=False,
        )

    async def test_install_tensorrt_packages_calls_install_with_correct_packages(
        self, service, mock_settings_proxy
    ):
        mock_settings_proxy.tensorrt_version = "8.6.0"
        mock_settings_proxy.pytorch_version = "2.0.0"

        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_tensorrt_packages()

        mock_install.assert_called_once_with(
            ["tensorrt==8.6.0", "torch_tensorrt==2.0.0"],
            upgrade=False,
        )

    async def test_install_torch_packages_uses_settings(
        self, service, mock_settings_proxy
    ):
        mock_settings_proxy.pytorch_version = "1.13.0"
        mock_settings_proxy.torch_accelerator = "cu117"
        mock_settings_proxy.torchvision_version = "0.14.0"

        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_torch_packages()

        mock_install.assert_called_once_with(
            ["torch==1.13.0+cu117", "torchvision==0.14.0+cu117"],
            upgrade=False,
        )

    async def test_install_tensorrt_packages_uses_settings(
        self, service, mock_settings_proxy
    ):
        mock_settings_proxy.tensorrt_version = "8.5.3"
        mock_settings_proxy.pytorch_version = "1.13.0"

        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_tensorrt_packages()

        mock_install.assert_called_once_with(
            ["tensorrt==8.5.3", "torch_tensorrt==1.13.0"],
            upgrade=False,
        )

    async def test_install_torch_packages_no_upgrade_by_default(self, service):
        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_torch_packages()

        call_kwargs = mock_install.call_args[1]
        assert call_kwargs.get("upgrade", False) is False

    async def test_install_base_packages_no_upgrade_by_default(self, service):
        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_base_packages()

        call_kwargs = mock_install.call_args[1]
        assert call_kwargs.get("upgrade", False) is False

    async def test_install_ncnn_packages_no_upgrade_by_default(self, service):
        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_ncnn_packages()

        call_kwargs = mock_install.call_args[1]
        assert call_kwargs.get("upgrade", False) is False

    async def test_install_tensorrt_packages_no_upgrade_by_default(self, service):
        with patch.object(
            service, "_install_packages", new_callable=AsyncMock
        ) as mock_install:
            await service.install_tensorrt_packages()

        call_kwargs = mock_install.call_args[1]
        assert call_kwargs.get("upgrade", False) is False
