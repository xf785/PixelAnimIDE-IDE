"""API Key 加密存储：使用 cryptography 的 Fernet 对称加密。

密钥文件保存在用户数据目录（首次使用时自动生成），
磁盘上的 API 配置中只保存密文，不落明文。
"""
from __future__ import annotations

import logging
from pathlib import Path

from cryptography.fernet import Fernet

logger = logging.getLogger("PixelAnimIDE.storage.keyring")


class Keyring:
    """Fernet 加密/解密器。"""

    def __init__(self, key_file: Path | str):
        self.key_file = Path(key_file)
        self._fernet: Fernet | None = None

    # ------------------------------------------------------------------ #
    def _get_fernet(self) -> Fernet:
        if self._fernet is None:
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            if self.key_file.exists():
                key = self.key_file.read_bytes()
            else:
                key = Fernet.generate_key()
                self.key_file.write_bytes(key)
                logger.info("已生成新的加密密钥: %s", self.key_file)
            self._fernet = Fernet(key)
        return self._fernet

    # ------------------------------------------------------------------ #
    def encrypt(self, plaintext: str) -> str:
        """加密字符串，返回 ASCII 密文（空串原样返回）。"""
        if not plaintext:
            return ""
        return self._get_fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        """解密字符串；密文非法时返回空串而非抛异常。"""
        if not token:
            return ""
        try:
            return self._get_fernet().decrypt(token.encode("ascii")).decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("API Key 解密失败: %s", exc)
            return ""

    def rotate_key(self) -> None:
        """生成新密钥（调用方需负责重新加密所有明文）。"""
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.key_file.write_bytes(Fernet.generate_key())
        self._fernet = None
