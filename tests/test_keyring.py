"""密钥加密存储测试。"""
from pathlib import Path

import pytest

from core.storage.keyring import Keyring


def test_encrypt_decrypt_roundtrip(tmp_path):
    key_file = tmp_path / "sub" / ".keyring"
    ring = Keyring(key_file)
    token = ring.encrypt("sk-123456")
    assert token and token != "sk-123456"
    assert ring.decrypt(token) == "sk-123456"
    # 密钥文件被创建
    assert key_file.exists()


def test_key_file_persists_across_instances(tmp_path):
    key_file = tmp_path / ".keyring"
    ring1 = Keyring(key_file)
    token = ring1.encrypt("secret-value")
    ring2 = Keyring(key_file)  # 新实例复用同一密钥
    assert ring2.decrypt(token) == "secret-value"


def test_empty_string_roundtrip(tmp_path):
    ring = Keyring(tmp_path / ".keyring")
    assert ring.encrypt("") == ""
    assert ring.decrypt("") == ""


def test_invalid_token_returns_empty(tmp_path):
    ring = Keyring(tmp_path / ".keyring")
    assert ring.decrypt("not-a-valid-token!!") == ""


def test_rotate_key_invalidates_old(tmp_path):
    key_file = tmp_path / ".keyring"
    ring = Keyring(key_file)
    token = ring.encrypt("value")
    ring.rotate_key()
    # 旧密文用新密钥解不开
    assert ring.decrypt(token) == ""
