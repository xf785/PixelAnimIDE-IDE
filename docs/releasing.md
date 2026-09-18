# 发布流程（维护者）

1. **升版本号**：`config/settings.py` 的 `APP_VERSION` + `pyproject.toml` 的 `version`，
   并同步 README 顶部的 Release 徽章与 `ROADMAP*.md` 的「当前状态」。
2. **写发行说明**：新增 `docs/releases/v<版本>.md`（中英双语，首个 `# 标题` 会被用作 Release 标题）。
3. **提交并打附注标签**：

   ```bash
   git add -A && git commit -m "chore(release): v1.0.0 —— …"
   git tag -a v1.0.0 -m "v1.0.0 — <一句话摘要>"
   git push origin main && git push origin v1.0.0
   ```

4. **CI 自动发布**：`.github/workflows/release.yml` 监听 `v*` 标签，在 windows runner 上
   `PyInstaller onedir` 构建 → `tools/package_release.py` 打成 `PixelFoundry-<tag>-win64.zip`
   → 用仓库自带的 `GITHUB_TOKEN` 创建 Release 并附上 zip（说明文字取 `docs/releases/<tag>.md`）。

5. **本地也可手工出包**（与 CI 等价）：

   ```bash
   .\.venv\Scripts\python.exe -m PyInstaller PixelFoundry.spec --noconfirm --clean
   .\.venv\Scripts\python.exe tools\package_release.py --out release
   ```

   > 绿色包不写注册表；配置与密钥保存在 `%APPDATA%\PixelFoundry`。
   > 发布前请确认 zip 内**不含** `api_config.json`、`*.keyring`、`ui_settings.json` 等本地运行期文件。
