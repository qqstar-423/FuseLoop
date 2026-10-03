developer@5989a23a0e2c433b87152c2132a9c46d:/mnt/workspace$ curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash


____    _    _   _ _   _ ____        _
  / ___|  / \  | \ | | \ | | __ )  ___ | |_ 
 | |     / _ \ |  \| |  \| |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  | |_) | (_) | |_ 
  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|

🚀 CANNBot Install Helper 安装脚本

✓ Node.js v18.19.1

📦 正在安装 @cannbot-ai/install-helper@latest...

npm WARN EBADENGINE Unsupported engine {
npm WARN EBADENGINE   package: '@cannbot-ai/install-helper@1.2.0',
npm WARN EBADENGINE   required: { node: '>=20.0.0' },
npm WARN EBADENGINE   current: { node: 'v18.19.1', npm: '10.2.4' }
npm WARN EBADENGINE }

added 78 packages in 18s

34 packages are looking for funding
  run `npm fund` for details
npm notice 
npm notice New major version of npm available! 10.2.4 -> 12.1.0
npm notice Changelog: https://github.com/npm/cli/releases/tag/v12.1.0
npm notice Run npm install -g npm@12.1.0 to update!
npm notice 

✅ 安装成功！

请运行: install-helper
developer@5989a23a0e2c433b87152c2132a9c46d:/mnt/workspace$ 
developer@5989a23a0e2c433b87152c2132a9c46d:/mnt/workspace$ install-helper

____    _    _   _ _   _ ____        _
  / ___|  / \  | \ | | \ | | __ )  ___ | |_
 | |     / _ \ |  \| |  \| |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  | |_) | (_) | |_
  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|

  CANNBot Install Helper

⠋ 正在加载 Skills 列表...→ 首次使用，正在克隆 cannbot-skills 仓库...
✔ Skills 列表加载完成

  ╔═════════════════════════════════════════════════════════════════╗
  ║                          选择安装类型                           ║
  ╚═════════════════════════════════════════════════════════════════╝

  💡 ↑↓ 移动 | ⏎ 确认

✔ 选择安装类型 > 安装 Plugin — 完整开发工作流（Skills + Agents + 配置文件）


  ╔═════════════════════════════════════════════════════════════════╗
  ║                     [1/4] 选择 AI 编程工具                      ║
  ╚═════════════════════════════════════════════════════════════════╝

✓ 检测到: OpenCode (v1.18.32)
  💡 ↑↓ 移动 | ⏎ 确认

✔ 选择 AI 编程工具 > 使用 OpenCode

  ╔═════════════════════════════════════════════════════════════════╗
  ║                       [2/4] 选择安装位置                        ║
  ╚═════════════════════════════════════════════════════════════════╝

  💡 ↑↓ 移动 | ⏎ 确认

✔ 选择安装位置 > global — 安装到全局（所有项目共享）

  ╔═════════════════════════════════════════════════════════════════╗
  ║                     [3/4] 选择要安装的插件                      ║
  ╚═════════════════════════════════════════════════════════════════╝

  💡 ↑↓ 移动 | ⏎ 确认

✔ 选择要安装的插件 > Triton 算子开发

  ╔═════════════════════════════════════════════════════════════════╗
  ║                         [4/4] 确认安装                          ║
  ╚═════════════════════════════════════════════════════════════════╝

→ 即将安装 1 个插件到 OpenCode（global 级别）：
→ 安装路径: /home/developer/.config/opencode
  • Triton 算子开发

  💡 ↑↓ 移动 | ⏎ 确认

✔ 确认安装 > 确认安装
✔ 插件数据加载完成
✔ [1/1] Triton 算子开发 — 8 skills, 0 agents

  安装完成! 1/1 成功

  ✓ Triton 算子开发 (8 skills, 0 agents)


  安装完成!

  安装到: /home/developer/.config/opencode
  Skills: 8 个  |  Agents: 0 个

  下一步:
    1. 启动: opencode
    2. 试试: 帮我开发一个 Abs 算子
    3. 更多: install-helper list
    4. 检查: install-helper doctor

  文档: plugins-official/triton-op-generator/quickstart.md