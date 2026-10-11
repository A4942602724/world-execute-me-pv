# world.execute(me) — 代码渲染的水墨 PV

为原创角色 "Claude" 制作的歌曲 PV。前半段由代码逐帧渲染的水墨剪影（Python + HTML/Playwright），
后半段在 Blender 5.2 中制作三维场景。每一帧都是歌曲时间的函数：歌词逐字时间、鼓点、段落都写进关键帧。

## 目录

- `pv/` — 渲染管线：歌曲分析、歌词对齐、各场景脚本、转场、Blender 脚本（`pv/blender/`）
- `docs/pv_v3_design.md` — v3 分镜设计
- `requirements.txt` / `package.json` / `env.ps1` — Python、Node 环境

## 未包含的内容

美术素材（`assets/`）、输入音频与歌词（`input/`）、渲染输出、模型文件和 API 配置不在仓库中。
`pv/imagegen.py` 运行时从 `input/shouquanapi.txt` 读取接口与密钥，需自备。

## 许可

- 代码：MIT，见 `LICENSE`。
- 角色设定、美术、歌曲及歌词：保留所有权利，未随本许可开放。
