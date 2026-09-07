
## 2026-09-05T02:00Z P5.2 v13 训练启动失败（1 次，即时纠正）
- 方案：train_p2.py 默认参数启动 v13
- 症状：`SparseEncoder.__init__() got unexpected keyword 'cmd_dim'` 启动即崩
- 根因：默认 `--ablation none` + 未加 `--explicit-cmd` → enc_cls 路由到
  SparseEncoder；v12 实际启动参数集含 --explicit-cmd --loco-cmd
  --graft 0.25 --graft-neck 0.5（P3.2.2 日志有记录但未写入启动脚本化）
- 修复：带全参数重启；教训=启动命令须从上轮日志参数集原样复制
