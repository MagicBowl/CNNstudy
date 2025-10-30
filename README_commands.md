# 常用运行命令（Windows / PowerShell）

下面是一些在这个工作区常用的命令，复制到 PowerShell 终端后运行。

注意：在 VS Code 集成终端中请确保当前终端是 PowerShell（提示符以 `PS` 开头），且不要把 PowerShell 命令粘到 Python REPL（>>>）中。

切到项目目录：

```powershell
Set-Location -Path 'C:\Users\MagicBowl\source\repos\out'
```

使用特定 Python 解释器运行脚本（推荐）：

```powershell
& 'C:\Users\MagicBowl\AppData\Local\Programs\Python\Python313\python.exe' .\scripts\gpu_smoketest.py
& 'C:\Users\MagicBowl\AppData\Local\Programs\Python\Python313\python.exe' .\scripts\amp_smoketest.py
```

短训练（1 epoch，debug 模式 num_workers=0）：

```powershell
# $env:DEBUG = '1'  # 可选：在 PowerShell 中设置环境变量
& 'C:\Users\MagicBowl\AppData\Local\Programs\Python\Python313\python.exe' .\scripts\train_runner.py --parquets .\a.parquet .\b.parquet --batch-size 8 --epochs 1 --save-path short_model.pth --debug
# Remove-Item Env:DEBUG  # 完成后可清除
```

单张预测（读取 parquet 的第 0 行并把图片保存到 preds/）：

```powershell
& 'C:\Users\MagicBowl\AppData\Local\Programs\Python\Python313\python.exe' .\scripts\predict_single.py --parquet .\a.parquet --index 0 --save-dir .\preds
```

在 PowerShell 中运行时遇到 `unicodeescape` 错误：

- 原因：把含反斜杠的路径直接放到 Python 字符串或 -c 参数里，触发了转义。解决办法是：

  - 在 Python 脚本中使用原始字符串 r'...' 或双反斜杠。
  - 在 PowerShell 中把路径用引号包起来（如本文件示例），或使用正斜杠 `/`。

在 VS Code 中运行小片段 Python 代码：

- 推荐：选中代码然后点击“Run Selection/Line in Python Terminal”（确保目标是 Python 终端）。

- 不推荐：在 PowerShell 里粘贴 Python 代码（两者语法不同，会产生 “缺少表达式” 或 “意外的标记” 错误）。

如果你想把这些命令加入到 VS Code 的任务或调试配置中，我可以帮你创建对应的 `tasks.json` / `launch.json` 配置。
