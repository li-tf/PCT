# simulation0716

这是`results0716`对应的Windows原生OpenGATE 10.1.0仿真配置。物理与几何沿用
test0713，通量提高至论文的900 protons/mm²/projection，即450,000质子/角度。

正式仿真已经完成并通过迁移校验：720个角度全部成功，12个单线程进程并行，
launcher墙钟时间为6,639.24 s（1 h 50 min 39 s），配置事件数为324,000,000。
仿真采用200 MeV单能质子、半径100 mm水圆柱、25根直径5 mm铝柱、Vacuum外部
介质和`QGSP_BIC_EMZ`物理列表。

代码与数据分离：

- ROOT：`D:\OpenGATE\data\simulation_data\results0716\run_###`；
- 日志、统计、完成标志和manifest：本目录`qc\`。

PowerShell中依次运行：

```powershell
.\01_check_environment.bat
.\02_smoke_test.bat
.\03_run_full.bat 12
```

每个角度使用单独进程和`20260713 + angle_index`随机种子，支持依靠代码侧
`qc\runs\run_###\completed.flag`断点续跑。

当前正式ROOT已迁移到仓库侧`data/simulation_data/results0716/run_###/`，入口和
出口文件分别为`PhaseSpaceIn.root`与`PhaseSpaceOut.root`。代码侧`qc/`保留
launcher摘要、逐角度OpenGATE统计、运行元数据、manifest和ROOT SHA-256清单。
上述批处理命令用于复现或断点续跑；已有720角度完整数据时不需要再次启动仿真。
