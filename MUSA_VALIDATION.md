# IndexTTS-2.5 / MUSA 验证记录（脱敏）

使用单个可见 MUSA 设备，进程内为 `musa:0`。
具体硬件型号、物理设备编号、主机标识和本地目录不公开；以下历史测量只代表
验证环境，不构成所有 MUSA 设备的性能承诺。权重通过 `--model-dir` 指定。

## 已验证范围

- v2.5 标准语音生成；GPT BF16，其余模型 FP32；不启用 TorchInductor。
- 参考 mel 的 1024 点 STFT、CAMPPlus fbank 的 512 点 FFT 使用 native MUSA C++。
- CAMPPlus、BigVGAN 的全部参数和 buffer 均在 MUSA，实际输出与 CPU 对拍。
- BigVGAN 是神经声码器，不是算子库。forward 在
  `indextts/s2mel/modules/bigvgan/bigvgan.py`，MUSA 使用 PyTorch 激活实现，
  不加载 `alias_free_activation/cuda` 扩展。
- 主推理入口为 `indextts/infer_v2_5.py` 的 `infer` / `infer_generator`。

音频读取、重采样、文本处理和 Hugging Face feature extractor 仍使用原有 CPU
实现。QwenEmotion 已验证独立进程的 MUSA 路径，见 README 的隔离依赖说明；
未验证 DeepSpeed、CUDA 专用加速引擎；未声称全模型 CPU
逐元素一致、全 GPU 预处理或端到端加速比。

## 精度策略与证据

`IndexTTS2` 初始化 MUSA 后默认关闭进程级 `torch.backends.mudnn.allow_tf32`。
不能由其它线程在推理过程中切回 TF32。GPT 的显式 BF16 不受此设置影响。

同一真实参考音频 `examples/voice_01.wav`，CPU PyTorch / torchaudio 为基线：

| 项目 | mean abs | max abs | 相对 L2 |
|---|---:|---:|---:|
| fbank | 1.96e-6 | 9.61e-4 | — |
| mel | 5.92e-7 | 8.49e-5 | — |
| CAMPPlus（包含 fbank 替换） | 1.77e-6 | 6.44e-6 | 2.50e-6 |
| BigVGAN（相同 CPU mel 输入） | 5.78e-7 | 1.32e-5 | 1.54e-5 |

FFT 回归门限：`atol=4e-5, rtol=3e-5`；mel/fbank 门限：
`atol=2e-3, rtol=1e-4`。模块诊断门限：输出有限、相对 L2 ≤1%、余弦相似度
≥0.9999；不替代语音主观质量验收。相关测试共 **37 passed**。

发布前重新运行音频、余弦检索、CPU trace、音频保存、尾部淡出及 WebUI 回归：
**68 passed**。本次 native 扩展在新的构建目录中按可见设备自动选择架构编译。

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 python3 -m pytest -q \
  tests/test_musa_audio.py tests/test_musa_cosine.py tests/test_inference_trace.py \
  tests/test_audio_save.py tests/test_tail_fade.py \
  tests/test_webui_handlers.py tests/test_webui_syntax.py
```

## 冷数据性能

CPU 4 线程；交替 CPU/MUSA；每次使用 CPU 128 MiB 和 GPU 256 MiB 缓冲冲刷
缓存，冲刷和首次编译不计时；MUSA 同步墙钟。特征测试每项 20 轮，模块 3 轮。
以下均为中位数，CPU/GPU 列单位为 µs；这些不是纯 GPU kernel 时长。

| 项目 | CPU | MUSA | 加速比 |
|---|---:|---:|---:|
| 15 秒 STFT，输入驻留 | 396.84 | 256.99 | 1.54x |
| 15 秒 fbank，输入驻留 | 2152.60 | 524.20 | 4.11x |
| 15 秒 fbank，包含 H2D | 2103.39 | 728.37 | 2.89x |
| 15 秒 mel，输入驻留 | 3395.27 | 421.83 | 8.05x |
| 15 秒 mel，包含 H2D | 3056.60 | 676.40 | 4.52x |
| CAMPPlus，同一真实特征，输入驻留 | 30820.08 | 19238.37 | 1.60x |
| BigVGAN，同一真实 mel，输入驻留 | 1343904.17 | 77445.72 | 17.35x |

3 秒 STFT 为 131.25 → 130.44 µs，基本持平，不能宣称短音频 FFT 有显著加速。
15 秒输入由真实音频重复截取；模块对拍使用原始约 2.44 秒参考音频。
BigVGAN 的速度来自可用的 MUSA PyTorch 算子，不是新增的 BigVGAN 融合核。

## 产物与复现

通过精度策略的历史结果保留在本地，不随提交发布。使用
`--output-dir outputs/musa_validation/new_run` 可在自己的设备生成新结果：

- `results.json`：特征精度、逐次计时、demo 自定义 kernel 计数。
- `modules.json`：模块对拍、设备审计和逐次性能。
- `demo.wav`：项目推理链生成的 22050 Hz 单声道音频。
- `bigvgan_cpu.wav` / `bigvgan_musa.wav`：相同 mel 的声码器对拍音频。
- `demo.gpu.json` / `modules.gpu.json`：GPU-only trace；`.raw.json` 保留原始导出。
- `run.log` / `modules.log` / `tests.log` / `source_sha256.txt`：复现证据。

demo trace 确认 1024 点 STFT 与 512 点 FFT 各执行一次。
有 profiler 的 demo 耗时不能直接与旧 demo 比较，生成长度也不同。

在仓库根执行，使用新的输出目录避免覆盖：

```bash
export PYTHONPATH=. OMP_NUM_THREADS=4
python3 -m pytest -q tests/test_musa_audio.py tests/test_audio_save.py tests/test_tail_fade.py
python3 -u tools/validate_musa_audio.py --model-dir checkpoints --output-dir outputs/musa_validation/new_run --demo
python3 -u tools/validate_musa_modules.py --model-dir checkpoints --output-dir outputs/musa_validation/new_run
```

## 避免重复踩坑

1. mel 输入在 CPU 时，即使注册了 native 算子，demo 仍会走 CPU STFT。
   必须显式传入 MUSA 输入，并用实际模型 trace 验证调用。
2. `self.device` 是字符串，不要写 `self.device.type`。
3. FFT 的 block 索引包含 batch，需要先计算 batch 内 frame；已添加多 batch 测试。
4. native 扩展显式指定构建目录时，加载前必须创建目录；产物放在可写缓存目录。
5. 单算子特征误差较小不代表整个声码器误差可接受：TF32 开启时 BigVGAN
   相对 L2 达 5.33%，CAMPPlus 达 1.08%。该配置的失败记录保留在本地，
   不能将其较高性能当作通过精度的方案。
6. 默认关闭 TF32 后重新跑精度、冷数据性能和 demo，不能沿用 TF32 性能数字。

## 正常推理 trace 与发布边界

权重、音频、构建产物、原始日志和 trace 不随本适配提交上传。
原始 trace 可能包含设备属性、进程信息和本机路径，公开前必须另行脱敏。

历史两个验证目录的 trace 只开启了 `ProfilerActivity.MUSA`，即使原始导出中
包含部分运行时事件，也没有完整的 CPU 算子活动；不能靠取消过滤恢复。
此后验证脚本同时开启 CPU/MUSA，完整记录保存在 `.raw.json`。
正常项目入口新增 `--trace`：

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 python3 indextts/infer_v2_5.py \
  --cfg_path checkpoints/config.yaml \
  --model_dir checkpoints \
  --prompt_wav examples/voice_01.wav --text '欢迎体验 MUSA 算子适配。' --lang ZH \
  --output outputs/normal_cpu_musa.wav \
  --trace outputs/traces/normal.cpu_musa.json
```

这个入口只运行一次模型生成；采集范围是 `IndexTTS2.infer`（包含请求预处理），
不包含构造模型和加载权重。完整 trace 同时保存 CPU 和 MUSA 活动，不进行 GPU-only 过滤。
