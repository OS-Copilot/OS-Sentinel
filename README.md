# OS-Sentinel
[![arXiv](https://img.shields.io/badge/arXiv-2510.24411-b31b1b.svg)](http://arxiv.org/abs/2510.24411) 
![License](https://img.shields.io/badge/License-MIT-blue)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2510.24411)
[![Discord](https://img.shields.io/discord/1222168244673314847?logo=discord&style=flat)](https://discord.com/invite/rXS2XbgfaD)
[![🌐 Website](https://img.shields.io/badge/Website-🌐-informational)](https://qiushisun.github.io/OS-Sentinel-Home/)

## 🛠️ Usage

### 📦 Installation

1. Clone this repository and [set up the environment of AndroidWorld](https://github.com/google-research/android_world?tab=readme-ov-file#installation); you may still need to install extra packages needed listed in `requirements.txt` although you have already installed AndroidWorld;

    ```shell
    git clone https://github.com/OS-Copilot/OS-Sentinel
    cd OS-Sentinel
    # install AndroidWorld
    # requirements.txt contains packages not included by AndroidWorld
    pip install -r requirements.txt
    ```

2. [Install Node.js and Appium following the instruction from MobileSafetyBench](https://github.com/jylee425/mobilesafetybench?tab=readme-ov-file#appium);
3. Run root.py and it will configure the environment of MobileSafetyBench automatically.

    ```shell
    conda activate android
    python root.py
    ```

    and you can run the script of MobileSafetyBench (`msb.py`) under the environment of AndroidWorld.

> [!NOTE]  
> Env `OPENAI_API_KEY` (while `OPENAI_BASE_URL` is optional) is needed when calling external VLM.

### 🔀 Modes

1. `step`: to check safety of single-step action in rule-based and VLM-based manners;

    ```shell
    timestep_new, in_danger = env.record(action)
    ```

2. `record`: to record trajectories of actions proposed by mobile agent.

    ```shell
    timestep_new = env.record(action)
    ```

    this method fix the system states before each action and `env.record("terminate()")` is needed at the end or the last action cannot be recorded.

## 📋 Citation
```bibtex
@article{sun2025ossentinel,
  title={OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows},
  author={Qiushi Sun and Mukai Li and Zhoumianze Liu and Zhihui Xie and Fangzhi Xu and Zhangyue Yin and Kanzhi Cheng and Zehao Li and Zichen Ding and Qi Liu and Zhiyong Wu and Zhuosheng Zhang and Ben Kao and Lingpeng Kong},
  journal={arXiv preprint arXiv:2510.24411},
  year={2025}
}

```