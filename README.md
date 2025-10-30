# OS-Sentinel

## Usage

### Installation

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

### Modes

## Citation
