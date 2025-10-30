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

    > [!NOTE]  
    > Env `OPENAI_API_KEY` (while `OPENAI_BASE_URL` is optional) is needed when calling external VLM.

### Modes

1. `step`: to check safety of single-step action in rule-based and VLM-based manners;

    ```shell
    timestep_new, in_danger = env.record(action)
    ```

2. `record`: to record trajectories of actions proposed by mobile agent.

    ```shell
    timestep_new = env.record(action)
    ```

    this method fix the system states before each action and `env.record("terminate()")` is needed at the end or the last action cannot be recorded.

## Citation
