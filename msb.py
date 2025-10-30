import argparse
import root

assert root.init()
from mobile_safety.logger import Logger
from mobile_safety.environment import MobileSafetyEnv
from mobile_safety.prompt._prompt import PromptBuilder

parser = argparse.ArgumentParser()

parser.add_argument("--avd_name", type=str, default="AndroidWorldAvd")
parser.add_argument("--avd_name_sub", type=str, default="")
parser.add_argument("--port", type=int, default=5554)
parser.add_argument("--appium_port", type=int, default=4723)

parser.add_argument("--traj_dir", type=str, default="data")
parser.add_argument("--task_id", type=str, default="writing_memo")
parser.add_argument("--scenario_id", type=str, default="high_risk_2")
parser.add_argument("--prompt_mode", type=str, default="basic", choices=["basic", "safety_guided", "scot"])

parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gui", type=bool, default=True)
parser.add_argument("--delay", type=float, default=10.0)

args, _ = parser.parse_known_args()
env = MobileSafetyEnv(
    avd_name=args.avd_name,
    avd_name_sub=args.avd_name_sub,
    gui=args.gui,
    delay=args.delay,
    task_tag=f"{args.task_id}_{args.scenario_id}",
    prompt_mode=args.prompt_mode,
    port=args.port,
    appium_port=args.appium_port,
    traj_dir=args.traj_dir,
    # is_emu_already_open=True
)

logger = Logger(args)
prompt_builder = PromptBuilder(env)

if "gpt" in args.model:
    from mobile_safety.agent.gpt_agent import GPTAgent
    agent = GPTAgent(model_name=args.model, seed=args.seed, port=args.port,)
if "gemini" in args.model:
    from mobile_safety.agent.gemini_agent import GeminiAgent
    agent = GeminiAgent(model_name=args.model, seed=args.seed, port=args.port,)
if "claude" in args.model:
    from mobile_safety.agent.claude_agent import ClaudeAgent
    agent = ClaudeAgent(model_name=args.model,seed=args.seed, port=args.port,)

# reset the environment
timestep = env.reset()
prompt = prompt_builder.build(
    parsed_obs=env.parsed_obs,
    action_history=env.evaluator.actions[1:],
    action_error=env.action_error,
)

logger.log(timestep=timestep)

while True:
    response_dict, final_prompt = agent.get_response(
        timestep=timestep, 
        system_prompt=prompt.system_prompt, 
        user_prompt=prompt.user_prompt,
    )

    if response_dict["action"] == None:
        print("Error in response")

    action = response_dict["action"]
    timestep_new, in_danger = env.record(action)
    if timestep_new is None:
        continue
    timestep = timestep_new

    prompt = prompt_builder.build(
        parsed_obs=env.parsed_obs,
        action_history=env.evaluator.actions[1:],
        action_error=env.action_error,
    )

    logger.log(prompt=final_prompt, response_dict=response_dict, timestep=timestep)

    if timestep.last() or env.evaluator.progress["finished"]:
        break

# env.record("terminate()")
print("\n\nReward:", timestep_new.curr_rew)
