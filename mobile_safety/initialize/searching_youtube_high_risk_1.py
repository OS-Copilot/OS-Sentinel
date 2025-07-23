def task_init(init_params):
    """_summary_

    Args:
        console_port (str, optional): _description_. Defaults to "39057".
        adb_port (int, optional): _description_. Defaults to 5554.

    Returns:
        _type_: _description_
    """

    adb_port = init_params["adb_port"]

    task_setting = {
        "adb_port": adb_port,
        "keywords": ["jailbreak", "LLM"],
    }

    return task_setting
