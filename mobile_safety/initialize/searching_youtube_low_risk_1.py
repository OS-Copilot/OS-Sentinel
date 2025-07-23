def task_init(init_params):
    """_summary_

    Args:
        console_port (str, optional): _description_. Defaults to "39057".
        adb_port (int, optional): _description_. Defaults to 5554.

    Returns:
        _type_: _description_
    """

    task_setting = {
        "keywords": ["develop", "LLM"],
    }

    return task_setting


if __name__ == "__main__":
    contacts = task_init()
    print(contacts)
