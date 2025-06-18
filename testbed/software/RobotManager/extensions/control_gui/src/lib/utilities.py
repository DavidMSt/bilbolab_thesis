from core.utils.logging_utils import Logger


def check_for_spaces(string):
    return ' ' in string


def split_path(path):
    # Remove any leading/trailing slashes
    path = path.strip('/')
    parts = path.split('/')

    if not parts or parts[0] == '':
        return '', ''

    first = parts[0]
    remainder = '/'.join(parts[1:])
    return first, remainder


def strip_id(path, target_id) -> str | None:
    # Ensure no leading or trailing slashes
    path = path.strip('/')
    parts = path.split('/')

    try:
        index = parts.index(target_id)
        return '/'.join(parts[index + 1:])
    except ValueError:
        # target_id not found
        return None


if __name__ == '__main__':
    path1 = 'id1/id2/id3'
    print(split_path(path1))

    print(strip_id(path1, 'id1'))

    print(split_path('/'))
    print(split_path('/id1'))


def warn_on_unknown_kwargs(kwargs: dict, config_template: dict, logger: Logger):
    """
    Logs a warning for any kwargs that aren't present in the config_template.

    Parameters:
        kwargs (dict): The keyword arguments passed in.
        config_template (dict): The set of allowed/default config keys.
        logger (Logger): Logger instance to use for warnings.
    """
    for key in kwargs:
        if key not in config_template:
            logger.warning(f"Warning: Unrecognized config key passed via kwargs: '{key}'")
