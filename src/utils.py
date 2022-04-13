from omegaconf import DictConfig, OmegaConf, open_dict


def add_new_key_to_cfg(cfg: DictConfig, value: str, *keys: str) -> None:
    """Add value to config section following key path, where key(s) do not already exist in config

    Args:
        cfg: config to add value to
        value: value to add to config
        *keys: config section names pointing to the desired key. Final item in list will be given value of value

    Returns: None

    """
    cfg_section = cfg
    for key in keys[:-1]:
        cfg_section = cfg_section[key]
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg_section[keys[-1]] = value
