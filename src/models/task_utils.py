def is_masked_lm_config(config: dict) -> bool:
    """
    Returns True when the experiment should be treated as a masked-LM setup.

    The explicit config flag takes precedence. As a fallback, BERT-like model
    names are treated as masked-LM because this project uses them as MLM
    baselines.
    """
    if bool(config.get("masked_lm", False)):
        return True
    model_name = str(config.get("model_name", "")).lower()
    model_family = str(config.get("model_family", "")).lower()
    return model_family == "masked_lm" or "bert" in model_name
