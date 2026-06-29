TRAIN_HISTORY_FIELDS = [
    "epoch",
    "phase",
    "train_loss",
    "train_model_loss",
    "train_objective_loss",
    "legacy_test_loss",
    "test_loss",
    "optimizer_steps_epoch",
    "optimizer_steps_total",
    "epoch_time_sec",
    "cumulative_time_sec",
]


def build_pretrain_history_row(pretrain_legacy_test_loss):
    return {
        "epoch": -1,
        "phase": "pretrain_eval",
        "train_loss": None,
        "train_model_loss": None,
        "train_objective_loss": None,
        "legacy_test_loss": pretrain_legacy_test_loss,
        "test_loss": pretrain_legacy_test_loss,
        "optimizer_steps_epoch": 0,
        "optimizer_steps_total": 0,
        "epoch_time_sec": 0.0,
        "cumulative_time_sec": 0.0,
    }


def build_epoch_history_row(
    epoch,
    train_loss,
    train_model_loss,
    train_objective_loss,
    test_loss,
    optimizer_steps_epoch,
    optimizer_steps_total,
    epoch_time_sec,
    cumulative_time_sec,
):
    return {
        "epoch": epoch,
        "phase": "epoch_end",
        "train_loss": train_loss,
        "train_model_loss": train_model_loss,
        "train_objective_loss": train_objective_loss,
        "legacy_test_loss": test_loss,
        "test_loss": test_loss,
        "optimizer_steps_epoch": optimizer_steps_epoch,
        "optimizer_steps_total": optimizer_steps_total,
        "epoch_time_sec": epoch_time_sec,
        "cumulative_time_sec": cumulative_time_sec,
    }
