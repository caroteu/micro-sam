from typing import Dict, List, Optional

# TODO fully define the dict type
ExperimentSettings = List[Dict]
"""@private"""


def full_experiment_settings(
    use_boxes: bool = False,
    positive_range: Optional[List[int]] = None,
    negative_range: Optional[List[int]] = None,
) -> ExperimentSettings:
    """The full experiment settings.

    Args:
        use_boxes: Whether to run the experiments with or without boxes.
        positive_range: The different number of positive points that will be used.
            By defaul the values are set to [1, 2, 4, 8, 16].
        negative_range: The different number of negative points that will be used.
            By defaul the values are set to [1, 2, 4, 8, 16].

    Returns:
        The list of experiment settings.
    """
    experiment_settings = []
    if use_boxes:
        experiment_settings.append(
            {"use_points": False, "use_boxes": True, "n_positives": 0, "n_negatives": 0}
        )

    # set default values for the ranges if none were passed
    if positive_range is None:
        positive_range = [1, 2, 4, 8, 16]
    if negative_range is None:
        negative_range = [1, 2, 4, 8, 16]

    for n_positives in positive_range:
        for n_negatives in negative_range:
            experiment_settings.append(
                {"use_points": True, "use_boxes": use_boxes, "n_positives": n_negatives, "n_negatives": n_negatives}
            )

    return experiment_settings


def default_experiment_settings() -> ExperimentSettings:
    """The three default experiment settings.

    For the default experiments we use a single positive prompt,
    two positive and four negative prompts and box prompts.

    Returns:
        The list of experiment settings.
    """
    experiment_settings = [
        {"use_points": True, "use_boxes": False, "n_positives": 1, "n_negatives": 0},  # p1-n0
        {"use_points": True, "use_boxes": False, "n_positives": 2, "n_negatives": 4},  # p2-n4
        {"use_points": False, "use_boxes": True, "n_positives": 0, "n_negatives": 0},  # only box prompts
    ]
    return experiment_settings