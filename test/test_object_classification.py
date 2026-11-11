import numpy as np

from micro_sam.object_classification import select_uncertain_objects


def test_select_uncertain_objects():
    probabilities = np.array([
        [0.9, 0.1, 0.0],  # margin 0.8
        [0.4, 0.35, 0.25],  # margin 0.05
        [0.5, 0.5, 0.0],  # margin 0.0
        [0.2, 0.5, 0.3],  # margin 0.2
    ])

    # The objects are sorted from the smallest to the largest margin.
    np.testing.assert_array_equal(select_uncertain_objects(probabilities, 3), [2, 1, 3])
    np.testing.assert_array_equal(select_uncertain_objects(probabilities, 10), [2, 1, 3, 0])

    # Annotated objects are not selected, even if fewer objects than requested remain.
    annotated = np.array([False, True, True, False])
    np.testing.assert_array_equal(select_uncertain_objects(probabilities, 3, annotated), [3, 0])
    assert len(select_uncertain_objects(probabilities, 3, np.ones(4, dtype=bool))) == 0
