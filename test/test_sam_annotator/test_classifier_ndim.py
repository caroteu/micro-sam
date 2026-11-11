import platform

import numpy as np
import pytest
from napari.utils.key_bindings import coerce_keybinding
from qtpy import QtWidgets
from skimage.data import binary_blobs

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.pixel_classifier import PixelClassifier
from micro_sam.sam_annotator.object_classifier import ObjectClassifier


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize("classifier_cls", [PixelClassifier, ObjectClassifier])
def test_classifier_shortcuts_and_button_labels(make_napari_viewer_proxy, classifier_cls):
    viewer = make_napari_viewer_proxy()
    annotator = classifier_cls(viewer)

    assert coerce_keybinding("Shift-C") in viewer.keymap
    assert coerce_keybinding("C") not in viewer.keymap

    buttons = annotator._train_and_predict_widget.native.findChildren(QtWidgets.QPushButton)
    button_labels = {button.text() for button in buttons}
    assert "Train and Predict [Shift + T]" in button_labels
    assert "Clear Annotations [Shift + C]" in button_labels

    for key, name in (("Alt-A", "annotations"), ("Alt-P", "prediction")):
        toggle = viewer.keymap[coerce_keybinding(key)]
        assert viewer.layers[name].visible
        toggle(viewer)
        assert not viewer.layers[name].visible
        toggle(viewer)
        assert viewer.layers[name].visible

    annotator._select_class(3)
    annotator._select_class(1)
    next_class = viewer.keymap[coerce_keybinding("Alt-Right")]
    previous_class = viewer.keymap[coerce_keybinding("Alt-Left")]
    selected = []
    for cycle in (next_class, next_class, previous_class, previous_class):
        cycle(viewer)
        selected.append(viewer.layers["annotations"].selected_label)
    assert selected == [3, 1, 3, 1]
    assert annotator._class_group.checkedId() == 1
    label_texts = {label.text() for label in annotator.findChildren(QtWidgets.QLabel)}
    assert "Previous / next class [Alt + Left / Right]" in label_texts

    viewer.layers.remove("annotations")
    next_class(viewer)
    viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize("classifier_cls", [PixelClassifier, ObjectClassifier])
def test_classifier_recreates_label_layers_for_3d(make_napari_viewer_proxy, classifier_cls):
    # Regression for the pixel/object classifier 3d crash: the tool opens with 2d placeholder label
    # layers. Loading a 3d image must recreate them at ndim=3 rather than reassigning 3d data + a
    # 3-element scale onto stale 2d layers (which crashes napari in Affine.set_slice).
    state = AnnotatorState()

    viewer = make_napari_viewer_proxy()
    annotator = classifier_cls(viewer)
    assert viewer.layers["annotations"].ndim == 2

    # Simulate a 3d image being loaded and selected.
    volume = np.stack(4 * [binary_blobs(64)])
    state.image_shape = volume.shape
    state.image_scale = (1.0, 1.0, 1.0)
    annotator._update_image()

    for name in ("annotations", "prediction"):
        layer = viewer.layers[name]
        assert layer.ndim == 3
        assert layer.data.ndim == 3
        assert tuple(layer.data.shape) == volume.shape
        assert len(layer.scale) == 3

    viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize("classifier_cls", [PixelClassifier, ObjectClassifier])
def test_classifier_classical_features_option(make_napari_viewer_proxy, classifier_cls):
    viewer = make_napari_viewer_proxy()
    annotator = classifier_cls(viewer)
    state = AnnotatorState()
    choices = list(classifier_cls.classical_feature_choices)

    # Off by default, which switches the classical features off in the backend.
    assert annotator._get_classical_features() is None

    # Restoring a spec enables the option with the stored selection and clears the cached features.
    setattr(state, annotator.features_attr, np.zeros((1, 1)))
    annotator._set_options(False, None, False, "fixed", choices[:2])
    assert annotator._get_classical_features() == choices[:2]
    assert getattr(state, annotator.features_attr) is None

    # An empty selection is treated as switched off.
    annotator._set_options(False, None, False, "fixed", [])
    assert annotator._get_classical_features() is None
    annotator._set_options(False, None, False, "fixed", None)
    assert annotator._get_classical_features() is None
    viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_object_classifier_toggles_segmentation(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    segmentation = viewer.add_labels(np.zeros((64, 64), dtype="uint32"), name="my_cells")
    ObjectClassifier(viewer)

    toggle = viewer.keymap[coerce_keybinding("Shift-S")]
    toggle(viewer)
    assert not segmentation.visible
    toggle(viewer)
    assert segmentation.visible
    viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_object_classifier_highlights_uncertain_objects(make_napari_viewer_proxy):
    state = AnnotatorState()
    state.image_shape, state.ndim = (64, 64), 2

    # 16 square objects. The only feature is the object id, so the classes are split along the ids.
    segmentation = np.zeros((64, 64), dtype="uint32")
    for i in range(16):
        row, col = divmod(i, 4)
        segmentation[16 * row:16 * row + 8, 16 * col:16 * col + 8] = i + 1

    viewer = make_napari_viewer_proxy()
    viewer.add_labels(segmentation, name="segmentation")
    annotator = ObjectClassifier(viewer)
    annotator._update_image()
    seg_ids = np.arange(1, 17)
    state.object_features, state.seg_ids = seg_ids[:, None].astype("float32"), seg_ids

    # Annotate the objects 1 and 2 with class 1, and the objects 15 and 16 with class 2.
    annotations = np.zeros_like(segmentation)
    annotations[2:4, 2:4], annotations[2:4, 18:20] = 1, 1
    annotations[50:52, 34:36], annotations[50:52, 50:52] = 2, 2
    viewer.layers["annotations"].data = annotations

    # Off by default.
    annotator._run_train_and_predict()
    assert "uncertain objects" not in viewer.layers

    # Switching it on highlights the requested number of unannotated objects for the current classifier.
    annotator._highlight_uncertain.value = True
    annotator._n_uncertain.value = 3
    layer = viewer.layers["uncertain objects"]
    assert len(layer.data) == 3
    np.testing.assert_array_equal(layer.features["rank"], [1, 2, 3])
    highlighted = segmentation[tuple(np.round(layer.data).astype(int).T)]
    assert not np.isin(highlighted, [0, 1, 2, 15, 16]).any()
    assert viewer.layers.selection.active is viewer.layers["annotations"]

    # Training again updates the highlight. Clearing the annotations and switching it off remove it.
    annotator._n_uncertain.value = 5
    annotator._run_train_and_predict()
    assert len(viewer.layers["uncertain objects"].data) == 5
    annotator._clear_annotations()
    assert "uncertain objects" not in viewer.layers
    viewer.layers["annotations"].data = annotations
    annotator._run_train_and_predict()
    assert "uncertain objects" in viewer.layers
    annotator._highlight_uncertain.value = False
    assert "uncertain objects" not in viewer.layers
    viewer.close()
