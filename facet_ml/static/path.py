from pathlib import Path

STATIC_FOLDER = Path(__file__).resolve().parent
STATIC_MODELS = {
    "edge_classifier": STATIC_FOLDER / "Models" / "edge_classifier.pickle",
    "bg_segmenter": STATIC_FOLDER / "Models" / "bg_segmenter.pickle",
    "maskrcnn": STATIC_FOLDER / "Models" / "torch" / "maskrcnn_model.pth",
    "detectron2_model": STATIC_FOLDER / "Models" / "detectron2" / "model_final.pth",
    "detectron2_config": STATIC_FOLDER / "Models" / "detectron2" / "config.yaml",
    "crystal_multicrystal": STATIC_FOLDER
    / "Models"
    / "2023_11_models_length-agnostic"
    / "RF_C_MC.sav",
    "crystalline_noncrystalline": STATIC_FOLDER
    / "Models"
    / "2023_11_models_length-agnostic"
    / "RF_C-MC_I.sav",
    "incomplete_poorlysegmented": STATIC_FOLDER
    / "Models"
    / "2023_11_original_default_features-agnostic"
    / "RF_I_P.sav",
    "crystal_multicrystal_incomplete": STATIC_FOLDER
    / "Models"
    / "2023_11_models_length-agnostic"
    / "RF_C_MC_I.sav",
    "segment_anything_vit_l": STATIC_FOLDER / "Models" / "sam_vit_l_0b3195.pth",
}
