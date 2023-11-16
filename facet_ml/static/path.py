
from pathlib import Path
import os

STATIC_FOLDER = Path(__file__).parent
STATIC_MODELS = {}

STATIC_MODELS["edge_classifier"] = os.path.join(STATIC_FOLDER,
                                                "Models",
                                                "edge_classifier.pickle")

STATIC_MODELS["bg_segmenter"] = os.path.join(STATIC_FOLDER,
                                                "Models",
                                                "bg_segmenter.pickle")

STATIC_MODELS["detectron2_model"] = os.path.join(STATIC_FOLDER,
                                                "Models",
                                                "detectron2",
                                                "model_final.pth")

STATIC_MODELS["detectron2_config"] = os.path.join(STATIC_FOLDER,
                                                "Models",
                                                "detectron2"
                                                "model_final.pth")

STATIC_MODELS["crystal_multicrystal"] = os.path.join(STATIC_FOLDER,
                                                "Models",
                                                "RF_C_MC.sav")

STATIC_MODELS["crystalline_noncrystalline"] = os.path.join(STATIC_FOLDER,
                                                "Models",
                                                "RF_C-MC_I-P.sav")

STATIC_MODELS["incomplete_poorlysegmented"] = os.path.join(STATIC_FOLDER,
                                                "Models",
                                                "RF_I_P.sav")

STATIC_MODELS["segment_anything_vit_l"] = os.path.join(STATIC_FOLDER,
                                                       "Models",
                                                       "sam_vit_l_0b3195.pth")