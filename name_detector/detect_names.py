import sys

import pkg_resources  # type: ignore

from name_detector.model import CatBoostModel
from name_detector.pipeline import TextPipeline


class NameDetector:
    __pipeline_path = pkg_resources.resource_filename("name_detector", "checkpoints/pipeline.joblib")
    __model_path = pkg_resources.resource_filename("name_detector", "checkpoints/catboost_model.cbm")

    def __init__(self):
        self.pipeline = TextPipeline.init_from(self.__pipeline_path)
        self.model = CatBoostModel.init_from(self.__model_path)

    def predict(self, text):
        # window all two and three consecutive word tuples
        windows = self.pipeline.get_windows(text)
        X_input, _ = self.pipeline.transform(windows)
        y_prob = self.model.predict_proba(X_input)
        return windows, y_prob


def main():
    if len(sys.argv) != 2:
        print("Usage: python name_detector_script.py <text>")
        sys.exit(1)

    text = sys.argv[1]

    name_detector = NameDetector()
    windows, y_prob = name_detector.predict(text)

    for window, prob in zip(windows, y_prob):
        print(f"{window:30s}  {prob[1]:.3f}")


if __name__ == "__main__":
    main()
