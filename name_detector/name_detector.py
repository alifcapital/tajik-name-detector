import pkg_resources  # type: ignore

from name_detector.model import CatBoostModel
from name_detector.pipeline import TextPipeline


class NameDetector:
    def __init__(self, pipeline_path, model_path):
        self.pipeline = TextPipeline.init_from(pipeline_path)
        self.model = CatBoostModel.init_from(model_path)

    def predict(self, text):
        # window all two and three consecutive word tuples
        windows = self.pipeline.get_windows(text)
        X_input, _ = self.pipeline.transform(windows)
        y_prob = self.model.predict_proba(X_input)
        return windows, y_prob


if __name__ == "__main__":
    pipeline_path = pkg_resources.resource_filename("name_detector", "checkpoints/pipeline.joblib")
    cb_model_path = pkg_resources.resource_filename("name_detector", "checkpoints/catboost_model.cbm")

    name_detector = NameDetector(pipeline_path, cb_model_path)
    text = (
        "Imsol soli xub boshad. Sardor vazir boshad. Eraj Boqiev salomat boshad. Karomatullo Hoshimiyon zinda boshad."
    )
    windows, y_prob = name_detector.predict(text)

    for window, prob in zip(windows, y_prob):
        print(f"{window:30s}  {prob[1]:.3f}")
