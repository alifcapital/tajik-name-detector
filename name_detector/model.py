from catboost import CatBoostClassifier


class CatBoostModel:
    def __init__(self, config):
        # Initialize CatBoostClassifier
        self._model = CatBoostClassifier(**config)

        self.predict = self._model.predict
        self.predict_proba = self._model.predict_proba
        self.fit = self._model.fit

    def save(self, filename: str):
        self._model.save_model(filename)

    @classmethod
    def init_from(cls, filename: str):
        """
        Initialize a CatBoostModel object from a saved file.

        :param filename: The name of the file to load the object state from.
        :return: An instance of CatBoostModel initialized with the saved state.
        """

        instance = cls(config={})
        instance._model.load_model(filename)
        return instance
