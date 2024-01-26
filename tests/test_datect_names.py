import pytest

from name_detector.detect_names import NameDetector


class TestNameDetector:
    name_detector = NameDetector()

    @pytest.mark.parametrize(
        "text",
        [
            "Телефон",
            "Сардор",
            "44",
            "Ааааа",
        ],
    )
    def test_single_words(self, text):
        windows, y_prob = self.name_detector.predict(text)
        assert len(windows) == 0
        assert len(y_prob) == 0

    @pytest.mark.parametrize(
        "text",
        [
            "Рустами Фарҳод",
            "Сардор Комронов",
            "Гулрӯи Фаридун",
            "Гулрӯ Фаридунова",
            "Гулрӯ Фаридунова Парвизович",
        ],
    )
    def test_names(self, text):
        windows, y_prob = self.name_detector.predict(text)
        assert len(windows) == len(y_prob) > 0
        assert max(y_prob) > 0.5

    @pytest.mark.parametrize(
        "text",
        [
            "рустами фарҳод",
            "сардор комронов",
            "гулрӯи фаридун",
            "гулрӯ фаридунова",
            "гулрӯ фаридунова парвизовна",
        ],
    )
    def test_lowercase_names(self, text):
        windows, y_prob = self.name_detector.predict(text)
        assert len(windows) == len(y_prob) > 0
        assert max(y_prob) > 0.5

    @pytest.mark.parametrize(
        "text",
        [
            "rustami farhod",
            "sardor komronov",
            "gulrui faridun",
            "gulru faridunova",
            "gulru faridunova parvizovna",
        ],
    )
    def test_lowercase_latin_names(self, text):
        windows, y_prob = self.name_detector.predict(text)
        assert len(windows) == len(y_prob) > 0
        assert max(y_prob) > 0.5

    @pytest.mark.parametrize(
        "text",
        [
            "Боғи Марказӣ",
            "shahri Dushanbe",
            "Корти Салом",
            "Корти Виза",
            "Alif Mobi",
            "Алиф Моби",
        ],
    )
    def test_not_names(self, text):
        windows, y_prob = self.name_detector.predict(text)
        assert len(windows) == len(y_prob) > 0
        assert max(y_prob) < 0.3
