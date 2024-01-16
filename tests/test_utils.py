import pytest

from name_detector.utils import is_cyrillic, latinize_text


def test_basic_latinization():
    assert latinize_text("Привет") == "Privet"
    assert latinize_text("Салом") == "Salom"


def test_latinization_with_tajik_characters():
    assert latinize_text("Ғ") == "Gh"
    assert latinize_text("Ӯ") == "U"


def test_empty_string():
    assert latinize_text("") == ""


def test_null_input():
    with pytest.raises(TypeError):
        latinize_text(None)


def test_numerical_input():
    with pytest.raises(TypeError):
        latinize_text(123)


def test_latinization_case_sensitivity():
    assert latinize_text("привет") == "privet"
    assert latinize_text("Привет") == "Privet"


mixed_content_cases = [
    ("Привет, как дела?", "Privet, kak dela?"),
    ("Шокиров ғайраталӣ", "Shokirov ghayratali"),
    ("Таджикистан", "Tadzhikistan"),
    ("Борис Ельцин", "Boris Eltsin"),
    ("Москва 2024", "Moskva 2024"),
    ("Достоевский и Толстой", "Dostoevskiy i Tolstoy"),
    ("Красная площадь", "Krasnaya ploshchad"),
    ("Компьютерные науки", "Kompyuternye nauki"),
    ("Фёдор Михайлович", "Fyodor Mikhaylovich"),
    ("Санкт-Петербург", "Sankt-Peterburg"),
    ("Царь колокол", "Tsar kolokol"),
    ("Театр Большой", "Teatr Bolshoy"),
    ("Владимир Путин", "Vladimir Putin"),
    ("Железная дорога", "Zheleznaya doroga"),
    ("Программирование", "Programmirovanie"),
    ("Живопись и искусство", "Zhivopis i iskusstvo"),
    ("Экономика России", "Ekonomika Rossii"),
    ("Космическая станция", "Kosmicheskaya stantsiya"),
    ("Культурное наследие", "Kulturnoe nasledie"),
    ("Зимний дворец", "Zimniy dvorets"),
    ("Хлеб и соль", "Khleb i sol"),
    ("Журналистика и медиа", "Zhurnalistika i media"),
]
tajik_content_cases = [
    ("Душанбе", "Dushanbe"),
    ("Панҷакент", "Panjakent"),
    ("Исмоили Сомонӣ", "Ismoili Somoni"),
    ("Тоҷикистон", "Tojikiston"),
    ("Қӯрғонтеппа", "Qurghonteppa"),
    ("Ҳисор", "Hisor"),
    ("Кӯли Кулоб", "Kuli Kulob"),
    ("Ҷомеъаи Тоҷик", "Jomeai Tojik"),
    ("Рӯдакӣ", "Rudaki"),
    ("Фирдавсӣ", "Firdavsi"),
    ("Ғазали Ҷалолуддин", "Ghazali Jaloluddin"),
    ("Садриддин Айнӣ", "Sadriddin Ayni"),
    ("Наврӯз", "Navruz"),
    ("Сомониён", "Somoniyon"),
    ("Мирзо Турсунзода", "Mirzo Tursunzoda"),
    ("Сино", "Sino"),
    ("Хоруғ", "Khorugh"),
    ("Ғафуров", "Ghafurov"),
    ("Ваҳдат", "Vahdat"),
    ("Ҳунарманди тоҷик", "Hunarmandi tojik"),
]

# Extending the mixed_content_cases with Tajik test cases
mixed_content_cases.extend(tajik_content_cases)


# Test cases for non-Cyrillic content
non_cyrillic_cases = [
    ("Hello", "Hello"),
    ("123", "123"),
    ("!@#", "!@#"),
    ("Test 456", "Test 456"),
    ("Simple Text", "Simple Text"),
    ("[Bracketed]", "[Bracketed]"),
    ("New-Line\nText", "New-Line\nText"),
    ("Tabs\tAre\tHere", "Tabs\tAre\tHere"),
    ("SpecialChars_*&^%$", "SpecialChars_*&^%$"),
    ("Mixed123Numbers", "Mixed123Numbers"),
    ("UPPERCASE", "UPPERCASE"),
    ("lowercase", "lowercase"),
    ("CamelCaseText", "CamelCaseText"),
    ("snake_case_text", "snake_case_text"),
    ("kebab-case-text", "kebab-case-text"),
    ("Dot.separated.text", "Dot.separated.text"),
    ("Path/To/File.txt", "Path/To/File.txt"),
    ("Email@example.com", "Email@example.com"),
    ("URL: http://example.com", "URL: http://example.com"),
    ("'Quoted Text'", "'Quoted Text'"),
]


@pytest.mark.parametrize("input_text,expected", mixed_content_cases)
def test_latinization_with_mixed_content(input_text, expected):
    assert latinize_text(input_text) == expected


@pytest.mark.parametrize("input_text,expected", non_cyrillic_cases)
def test_latinization_with_non_cyrillic(input_text, expected):
    assert latinize_text(input_text) == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("Салом", True),
        ("Салом алекум", True),
        ("Салом alekum", False),
        ("Салом 123", True),
        ("Салом 123!! ..", True),
        (" ", True),
    ],
)
def test_is_cyrillic(input_text, expected):
    assert is_cyrillic(input_text) == expected
