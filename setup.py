from setuptools import find_packages, setup

setup(
    name="name_detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "catboost",
        "scipy",
        "joblib",
        "numpy",
        "tqdm",
    ],
    package_data={
        "name_detector": ["checkpoints/*"],
    },
    author="Sobir Bobiev",
    author_email="sobir.bobiev@gmail.com",
    description="A simple fullname detector primarily built for Tajik names.",
)
