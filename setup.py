from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    #requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]
    requirements = [
        line.strip() for line in f.readlines()
        if line.strip() and not line.startswith("git+")
    ]

setup(
    name="lnco-transcribe",  # CLI tool name
    version="1.0.0",
    packages=find_packages(include=["src", "src.*"]),  # Include your `src/` package
    install_requires=[r for r in requirements if not r.startswith("git+")],  # Install dependencies from requirements.txt
    entry_points={
        "console_scripts": [
            "lnco-transcribe = run_diarize:main",  # Run `main.py`'s `main()` function
        ],
    },
)
