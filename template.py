import os

files = [
    "src/__init__.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",

    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",

    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/prediction_pipeline.py",

    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",

    "src/config/__init__.py",
    "src/config/paths.py",

    "templates/index.html",
    "templates/result.html",

    "app.py",
    "setup.py",
    "requirements.txt",
    "README.md",
    ".gitignore"
]

for file in files:
    filedir = os.path.dirname(file)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if not os.path.exists(file):
        with open(file, "w") as f:
            pass

print("Project Structure Created")