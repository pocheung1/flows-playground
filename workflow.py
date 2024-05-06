from utils.flyte import DominoTask, Input, Output
from flytekit import workflow
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import TypeVar, List, Dict

DSE = "Domino Standard Environment Py3.9 R4.3"


@workflow
def training_workflow(data_path: str) -> FlyteFile: 
    """
    Sample data preparation and training workflow

    This workflow accepts a path to a CSV for some initial input and simulates
    the processing of the data and usage of the processed data in a training job.

    To run this workflowp, execute the following line in the terminal

    pyflyte run --remote workflow.py training_workflow --data_path /mnt/code/artifacts/data.csv

    :param data_path: Path of the CSV file data
    :return: The training results as a model
    """

    data_prep_results = DominoTask(
        name="Prepare data",
        command="python /mnt/code/scripts/prep-data.py",
        environment=DSE,
        hardware_tier="Small",
        inputs=[
            Input(name="data_path", type=str, value=data_path),
        ],
        outputs=[
            Output(name="processed_data", type=FlyteFile)
        ]
    )

    training_results = DominoTask(
        name="Train model",
        command="python /mnt/code/scripts/train-model.py",
        environment=DSE,
        hardware_tier="Medium",
        inputs=[
            Input(name="processed_data", type=FlyteFile, value=data_prep_results['processed_data']),
            Input(name="epochs", type=int, value=10),
            Input(name="batch_size", type=int, value=32)
        ],
        outputs=[
            Output(name="model", type=FlyteFile)
        ]
    )

    return training_results['model']

@workflow
def training_subworkflow(data_path: str) -> FlyteFile: 

    data_prep_results = DominoTask(
        name="Prepare data",
        command="python /mnt/code/scripts/prep-data.py",
        environment=DSE,
        hardware_tier="Small",
        inputs=[
            Input(name="data_path", type=str, value=data_path),
        ],
        outputs=[
            Output(name="processed_data", type=FlyteFile)
        ]
    )

    training_results = DominoTask(
        name="Train model",
        command="python /mnt/code/scripts/train-model.py",
        environment=DSE,
        hardware_tier="Medium",
        inputs=[
            Input(name="processed_data", type=FlyteFile, value=data_prep_results['processed_data']),
            Input(name="epochs", type=int, value=10),
            Input(name="batch_size", type=int, value=32)
        ],
        outputs=[
            Output(name="model", type=FlyteFile)
        ]
    )

    return training_results['model']

# pyflyte run --remote workflow.py generate_types 
@workflow
def generate_types():
    sce_types = DominoTask(
        name="Generate SCE Types",
        command="python /mnt/code/scripts/generate-sce-types.py",
        environment=DSE,
        hardware_tier="Small",
        inputs=[
            Input(name="sdtm_data_path", type=str, value="/some/path/to/data")
        ],
        outputs=[
            Output(name="pdf", type=FlyteFile[TypeVar("pdf")]),
            Output(name="sas7bdat", type=FlyteFile[TypeVar("sas7bdat")])
        ]
    )

    ml_types = DominoTask(
        name="Generate ML Types",
        command="python /mnt/code/scripts/generate-ml-types.py",
        environment=DSE,
        hardware_tier="Small",
        inputs=[
            Input(name="batch_size", type=int, value=32),
            Input(name="learning_rate", type=float, value=0.001),
            Input(name="do_eval", type=bool, value=True),
            Input(name="list", type=List[int], value=[1,2,3,4,5]),
            Input(
                name="dict", 
                type=Dict, 
                value={
                    'param1': 10, 
                    "param2": {
                        "a": 4,
                        "b": {
                            "x": True
                        }
                    }
                })
        ],
        outputs=[
            Output(name="csv", type=FlyteFile[TypeVar("csv")]),
            Output(name="json", type=FlyteFile[TypeVar("json")]),
            Output(name="png", type=FlyteFile[TypeVar("png")]),
            Output(name="jpeg", type=FlyteFile[TypeVar("jpeg")]),
            Output(name="notebook", type=FlyteFile[TypeVar("ipynb")]),
            Output(name="mlflow_model", type=FlyteDirectory)
        ]
    )

    return 


# pyflyte run --remote workflow.py training_workflow --data_path /mnt/code/artifacts/data.csv
@workflow
def training_workflow_git(data_path: str) -> FlyteFile: 
    """
    Sample data preparation and training workflow

    This workflow accepts a path to a CSV for some initial input and simulates
    the processing of the data and usage of the processed data in a training job.

    To run this workflowp, execute the following line in the terminal

    pyflyte run --remote workflow.py training_workflow --data_path /mnt/code/artifacts/data.csv

    :param data_path: Path of the CSV file data
    :return: The training results as a model
    """

    data_prep_results = DominoTask(
        name="Prepare data ",
        command="python /mnt/code/scripts/prep-data.py",
        environment=DSE,
        hardware_tier="Small",
        inputs=[
            Input(name="data_path", type=str, value=data_path)
        ],
        outputs=[
            Output(name="processed_data", type=FlyteFile)
        ]
    )

    training_results = DominoTask(
        name="Train model ",
        command="python /mnt/code/scripts/train-model.py",
        environment=DSE,
        hardware_tier="Medium",
        inputs=[
            Input(name="processed_data", type=FlyteFile, value=data_prep_results['processed_data']),
            Input(name="epochs", type=int, value=10),
            Input(name="batch_size", type=int, value=32)
        ],
        outputs=[
            Output(name="model", type=FlyteFile)
        ]
    )

    return training_results['model']


# pyflyte run --remote workflow.py training_workflow_nested --data_path /mnt/code/artifacts/data.csv
@workflow
def training_workflow_nested(data_path: str): 

    model = training_subworkflow(data_path=data_path)

    results = DominoTask(
        name="Final task",
        command="sleep 100",
        environment=DSE,
        hardware_tier="Medium",
        inputs=[
            Input(name="model", type=FlyteFile, value=model)
        ]
    )

    return 
