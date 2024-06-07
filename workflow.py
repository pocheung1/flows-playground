from utils.flyte import DominoTask, Input, Output
from flytekit import workflow
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import TypeVar, List, Dict
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

DSE = "Domino Standard Environment Py3.10 R4.4"


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

    prepare_data = DominoJobTask(
        name="Prepare data",
        domino_job_config=DominoJobConfig(
            Command="python /mnt/code/scripts/prep-data.py",
        ),
        inputs={'data_path': str},
        outputs={'processed_data': FlyteFile},
        use_latest=True,
    )
    prepare_data_results = prepare_data(data_path=data_path)

    train_model = DominoJobTask(
        name="Train model",
        domino_job_config=DominoJobConfig(
            Command="python /mnt/code/scripts/train-model.py",
        ),
        inputs={
            'processed_data': FlyteFile,
            'epochs': int,
            'batch_size': int,
        },
        outputs={
            'model': FlyteFile,
        },
        use_latest=True,
    )
    train_model_results = train_model(
        processed_data=prepare_data_results['processed_data'],
        epochs=10,
        batch_size=32,
    )

    return train_model_results['model']


@workflow
def training_subworkflow(data_path: str) -> FlyteFile:

    prepare_data = DominoJobTask(
        name="Prepare data",
        domino_job_config=DominoJobConfig(
            Command="python /mnt/code/scripts/prep-data.py",
        ),
        inputs={'data_path': str},
        outputs={'processed_data': FlyteFile},
        use_latest=True,
    )
    prepare_data_results = prepare_data(data_path=data_path)

    train_model = DominoJobTask(
        name="Train model",
        domino_job_config=DominoJobConfig(
            Command="python /mnt/code/scripts/train-model.py",
        ),
        inputs={
            'processed_data': FlyteFile,
            'epochs': int,
            'batch_size': int,
        },
        outputs={
            'model': FlyteFile,
        },
        use_latest=True,
    )
    train_model_results = train_model(
        processed_data=prepare_data_results['processed_data'],
        epochs=10,
        batch_size=32,
    )

    return train_model_results['model']


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


# pyflyte run --remote workflow.py training_workflow_nested --data_path /mnt/code/artifacts/data.csv
@workflow
def training_workflow_nested(data_path: str): 

    model = training_subworkflow(data_path=data_path)

    training_task = DominoJobTask(
        name="Final task",
        domino_job_config=DominoJobConfig(
            Command="sleep 100",
        ),
        inputs={'model': FlyteFile},
        use_latest=True,
    )
    return training_task(model=model)
