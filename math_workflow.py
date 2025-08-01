from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

# pyflyte run --remote math_workflow.py math_workflow --a 10 --b 6
@workflow
def math_workflow(a: int, b: int) -> float:

    # Create first task
    add_task = DominoJobTask(
        name='Add numbers',
        domino_job_config=DominoJobConfig(Command="python /mnt/code/add.py"),
        inputs={'first_value': int, 'second_value': int},
        outputs={'sum': int},
        use_latest=True
    )
    sum = add_task(first_value=a, second_value=b)

    # Create second task
    sqrt_task = DominoJobTask(
        name='Square root',
        domino_job_config=DominoJobConfig(Command="python /mnt/code/sqrt.py"),
        inputs={'value': int},
        outputs={'sqrt': float},
        use_latest=True
    )
    sqrt = sqrt_task(value=sum)

    return sqrt
