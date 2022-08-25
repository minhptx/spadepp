import os
import random
import shutil
import sys

import click
from loguru import logger

from spader.evaluation.spader import SpaderEvaluation

config = {
    "handlers": [
        {"sink": sys.stdout, "format": "{time} - {message}", "level": "INFO"},
        {
            "sink": "logs/error.log",
            "format": "{time} - {message}",
            "level": "ERROR",
            "backtrace": True,
            "diagnose": True,
        }
    ]
}


logger.configure(**config)

logger.info(
    "=========================================================================="
)

random.seed(1811)

import warnings

warnings.filterwarnings('ignore')


def prepare_environment():
    if not os.path.exists("debug"):
        os.mkdir("debug")



@click.group()
def cli():
    pass


@cli.command()
@click.option("--data_path", "-d", help="Path to dataset")
@click.option("--output_path", "-o", help="Path to output directory", default="output")
@click.option("--num_examples", "-n", help="Number of examples to use", default=20)

def evaluate(data_path, num_examples, output_path):

    prepare_environment()
    evaluator = SpaderEvaluation()
    evaluator.evaluate(data_path, num_examples, output_path)


@cli.command()
@click.option("--dataset", "-d", help="Dataset name")
@click.option("--num_examples", "-n", help="Number of examples to use", default=20)

def run(dataset, num_examples):
    output_path = f"reports/{dataset}"
    data_path = f"data/{dataset}"
    prepare_environment()
    evaluator = SpaderEvaluation()
    evaluator.evaluate(data_path, num_examples, output_path, 1)


@cli.command()
def clear():
    shutil.rmtree("output", ignore_errors=True)
    shutil.rmtree("debug", ignore_errors=True)
    os.remove("info.log")
    os.remove("error.log")
    os.remove("debug.log")



if __name__ == "__main__":
    cli()
