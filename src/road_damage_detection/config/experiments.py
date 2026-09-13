from dataclasses import dataclass

from road_damage_detection.training.mean_subtraction import (
    MeanSubtractionPredictor,
    MeanSubtractionTrainer,
    MeanSubtractionValidator,
)

from road_damage_detection.training.standardization import (
    StandardizationPredictor,
    StandardizationTrainer,
    StandardizationValidator
)

@dataclass(frozen=True)
class Experiment:
    name: str
    run_suffix: str | None = None
    trainer: type | None = None
    validator: type | None = None
    predictor: type | None = None


EXPERIMENTS = {
    "none": Experiment(
        name="none"
    ),
    "mean-subtraction": Experiment(
        name="mean-subtraction",
        run_suffix="mean_subtraction",
        trainer=MeanSubtractionTrainer,
        validator=MeanSubtractionValidator,
        predictor=MeanSubtractionPredictor,
    ),
    "standardization" : Experiment(
        name = "standardization",
        run_suffix = "standardization",
        trainer = StandardizationTrainer,
        validator = StandardizationValidator,
        predictor = StandardizationPredictor
    ),
}


def get_experiment(
    experiment_name
):
    return EXPERIMENTS[
        experiment_name
    ]


def resolve_experiment(
    experiment_name=None,
    mean_subtraction=False
):

    if mean_subtraction:

        if experiment_name not in (
            None,
            "none",
            "mean-subtraction",
        ):
            raise ValueError(
                "--mean-subtraction conflicts with "
                f"--experiment {experiment_name}"
            )

        experiment_name = "mean-subtraction"

    if experiment_name is None:
        experiment_name = "none"

    return get_experiment(
        experiment_name
    )


def add_experiment_arguments(
    parser
):

    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS.keys(),
        default=None
    )

    parser.add_argument(
        "--mean-subtraction",
        action="store_true",
        help=(
            "Backward-compatible alias for "
            "--experiment mean-subtraction"
        )
    )
