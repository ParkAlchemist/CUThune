from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints
import tomllib

from vocoder.src.nes_vocoder.config.schema import (
    AMPConfig,
    AugmentationConfig,
    ControlConfig,
    LogConfig,
    OptimizerConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
    VocoderDataConfig,
    VocoderDiscriminatorConfig,
    VocoderExperimentConfig,
    VocoderGeneratorModelConfig,
    VocoderLossConfig,
)


T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration file contains invalid values."""


def _format_path(path: str, key: str) -> str:
    if not path:
        return key

    return f"{path}.{key}"


def _is_union(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in {Union, UnionType}


def _convert_scalar(
    value: Any,
    expected_type: type,
    *,
    path: str,
) -> Any:
    if expected_type is bool:
        if not isinstance(value, bool):
            raise ConfigError(
                f"{path} must be a Boolean, got "
                f"{type(value).__name__}: {value!r}"
            )

        return value

    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(
                f"{path} must be an integer, got "
                f"{type(value).__name__}: {value!r}"
            )

        return value

    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"{path} must be a number, got "
                f"{type(value).__name__}: {value!r}"
            )

        return float(value)

    if expected_type is str:
        if not isinstance(value, str):
            raise ConfigError(
                f"{path} must be a string, got "
                f"{type(value).__name__}: {value!r}"
            )

        return value

    if not isinstance(value, expected_type):
        raise ConfigError(
            f"{path} must be {expected_type.__name__}, got "
            f"{type(value).__name__}: {value!r}"
        )

    return value


def _convert_value(
    value: Any,
    annotation: Any,
    *,
    path: str,
) -> Any:
    """
    Convert a TOML value into the type used by a dataclass field.

    Supported constructs include:
        int
        float
        bool
        str
        Literal
        Optional / unions
        tuple[T, ...]
        tuple[tuple[T, ...], ...]
        nested dataclasses
    """
    if annotation is Any:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        if value not in args:
            allowed = ", ".join(repr(item) for item in args)

            raise ConfigError(
                f"{path} must be one of: {allowed}. "
                f"Got {value!r}."
            )

        return value

    if _is_union(annotation):
        if value is None and type(None) in args:
            return None

        conversion_errors: list[str] = []

        for member_type in args:
            if member_type is type(None):
                continue

            try:
                return _convert_value(
                    value,
                    member_type,
                    path=path,
                )
            except ConfigError as exc:
                conversion_errors.append(str(exc))

        raise ConfigError(
            f"{path} does not match any allowed type. "
            f"Got {value!r}. Details: "
            + " | ".join(conversion_errors)
        )

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(
                f"{path} must be a TOML array, got "
                f"{type(value).__name__}: {value!r}"
            )

        if len(args) == 2 and args[1] is Ellipsis:
            item_type = args[0]

            return tuple(
                _convert_value(
                    item,
                    item_type,
                    path=f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            )

        if len(value) != len(args):
            raise ConfigError(
                f"{path} must contain exactly {len(args)} items. "
                f"Got {len(value)}."
            )

        return tuple(
            _convert_value(
                item,
                item_type,
                path=f"{path}[{index}]",
            )
            for index, (item, item_type) in enumerate(
                zip(value, args)
            )
        )

    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigError(
                f"{path} must be a TOML table, got "
                f"{type(value).__name__}: {value!r}"
            )

        return dataclass_from_dict(
            annotation,
            value,
            path=path,
        )

    if annotation in {bool, int, float, str}:
        return _convert_scalar(
            value,
            annotation,
            path=path,
        )

    if isinstance(annotation, type):
        return _convert_scalar(
            value,
            annotation,
            path=path,
        )

    raise ConfigError(
        f"Unsupported type annotation for {path}: {annotation!r}"
    )


def dataclass_from_dict(
    cls: type[T],
    data: dict[str, Any],
    *,
    path: str,
) -> T:
    """
    Construct a dataclass from a dictionary with strict key validation.

    Unknown keys are rejected instead of silently ignored.
    Missing keys retain their dataclass defaults.
    """
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path or cls.__name__} must be a TOML table."
        )

    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type.")

    field_definitions = {
        field_info.name: field_info
        for field_info in fields(cls)
    }

    unknown_keys = sorted(
        set(data) - set(field_definitions)
    )

    if unknown_keys:
        formatted = ", ".join(
            _format_path(path, key)
            for key in unknown_keys
        )

        raise ConfigError(
            f"Unknown configuration key(s): {formatted}"
        )

    # get_type_hints resolves annotations correctly despite:
    # from __future__ import annotations
    type_hints = get_type_hints(cls)

    kwargs: dict[str, Any] = {}

    for name, value in data.items():
        field_path = _format_path(path, name)
        annotation = type_hints.get(name, Any)

        kwargs[name] = _convert_value(
            value,
            annotation,
            path=field_path,
        )

    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(
            f"Could not construct configuration section "
            f"{path or cls.__name__}: {exc}"
        ) from exc

ROOT_SECTIONS = {
    "schema_version",
    "run",
    "data",
    "generator",
    "discriminator",
    "loss",
    "optimizer",
    "scheduler",
    "training",
    "amp",
    "augmentation",
    "logging",
    "control",
}


def _require_table(
    raw: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = raw.get(key, {})

    if not isinstance(value, dict):
        raise ConfigError(
            f"[{key}] must be a TOML table."
        )

    return value


def _validate_nested_container_keys(
    table: dict[str, Any],
    *,
    section: str,
    allowed: set[str],
) -> None:
    unknown = sorted(set(table) - allowed)

    if unknown:
        formatted = ", ".join(
            f"{section}.{key}"
            for key in unknown
        )

        raise ConfigError(
            f"Unknown configuration key(s): {formatted}"
        )


def load_vocoder_config(
    path: Path,
) -> VocoderExperimentConfig:
    """
    Load and validate a components experiment TOML file.

    Expected nested sections include:

        [optimizer.generator]
        [optimizer.discriminator]
        [scheduler.generator]
        [scheduler.discriminator]
    """
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Vocoder config file not found: {path}"
        )

    if not path.is_file():
        raise ConfigError(
            f"Vocoder config path is not a file: {path}"
        )

    try:
        with path.open("rb") as file:
            raw = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Invalid TOML in {path}: {exc}"
        ) from exc

    unknown_root_sections = sorted(
        set(raw) - ROOT_SECTIONS
    )

    if unknown_root_sections:
        raise ConfigError(
            "Unknown top-level configuration section(s): "
            + ", ".join(unknown_root_sections)
        )

    optimizer_raw = _require_table(raw, "optimizer")
    scheduler_raw = _require_table(raw, "scheduler")

    _validate_nested_container_keys(
        optimizer_raw,
        section="optimizer",
        allowed={"generator", "discriminator"},
    )

    _validate_nested_container_keys(
        scheduler_raw,
        section="scheduler",
        allowed={"generator", "discriminator"},
    )

    generator_optimizer_raw = optimizer_raw.get(
        "generator",
        {},
    )
    discriminator_optimizer_raw = optimizer_raw.get(
        "discriminator",
        {},
    )

    generator_scheduler_raw = scheduler_raw.get(
        "generator",
        {},
    )
    discriminator_scheduler_raw = scheduler_raw.get(
        "discriminator",
        {},
    )

    if not isinstance(generator_optimizer_raw, dict):
        raise ConfigError(
            "[optimizer.generator] must be a TOML table."
        )

    if not isinstance(discriminator_optimizer_raw, dict):
        raise ConfigError(
            "[optimizer.discriminator] must be a TOML table."
        )
    if not isinstance(generator_scheduler_raw, dict):
        raise ConfigError(
            "[scheduler.generator] must be a TOML table."
        )

    if not isinstance(discriminator_scheduler_raw, dict):
        raise ConfigError(
            "[scheduler.discriminator] must be a TOML table."
        )

    schema_version = raw.get("schema_version", 1)

    if isinstance(schema_version, bool) or not isinstance(
        schema_version,
        int,
    ):
        raise ConfigError(
            "schema_version must be an integer."
        )

    config = VocoderExperimentConfig(
        schema_version=schema_version,
        run=dataclass_from_dict(
            RunConfig,
            _require_table(raw, "run"),
            path="run",
        ),
        data=dataclass_from_dict(
            VocoderDataConfig,
            _require_table(raw, "data"),
            path="data",
        ),
        generator=dataclass_from_dict(
            VocoderGeneratorModelConfig,
            _require_table(raw, "generator"),
            path="generator",
        ),
        discriminator=dataclass_from_dict(
            VocoderDiscriminatorConfig,
            _require_table(raw, "discriminator"),
            path="discriminator",
        ),
        loss=dataclass_from_dict(
            VocoderLossConfig,
            _require_table(raw, "loss"),
            path="loss",
        ),
        optimizer_generator=dataclass_from_dict(
            OptimizerConfig,
            generator_optimizer_raw,
            path="optimizer.generator",
        ),
        optimizer_discriminator=dataclass_from_dict(
            OptimizerConfig,
            discriminator_optimizer_raw,
            path="optimizer.discriminator",
        ),
        scheduler_generator=dataclass_from_dict(
            SchedulerConfig,
            generator_scheduler_raw,
            path="scheduler.generator",
        ),
        scheduler_discriminator=dataclass_from_dict(
            SchedulerConfig,
            discriminator_scheduler_raw,
            path="scheduler.discriminator",
        ),
        training=dataclass_from_dict(
            TrainingConfig,
            _require_table(raw, "training"),
            path="training",
        ),
        amp=dataclass_from_dict(
            AMPConfig,
            _require_table(raw, "amp"),
            path="amp",
        ),
        augmentation=dataclass_from_dict(
            AugmentationConfig,
            _require_table(raw, "augmentation"),
            path="augmentation",
        ),
        logging=dataclass_from_dict(
            LogConfig,
            _require_table(raw, "logging"),
            path="logging",
        ),
        control=dataclass_from_dict(
            ControlConfig,
            _require_table(raw, "control"),
            path="control",
        ),
    )

    from vocoder.src.nes_vocoder.config.validator import (
        validate_vocoder_config,
    )

    validate_vocoder_config(
        config,
        check_paths=False,
    )

    return config

