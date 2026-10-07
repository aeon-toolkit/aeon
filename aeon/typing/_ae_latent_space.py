from enum import StrEnum, auto, unique


@unique
class LATENT_SPACE(StrEnum):
    FLAT = auto()
    TIME = auto()
    REPEATED = auto()

    @staticmethod
    def exists(latent_space: str) -> bool:
        """Check if the given latent_space exists in the LATENT_SPACE enum."""
        return latent_space.lower() in (item.value for item in LATENT_SPACE)

    @staticmethod
    def _check_param(latent_space: str):
        if not LATENT_SPACE.exists(latent_space):
            raise ValueError(
                f"Invalid value for 'latent_space' ({latent_space}). "
                f"Valid options are: {[item.value for item in LATENT_SPACE]}"
            )
        return LATENT_SPACE[latent_space.upper()]
