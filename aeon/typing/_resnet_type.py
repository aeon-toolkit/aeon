from enum import StrEnum, auto, unique


@unique
class RESNET_TYPE(StrEnum):
    V1 = auto()
    V2 = auto()

    @staticmethod
    def exists(resnet_type: str) -> bool:
        """Check if the given resnet_type exists in the RESNET_TYPE enum."""
        return resnet_type.lower() in (item.value for item in RESNET_TYPE)

    @staticmethod
    def _check_params(resnet_type: str):
        if not RESNET_TYPE.exists(resnet_type):
            raise ValueError(
                f"Invalid value for 'resnet_type' ({resnet_type}). "
                f"Valid options are: {[item.value for item in RESNET_TYPE]}"
            )
        return RESNET_TYPE[resnet_type.upper()]
