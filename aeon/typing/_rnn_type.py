from enum import StrEnum, auto, unique


@unique
class RNN_TYPE(StrEnum):
    LSTM = auto()
    GRU = auto()
    SIMPLE = auto()

    @staticmethod
    def exists(rnn_type: str) -> bool:
        """Check if the given rnn_type exists in the RNN_TYPE enum."""
        return rnn_type.lower() in (item.value for item in RNN_TYPE)

    @staticmethod
    def _check_params(rnn_type: str):
        if not RNN_TYPE.exists(rnn_type):
            raise ValueError(
                f"Invalid value for 'rnn_type' ({rnn_type}). "
                f"Valid options are: {[item.value for item in RNN_TYPE]}"
            )
        return RNN_TYPE[rnn_type.upper()]

    def __call__(self, **kwargs):
        import tensorflow as tf

        match self:
            case RNN_TYPE.LSTM:
                return tf.keras.layers.LSTM(**kwargs)
            case RNN_TYPE.GRU:
                return tf.keras.layers.GRU(**kwargs)
            case RNN_TYPE.SIMPLE:
                return tf.keras.layers.SimpleRNN(**kwargs)
