# Adding Type Hints

## Introduction to Type Hints

Type hints are a way to indicate the expected data types of variables, function parameters, and return values in Python. They enhance code readability and help with static type checking, making it easier to catch errors before runtime.


Type hints act as a form of documentation that helps developers understand the types of arguments a function expects and what it returns.


## Basic Syntax

For example, here is a simple function whose argument and return type are declared in the annotations:


```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```


Learn more about type hints in [python docs](https://docs.python.org/3/library/typing.html) and [PEP 484](https://peps.python.org/pep-0484/)


# Dealing with Soft Dependency Type Hints



When working with models that have soft dependencies, additional considerations are required to ensure that your code remains robust and maintainable. Soft dependencies are optional packages or modules that your application does not require at runtime but may be used in specific situations, such as during type-checking or when certain features are enabled.

 The typing.TYPE_CHECKING constant ensures that imports for type hints are only evaluated when type-checking is done and NOT in the runtime. This prevents errors when the soft dependancies are not available. Here is an example that of [PyODAdapter](https://github.com/aeon-toolkit/aeon/blob/main/aeon/anomaly_detection/_pyodadapter.py):


 ```python
from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.validation._dependencies import _check_soft_dependencies
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pyod.models.base import BaseDetector


class PyODAdapter(BaseAnomalyDetector):
    ...

    def _is_pyod_model(model: Any) -> bool:
        """Check if the provided model is a PyOD model."""
        from pyod.models.base import BaseDetector

        return isinstance(model, BaseDetector)
   ...
```
