# Adding Type Hints

Type hints are a way to indicate the expected data types of variables, function
parameters, and return values. They enhance code readability and help with static
type checking, making it easier to catch errors.

For example, here is a simple function whose argument and return type are declared
in the annotations:

```python
from typing import List

def sum_ints_return_str(int_list: List[int]) -> str:
    return str(sum(int_list))
```

Type hints are not currently mandatory in `aeon`, but we aim to progressively integrate
them into the code base. Learn more about type hints in the
[Python documentation](https://docs.python.org/3/library/typing.html)
and [PEP 484](https://peps.python.org/pep-0484/).

## Soft Dependency Type Hints

When working with modules that use soft dependencies, additional considerations are
required to ensure that your code can still run even without these dependencies
installed.

Here is an example snippet taken from [PyODAdapter](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.anomaly_detection.PyODAdapter.html).
It uses the `pyod` library, which is a soft dependency. The `TYPE_CHECKING` constant
is used to ensure that the `pyod` library is only imported at the top level while type
checking is performed. `from __future__ import annotations` is used to allow forward
references in type hints. See [PEP 563](https://peps.python.org/pep-0563/) for more
information. The `pyod` `BaseDetector` class can now be used in type hints with
these additions.

 ```python
from __future__ import annotations

from aeon.anomaly_detection.base import BaseAnomalyDetector
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyod.models.base import BaseDetector

class PyODAdapter(BaseAnomalyDetector):
    def __init__(
        self, pyod_model: BaseDetector, window_size: int = 10, stride: int = 1
    ):
        self.pyod_model = pyod_model
        self.window_size = window_size
        self.stride = stride

        super().__init__(axis=0)
```
