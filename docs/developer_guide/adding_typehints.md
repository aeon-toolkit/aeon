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



When working with soft dependencies (optional packages that are used for specific purposes like type-checking or additional features), special care must be taken to ensure that these dependencies don't interfere with runtime execution. A soft dependency might be unavailable during runtime but necessary for type annotations or certain features.

### Using TYPE_CHECKING for Conditional Imports

To ensure that soft dependencies are imported only during type-checking and not at runtime, the TYPE_CHECKING constant from typing can be used. This allows type hints to be safely referenced without causing errors if the dependency is not installed at runtime.

Example:

```
from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from pyod.models.base import BaseDetector
    
```

In this code, the BaseDetector class is conditionally imported only when type-checking is performed. This prevents errors during runtime if the pyod package is not installed, making the code safe even when the soft dependency is not available.
 
 ### Future Annotations for Forward References

Using ```from __future__ import annotations``` enables type hints to reference classes that might not yet be defined or to use string-literal annotations. This feature ensures that type annotations can still be written safely and allows forward references in cases where classes or functions are not defined at the time of the annotation.

```python
from __future__ import annotations
```

This import ensures that all annotations are stored as string literals, allowing forward references and improving the flexibility of type hints.
 
 
 

#### Static Type Checking Tools  

- **mypy**: A popular static type checker for Python.  
  - **Install**: `pip install mypy`  
  - **Run**: `mypy your_file.py`  

