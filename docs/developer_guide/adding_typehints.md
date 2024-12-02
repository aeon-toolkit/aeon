# Adding Type Hints

## Introduction to Type Hints

Type hints are a way to indicate the expected data types of variables, function parameters, and return values in Python. They enhance code readability and help with static type checking, making it easier to catch errors before runtime. For more information, refer to [PEP 563](https://www.python.org/dev/peps/pep-0563/) which discusses forward references.


Type hints act as a form of documentation that helps developers understand the types of arguments a function expects and what it returns.

Example: 


You can provide type hints for function parameters and return values. This helps other developers understand what types of arguments are expected by the function and what type the function returns.

## Basic Syntax

Here are some examples of basic type hinting syntax:


```python
age: int = 20
name: str = "Alice"
is_active: bool = True

def greet(name: str) -> str:
    return f"Hello, {name}!"

def add_numbers(a: int, b: int) -> int:
    return a + b

def get_user_data() -> dict[str, str]:
    return {"username": "user1", "email": "user1@example.com"}


```

Learn more about type hints [here](https://dagster.io/blog/python-type-hinting)


# Dealing with Soft Dependency Type Hints

## Introduction

When working with models that have soft dependencies, it is essential to incorporate type hints effectively to maintain code clarity and functionality also it improves Early error detection and consistancy of the code

 The typing.TYPE_CHECKING constant ensures that imports for type hints are only evaluated when type-checking is done and NOT in the runtime. This prevents errors when the soft dependancies are not available. Here is an example that demonstrates it: 
 
 
 ```python
 from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optional_library import OptionalClass

def process(data: "OptionalClass") -> None:
    pass
```    




## Conclusion

By following these best practices for using type hints in Python, developers can create clearer and more maintainable code. Implementing straightforward annotations and utilizing tools like `TYPE_CHECKING` will help manage dependencies effectively while enhancing overall code quality.
