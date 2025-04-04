class MetaTracer(type):
    """
    Metaclasses are used to create other classes. 
    This metaclass wraps all methods of a class to automatically intercept and log method calls when invoked, 
    including magic methods such as `__init__` or `__del__`.

    Example
    --------
        >>> class MyClass(metaclass=MetaTracer):
        >>>     def method(self, x):
        >>>         return x * 2
        >>> obj = MyClass()
        >>> obj.method(10)

        CALL: method((<__main__.MyClass object at 0x...>, 10), {})

    Use case
    --------
        Use this metaclass to trace lifespan of specific class instances referenced as a Streamlit application is running. 
        It can help to understand when and which magic methods, for example destructor, are invoked. 
    """

    def __new__(cls: type, name: str, bases: tuple, class_dict: dict) -> type:
        """
        Override the __new__ method to modify how classes are created. 
        Creates a new class with wrapped methods that log calls.

        Parameters
        ------------
            cls : type
                The metaclass (`MetaTracer` itself).
            name : str
                The name of the class being created.
            bases : tuple
                Base classes of the class being created.
            class_dict : dict
                Dictionary containing class attributes and methods.

        Returns
        --------
            type
                The newly created class with logging applied to its methods.
        """
        # Iterate through all atributes of the class
        for attr_name, attr_value in class_dict.items():
            # Only wrap methods
            if callable(attr_value):
                # Store the original method
                original_method = attr_value

                # Wrapper function to log every method call before executing it.
                # Avoid late binding issues by prepending inner function parameters with the __ prefix.
                def wrapper(self, *args, __original_method=original_method, **kwargs) -> any:
                    """
                    Wrapper function that logs method calls.

                    Parameters
                    ------------
                        self : object
                            The instance of the class.
                        *args : tuple
                            Positional arguments for the method.
                        **kwargs : dict
                            Keyword arguments for the method.

                    Returns
                    --------
                        Any
                            The return value of the original method.
                    """
                    # Log method call
                    print(
                        f"CALL: {__original_method.__name__}({args}, {kwargs})"
                    )

                    # Call original method
                    return __original_method(self, *args, **kwargs)

                # Replace original method with a wrapped version
                class_dict[attr_name] = wrapper

        # Call the parent metaclass (type) to create new class with modified methods
        return super().__new__(cls, name, bases, class_dict)
