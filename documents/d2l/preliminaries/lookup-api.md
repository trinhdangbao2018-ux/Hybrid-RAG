
# Documentation

While we cannot possibly introduce every single PyTorch function and class 
(and the information might become outdated quickly), 
the [API documentation](https://pytorch.org/docs/stable/index.html) and additional [tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html) and examples 
provide such documentation.
This section provides some guidance for how to explore the PyTorch API.

## Functions and Classes in a Module

To know which functions and classes can be called in a module,
we invoke the `dir` function. For instance, we can
(**query all properties in the module for generating random numbers**):

Generally, we can ignore functions that start and end with `__` (special objects in Python) 
or functions that start with a single `_`(usually internal functions). 
Based on the remaining function or attribute names, 
we might hazard a guess that this module offers 
various methods for generating random numbers, 
including sampling from the uniform distribution (`uniform`), 
normal distribution (`normal`), and multinomial distribution (`multinomial`).

## Specific Functions and Classes

For specific instructions on how to use a given function or class,
we can invoke the  `help` function. As an example, let's
[**explore the usage instructions for tensors' `ones` function**].

From the documentation, we can see that the `ones` function 
creates a new tensor with the specified shape 
and sets all the elements to the value of 1. 
Whenever possible, you should (**run a quick test**) 
to confirm your interpretation:

In the Jupyter notebook, we can use `?` to display the document in another
window. For example, `list?` will create content
that is almost identical to `help(list)`,
displaying it in a new browser window.
In addition, if we use two question marks, such as `list??`,
the Python code implementing the function will also be displayed.

The official documentation provides plenty of descriptions and examples that are beyond this book. 
We emphasize important use cases 
that will get you started quickly with practical problems, 
rather than completeness of coverage. 
We also encourage you to study the source code of the libraries 
to see examples of high-quality implementations of production code. 
By doing this you will become a better engineer 
in addition to becoming a better scientist.

[Discussions](https://discuss.d2l.ai/t/39)

