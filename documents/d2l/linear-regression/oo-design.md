
# Object-Oriented Design for Implementation
:label:`sec_oo-design`

In our introduction to linear regression,
we walked through various components
including
the data, the model, the loss function,
and the optimization algorithm.
Indeed,
linear regression is
one of the simplest machine learning models.
Training it,
however, uses many of the same components that other models in this book require.
Therefore, 
before diving into the implementation details
it is worth 
designing some of the APIs
that we use throughout. 
Treating components in deep learning
as objects,
we can start by
defining classes for these objects
and their interactions.
This object-oriented design
for implementation
will greatly
streamline the presentation and you might even want to use it in your projects.

Inspired by open-source libraries such as [PyTorch Lightning](https://www.pytorchlightning.ai/),
at a high level
we wish to have three classes: 
(i) `Module` contains models, losses, and optimization methods; 
(ii) `DataModule` provides data loaders for training and validation; 
(iii) both classes are combined using the `Trainer` class, which allows us to
train models on a variety of hardware platforms. 
Most code in this book adapts `Module` and `DataModule`. We will touch upon the `Trainer` class only when we discuss GPUs, CPUs, parallel training, and optimization algorithms.

## Utilities
:label:`oo-design-utilities`

We need a few utilities to simplify object-oriented programming in Jupyter notebooks. One of the challenges is that class definitions tend to be fairly long blocks of code. Notebook readability demands short code fragments, interspersed with explanations, a requirement incompatible with the style of programming common for Python libraries. The first
utility function allows us to register functions as methods in a class *after* the class has been created. In fact, we can do so *even after* we have created instances of the class! It allows us to split the implementation of a class into multiple code blocks.

Let's have a quick look at how to use it. We plan to implement a class `A` with a method `do`. Instead of having code for both `A` and `do` in the same code block, we can first declare the class `A` and create an instance `a`.

Next we define the method `do` as we normally would, but not in class `A`'s scope. Instead, we decorate this method by `add_to_class` with class `A` as its argument. In doing so, the method is able to access the member variables of `A` just as we would expect had it been included as part of `A`'s definition. Let's see what happens when we invoke it for the instance `a`.

The second one is a utility class that saves all arguments in a class's `__init__` method as class attributes. This allows us to extend constructor call signatures implicitly without additional code.

We defer its implementation into :numref:`sec_utils`. To use it, we define our class that inherits from `HyperParameters` and calls `save_hyperparameters` in the `__init__` method.

The final utility allows us to plot experiment progress interactively while it is going on. In deference to the much more powerful (and complex) [TensorBoard](https://www.tensorflow.org/tensorboard) we name it `ProgressBoard`. The  implementation is deferred to :numref:`sec_utils`. For now, let's simply see it in action.

The `draw` method plots a point `(x, y)` in the figure, with `label` specified in the legend. The optional `every_n` smooths the line by only showing $1/n$ points in the figure. Their values are averaged from the $n$ neighbor points in the original figure.

In the following example, we draw `sin` and `cos` with a different smoothness. If you run this code block, you will see the lines grow in animation.

## Models
:label:`subsec_oo-design-models`

The `Module` class is the base class of all models we will implement. At the very least we need three methods. The first, `__init__`, stores the learnable parameters, the `training_step` method accepts a data batch to return the loss value, and finally, `configure_optimizers` returns the optimization method, or a list of them, that is used to update the learnable parameters. Optionally we can define `validation_step` to report the evaluation measures.
Sometimes we put the code for computing the output into a separate `forward` method to make it more reusable.

You may notice that `Module` is a subclass of `nn.Module`, the base class of neural networks in PyTorch.
It provides convenient features for handling neural networks. For example, if we define a `forward` method, such as `forward(self, X)`, then for an instance `a` we can invoke this method by `a(X)`. This works since it calls the `forward` method in the built-in `__call__` method. You can find more details and examples about `nn.Module` in :numref:`sec_model_construction`.

##  Data
:label:`oo-design-data`

The `DataModule` class is the base class for data. Quite frequently the `__init__` method is used to prepare the data. This includes downloading and preprocessing if needed. The `train_dataloader` returns the data loader for the training dataset. A data loader is a (Python) generator that yields a data batch each time it is used. This batch is then fed into the `training_step` method of `Module` to compute the loss. There is an optional `val_dataloader` to return the validation dataset loader. It behaves in the same manner, except that it yields data batches for the `validation_step` method in `Module`.

## Training
:label:`oo-design-training`

The `Trainer` class trains the learnable parameters in the `Module` class with data specified in `DataModule`. The key method is `fit`, which accepts two arguments: `model`, an instance of `Module`, and `data`, an instance of `DataModule`. It then iterates over the entire dataset `max_epochs` times to train the model. As before, we will defer the implementation of this method to later chapters.

## Summary

To highlight the object-oriented design
for our future deep learning implementation,
the above classes simply show how their objects 
store data and interact with each other.
We will keep enriching implementations of these classes,
such as via `@add_to_class`,
in the rest of the book.
Moreover,
these fully implemented classes
are saved in the [D2L library](https://github.com/d2l-ai/d2l-en/tree/master/d2l),
a *lightweight toolkit* that makes structured modeling for deep learning easy. 
In particular, it facilitates reusing many components between projects without changing much at all. For instance, we can replace just the optimizer, just the model, just the dataset, etc.;
this degree of modularity pays dividends throughout the book in terms of conciseness and simplicity (this is why we added it) and it can do the same for your own projects. 

## Exercises

1. Locate full implementations of the above classes that are saved in the [D2L library](https://github.com/d2l-ai/d2l-en/tree/master/d2l). We strongly recommend that you look at the implementation in detail once you have gained some more familiarity with deep learning modeling.
1. Remove the `save_hyperparameters` statement in the `B` class. Can you still print `self.a` and `self.b`? Optional: if you have dived into the full implementation of the `HyperParameters` class, can you explain why?

[Discussions](https://discuss.d2l.ai/t/6646)

