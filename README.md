# Data science task

***
## Task Description
### Problem 1: Python programming, data processing.
In this problem we want to generate pseudo-random data that has certain desired statistical properties.
This can be useful for demo, research or testing purposes. First, let’s generate these “desired statistical properties”.
- Generate a random 6x6 correlation matrix rho.
- Regularization: write a test checking that rho is a valid correlation matrix, and if not - find the nearest valid one.

Now, let’s generate the data that would have these properties.
- Generate a set of 6 random variables (put them in a matrix 1000x6, the distribution doesn’t matter, but should 
be continuous), distributed between 0 and 1 with correlation defined by rho.

Slightly different, but related problem.
- Apply PCA to reduce the dimensionality to 5.
- Let the output variable y = round(x6).
- Build a couple of classifiers of your choice to predict y from {x1, x2, …, x5}.
- Compare their performance.

### Problem 2: Data Science, Model Build
Please use the attached file for your data exercise.
You have been provided with a dataset that has 116 Rows, 123 Columns (mix of continuous and categorical variables) 
and a target column.
The goal is to build a model that generalizes well over this dataset, you are free to transform the dataset as you feel
necessary. We are not looking for the highest scoring model. 
Our goal is to understand your thought process and decision making.

### Problem 3:
It's 3000 BC, and you are the leader of a tribe of 4000 people. You are leading your tribe to a new location where 
you must build a circular settlement from scratch. How big will it be and how long will it take to build a stone wall
around it?

### Problem 4:
Is there an inconsistency in the following paragraph?: "A suburban located Starbucks makes on average $100,000 per 
month in revenue and has 10,500 square meters of an adjacent area dedicated to parking for visitors only.
Despite good revenue and overall satisfaction with service, both the staff and visitors are complaining that parking 
is full more than half of the time."
***

# Guide for project and solution review
## Running problem solutions
**All solutions tested in main.py file**. I implemented tools encapsulated in multiple
modules and by using them I'm providing exact results in **main.py**.
Each problem solution tools **encapsulated** into different modules. For example
**problem 1 solution tools implemented in problem_1_data_processing.py** module.

For testing problem 1 and problem 2 solutions run following command
```
python main.py
```
You will see plots which will present correlation matrices and other useful information.
Take a look to the terminal logging messages, all metrics and results serialized there.

First will go problem 1 solution, second one will be problem 2.

**Pay attention** - whenever you will run, you **need to close plots after investigation
to be able to continue running this project**, otherwise project will wait infinite.

## Project review
Start review from ```main.py``` file. There I commented each step and problem solution.
It will help you investigate high level solutions. If you want to go deeper into implementation
details you need to investigate ```problems_solution_tools``` directory modules.

***

### Steps for running service
***
## Installation:
### Local Setup
If you have cloned this repository and created a virtual environment for it. You can install all the dependencies by running:
``` bash
pip3 install -r requirements.txt
```

***
# Usage
```
Todo
```

***
## Contributing guidelines
Thank you for following them!

### Branching strategy
Nothing new, nothing fancy:
* "Main" Branch:This is the primary branch that represents the production-ready version of the codebase. Developers 
should aim to keep this branch stable, and it should only contain code that has been fully tested and is ready
for release.

* "Development" Branch:This branch is used to integrate new code changes and features that are not yet production-ready.
Developers work on this branch to implement new functionality, fix bugs, and make improvements to the codebase. 
Once the changes are tested and validated, they are merged into the main branch for release.

* "Features" Branch:Feature branches are used to develop new features or major changes to the codebase. These 
branches are created off the development branch and allow developers to work independently on specific features 
without interfering with the development of other features.

* "Hotfixes" Branch:Hotfix branches are used to quickly address critical issues or bugs in the codebase that require
immediate attention. These branches are created off the main branch and allow developers to fix issues without
disrupting the development of new features on the development branch. Once the fix is complete, the hotfix branch is
merged back into the main branch.

### New features or refactorings
- Create a branch from development branch.
- Describe the changes you made and why you made them. If there is a JIRA task associated, please  write its reference.
- Implement your changes
- Ensure that the code is properly formatted and passes all existing tests. Create new tests for new methods or classes.
- Make sure to update the documentation if necessary
- Ask for a merge request to the development branch as soon as possible to avoid differences overlapping.

### CI/CD
#### Coverage
In this repo we are using coverage to check the code coverage of the tests. You can test it by running
``` bash
 coverage run -m pytest 
```
Then you can visualize the coverage report with:
``` bash
 coverage report
```
The highest the coverage the better! Also, make sure that every file is being covered.
Personally, if you are using Pycharm-Pro I recommend use the function "Run Python tests in tests with coverage" as it 
will allow you to see which lines are not under coverage.

## Future Work
- Add unit and integration tests (by mocking requests and databases)
