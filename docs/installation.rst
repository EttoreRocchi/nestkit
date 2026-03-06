Installation
============

Requirements
------------

Python 3.10+ and the following packages (installed automatically):

- scikit-learn >= 1.2
- numpy >= 1.22
- pandas >= 1.4
- joblib >= 1.2
- scipy

Install from PyPI
-----------------

.. code-block:: bash

   pip install nestkit

Optional dependencies
---------------------

.. code-block:: bash

   pip install nestkit[plotting]   # matplotlib + seaborn
   pip install nestkit[full]       # plotting + SHAP
   pip install nestkit[dev]        # testing + linting
   pip install nestkit[docs]       # Sphinx documentation

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/ettorerocchi/nestkit.git
   cd nestkit
   pip install -e ".[dev,full]"
