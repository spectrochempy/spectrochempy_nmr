{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:

.. rubric:: Methods

.. autosummary::
   :toctree: .
   :nosignatures:

{% for method in methods if method != '__init__' %}
   {{ objname }}.{{ method }}
{% endfor %}

.. rubric:: Attributes

{% for attribute in attributes %}
.. autoattribute:: {{ objname }}.{{ attribute }}

{% endfor %}
