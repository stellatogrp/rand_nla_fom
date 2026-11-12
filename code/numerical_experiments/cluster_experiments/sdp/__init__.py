"""Doubly inexact proximal ADMM (Algorithm 3 of the paper) for semidefinite programs.

The SDP application of ``tex/General_inexactness.tex``:

    minimize <C, X>  subject to  E(X) = b,  X positive semidefinite,

split as ``f(X) = <C, X>`` and ``g(y, Z)`` the indicator of ``y = b`` and
``Z`` positive semidefinite, coupled by ``(E(X), X) = (y, Z)``.

* :mod:`sdp.problems`: small synthetic instances with planted solutions.
* :mod:`sdp.projection`: the approximate projection onto the cone by block
  Lanczos, which returns a point of the cone and a normal vector there.
* :mod:`sdp.admm`: Algorithm 3 with conjugate gradients on the quadratic
  ``f``-subproblem and that approximate projection.
* :mod:`sdp.run`, :mod:`sdp.summarize`: the sanity experiments and their table.
"""
