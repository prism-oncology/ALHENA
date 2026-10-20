# Errors and warnings

Every exception and warning ALHENA raises. Catch `AlhenaError` for all eight concrete errors, and
filter on `AlhenaWarning` for every warning: the three below, and the demo's `ClassCodeWarning`.

!!! warning "ReferenceError shadows a builtin"
    `from alhena import ReferenceError` rebinds the builtin of the same spelling in that module.
    Where both can be raised, import the package and write `alhena.ReferenceError`.

::: alhena._errors.AlhenaError

::: alhena._errors.AlhenaValueError

::: alhena._errors.BundleError

::: alhena._errors.BackboneNotFoundError

::: alhena._errors.ReferenceError

::: alhena._errors.CellSpaceMismatchError

::: alhena._errors.GeometryError

::: alhena._errors.MembershipError

::: alhena._errors.MissingExtraError

::: alhena._errors.AlhenaWarning

::: alhena._errors.GeometryWarning

::: alhena._errors.ReferenceCoverageWarning

::: alhena._errors.EnvelopeWarning

---

**Next:** [Constants](constants.md)
