# Cells and pooling

Functions for mapping nuclei onto image tokens and pooling those tokens into one pre-projection
cell embedding per nucleus.

The membership builders support label maps, centroids, and polygons.
`AlhenaEncoder.encode_cells()` performs these steps automatically, while the functions below
expose the lower-level operations for custom workflows and validation.

Membership tables created by ALHENA record their image geometry. Cell-level workflows verify this
metadata against the current tile geometry before pooling. See
[Advanced: membership geometry checks](../concepts/cell-pooling.md#advanced-membership-geometry-checks).

When tile origins are available, ALHENA can identify a nucleus represented in several overlapping
tiles and select one primary tile. See
[A nucleus in several tiles](../concepts/cell-pooling.md#a-nucleus-in-several-tiles).

::: alhena.cells.membership_from_labels

::: alhena.cells.membership_from_centroids

::: alhena.cells.membership_from_polygons

::: alhena.cells.validate_membership

::: alhena.cells.pool_cells

::: alhena.cells.pool_tokens

::: alhena.cells.resolve_main_tile

::: alhena.cells.token_index_map

::: alhena.cells.fit_mask

::: alhena.cells.missing_cells

::: alhena.cells.membership_source

::: alhena.cells.membership_is_approximate

::: alhena.cells.GROUP_COLUMNS

::: alhena.cells.ORIGINS_FRAME

::: alhena.cells.POLYGONS_FRAME

---

**Next:** [Retrieval](annotate.md)
