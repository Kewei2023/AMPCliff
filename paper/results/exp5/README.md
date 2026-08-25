# Exp5 preset results (DC validation design v2)

These artifacts correspond to the upgraded **Exp5** pipeline (see repo README).

| Subdir | Document step | Content |
|--------|---------------|---------|
| `_shared/` | Step 1 | `dc_property_table.csv` and QC |
| `encoding/` | Step 3 / main experiment 1 | DC property probe tables and figures |
| `species_property_effects/` | Step 4 / main experiment 2A | Species×property activity OLS |
| `property_buckets/` | Step 5 / main experiment 2B | Property-bucket band/DC knockout (signed + \|ΔMSE\|) |
| `combined/` | paper figures | Optional combined panels |

Large intermediate DCT features (`dct_features/*.npz`) are **not** bundled; regenerate with Step 2 of `evaluation_scripts/run_dc_validation_v2.sh`.
