# Coastal Ghana groundwater–subsidence–urban growth screening

Code accompanying the manuscript *Integrated Multi-Sensor Screening of Groundwater Storage Variability, Surface Displacement, and Urban Growth in Coastal Ghana Using GRACE/GRACE-FO, Sentinel-1 InSAR, and Urban Remote-Sensing Data* (submitted to *Environmental and Sustainability Indicators*, manuscript INDIC-7259).

The workflow combines GRACE/GRACE-FO mascons and GLDAS (groundwater storage anomalies), Sentinel-1 SBAS InSAR (HyP3 ISCE burst products and MintPy), GHSL and WorldPop (urban pressure and exposure), and CHIRPS (climate stress) into a screening-level Coastal Groundwater-Subsidence Risk Index (CGSRI) on a 10 km grid covering the 50 km coastal belt of Ghana.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Reusable functions (configuration, groundwater, subsidence, urban growth, indices, plotting) |
| `config/` | Paths, study-area settings, thresholds and the MintPy configuration |
| `notebooks/` | Analysis notebooks 01–12, in pipeline order (outputs cleared) |
| `scripts/` | Data download, InSAR processing, index construction, diagnostics and figure scripts |
| `outputs/tables/` | Derived tables used in the manuscript and supplement, including the reviewer-response diagnostics |
| `data/derived/` | Small derived datasets: the per-cell CGSRI (v2), the regional GWSA series and the per-cell CHIRPS stress indicator |

Raw and processed gridded data (about 150 GB) are not included; see *Data* below.

## Environment

```bash
conda env create -f environment.yml
conda activate ghana-groundwater-subsidence
```

Earthdata (GRACE, GLDAS, HyP3) and Copernicus CDS (ERA5) credentials are read from `~/.netrc` and `~/.cdsapirc`. `scripts/setup_earthdata_credentials.py` and `scripts/setup_cds_credentials.py` create these files interactively; no credentials are stored in this repository.

## Pipeline

1. **Study area and grid**: `scripts/download_boundaries.py`, `scripts/prepare_study_area.py`, notebook 02.
2. **Groundwater storage**: `scripts/download_grace_gldas.py`, notebooks 03–04. GWSA = TWSA − SMSA − CWSA, area-averaged over land-masked GRACE cells in 4.0–11.5°N, 3.5°W–1.5°E, with anomalies relative to the 2004–2009 monthly climatology.
3. **Lake Volta storage**: `scripts/lake_volta_correction.py` removes Lake Volta storage (G-REALM V2 altimetry) from the GWSA for April 2002 to March 2019 and compares trends and drought episodes.
4. **Climate stress**: `scripts/download_chirps.py`, notebook 08, `scripts/chirps_below_normal.py` (percentage of months in 2000–2024 with rainfall below the 1981–2010 calendar-month mean).
5. **Sentinel-1 InSAR**: `scripts/submit_hyp3_isce_burst.py` and `scripts/submit_hyp3_network_fill.py` (230 gap-bridging and redundancy pairs), `scripts/fetch_hyp3_fill.py`, then `scripts/run_mintpy_v2.py`, which resamples the fill pairs to the common 80 m grid and runs MintPy on the connected 385-interferogram network (configuration in `config/mintpy_v2_smallbaselineApp.cfg`). `scripts/export_velocity_v2.py` exports the velocity fitted to the ERA5- and DEM-error-corrected time series (temporal coherence ≥ 0.7, average spatial coherence ≥ 0.5) as GeoTIFFs in EPSG:32630. `scripts/run_mintpy_sbas.py` and `scripts/run_era5_correction.py` are the superseded v1 workflow: the v1 velocity raster was dominated by the ERA5 delay rate, and its ERA5 delay file was empty after July 2022.
6. **Urban growth and UGPI**: notebooks 07 and 09.
7. **Risk index**: `scripts/build_cgsri_v2.py` builds the harmonised CGSRI, with identical component rules in every cell. The primary index has four components (groundwater hazard, climate stress, exposure, vulnerability); an exploratory five-component version adds InSAR-based subsidence hazard. Notebook 10 is the original (v1) construction.
8. **Diagnostics**: `scripts/insar_diagnostics_v2.py` (network connectivity, coherence, planar trend, GNSS comparison, internal consistency, Table S1; summary in `outputs/tables/revision_checks_v2/insar_v2_summary.csv`), `scripts/revision_checks_v2.py` (harmonised index: sensitivity, Monte Carlo, co-location, ramp sensitivity), `scripts/colocation_permutation.py` (Moran spectral randomisation test) and `scripts/revision_checks.py` (v1 audit).
9. **Figures**: `scripts/regen_all_12panels.py`, `scripts/regen_fig7.py`, `scripts/regen_supp_v2.py`, `scripts/insar_diagnostics_v2.py` (Supplementary Figs S1, S3, S4, S7), `scripts/build_conceptual_framework_fig.py`. `scripts/regenerate_supp_figures.py` produced the superseded v1 InSAR figures.

Run order after the MintPy run: `export_velocity_v2.py`, `build_cgsri_v2.py --velocity <ROOT>/data/processed/subsidence/mintpy_velocity_los_v2.tif`, `insar_diagnostics_v2.py`, `revision_checks_v2.py`, `colocation_permutation.py`, then the figure scripts.

## Data

All inputs are openly available:

- GRACE/GRACE-FO JPL mascon RL06.3 (PO.DAAC)
- GLDAS Noah v2.1 (GES DISC)
- Sentinel-1 bursts via ASF HyP3
- ERA5 (Copernicus CDS)
- CHIRPS v2.0 (UCSB CHC)
- GHSL GHS-BUILT-S R2023A (JRC)
- WorldPop 2020
- Copernicus GLO-30 DEM
- GADM v4.1
- G-REALM V2 lake heights (PO.DAAC, doi:10.5067/UCLRS-GREV2)
- Nevada Geodetic Laboratory MIDAS GNSS velocities

The processed datasets will be archived on Zenodo with a DOI on acceptance.

## Licence

MIT (see `LICENSE`).
