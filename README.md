# Coastal Ghana groundwater–subsidence–urban growth screening

Code accompanying the manuscript *Integrated Multi-Sensor Screening of Groundwater Storage Variability, Surface Displacement, and Urban Growth in Coastal Ghana Using GRACE/GRACE-FO, Sentinel-1 InSAR, and Urban Remote-Sensing Data* (submitted to *Environmental and Sustainability Indicators*, manuscript INDIC-7259).

The workflow combines GRACE/GRACE-FO mascons and GLDAS (groundwater storage anomalies), Sentinel-1 SBAS InSAR (HyP3 ISCE burst products and MintPy), GHSL and WorldPop (urban pressure and exposure), and CHIRPS (climate stress) into a screening-level Coastal Groundwater-Subsidence Risk Index (CGSRI) on a 10 km grid covering land within 50 km of Ghana's national boundary south of about 6.5°N: the coastal belt plus inland arms along the Ivorian and Togolese borders (151 of the 453 cells lie more than 50 km from the sea; `scripts/coastal_only_sensitivity.py` tests a coastal-only domain).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Reusable functions (configuration, groundwater, subsidence, urban growth, indices, plotting) |
| `config/` | Paths, study-area settings, thresholds and the MintPy configuration |
| `notebooks/` | Analysis notebooks 01–12, in pipeline order (outputs cleared) |
| `scripts/` | Data download, InSAR processing, index construction, diagnostics and figure scripts |
| `outputs/tables/` | Derived tables used in the manuscript and supplement, including the reviewer-response diagnostics |
| `data/derived/` | Small derived datasets: the per-cell CGSRI (v2), the regional GWSA series, the per-cell CHIRPS stress indicator and the built-up area of the 453-cell grid by GHSL epoch |

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
6. **Urban growth and UGPI**: notebooks 07 and 09; `scripts/regen_fig5.py` sums GHSL built-up surface inside the 453 grid cells for each epoch (the GHSL rasters cover all of Ghana).
7. **Risk index**: `scripts/build_cgsri_v2.py` builds the harmonised CGSRI, with identical component rules in every cell. The primary index has four components (groundwater hazard, climate stress, exposure, vulnerability); an exploratory five-component version adds InSAR-based subsidence hazard. Notebook 10 is the original (v1) construction.
8. **Diagnostics**: `scripts/insar_diagnostics_v2.py` (network connectivity, coherence, planar trend, GNSS comparison, internal consistency, Table S1; summary in `outputs/tables/revision_checks_v2/insar_v2_summary.csv`), `scripts/revision_checks_v2.py` (harmonised index: sensitivity, Monte Carlo, co-location, ramp sensitivity), `scripts/colocation_permutation.py` (Moran spectral randomisation test), `scripts/coastal_only_sensitivity.py` (index and co-location restricted to the 302 cells within 50 km of the sea) and `scripts/revision_checks.py` (v1 audit).
9. **Figures**: `scripts/regen_all_12panels.py`, `scripts/regen_fig7.py`, `scripts/regen_supp_v2.py`, `scripts/insar_diagnostics_v2.py` (Supplementary Figs S1, S3, S4, S7), `scripts/build_conceptual_framework_fig.py`. `scripts/regenerate_supp_figures.py` produced the superseded v1 InSAR figures.

Run order after the MintPy run: `export_velocity_v2.py`, `build_cgsri_v2.py --velocity <ROOT>/data/processed/subsidence/mintpy_velocity_los_v2.tif`, `insar_diagnostics_v2.py`, `revision_checks_v2.py`, `colocation_permutation.py`, then the figure scripts.

## Dry-season surface warming and urban groundwater-demand pressure (`scripts/paper2/`)

Code for the manuscript *Dry-season surface warming, urban expansion and groundwater-demand pressure in coastal Ghana: an exploratory multi-sensor screening* (in preparation). It uses the 10 km grid, UGPI and CHIRPS climate stress built above, and the regional storage ensemble built in `scripts/paper3/`.

1. `download_modis_lst.py`, `download_modis_h18.py`: MODIS MOD11A2 v6.1 tiles h17v08 and h18v08 (earthaccess; Earthdata login in `~/.netrc`).
2. `process_modis_lst.py`: quality-masks (QC_Day bits 0–1 = 00), scales, reprojects to EPSG:32630 at 1 km, mosaics the two tiles and writes `lst_8day_by_cell.csv` (mean of valid pixels per 10 km cell and 8-day composite). Its trend step is the superseded v1 analysis.
3. `build_uhgpi_v2.py`: restricts the grid to cells whose centroid lies within 50 km of the sea (Natural Earth 10 m coastline, read from the cartopy data directory); builds Dec–Feb seasons 2001–2023 (≥ 3 composites per season), Hamed–Rao trends with Benjamini–Hochberg FDR (≥ 18 seasons), the 2019–2023 LST anomaly, the Urban Heat–Groundwater Pressure Index (UHGPI; equal-weight mean of normalised LST trend, LST anomaly, UGPI and climate stress), the sensitivity tests, Moran spectral randomisation tests and the GWSA–LST temporal comparison. Outputs: `data/derived/paper2/` and `outputs/tables/paper2_v2/` (`build_log.txt` lists every reported statistic).
4. `download_modis_aqua.py`: Aqua MYD11A2 v6.1 tiles h17v08 and h18v08 for the December–February seasons 2003–2023.
5. `build_paper2_erc_checks.py`: night-time LST, trend sensitivity to the final season (Terra orbit drift), TerraClimate Tmax comparison, sensitivity of the warming–growth association (growth measure, projected GHSL epoch, climatic controls, Tarkwa–Prestea exclusion), cloud sampling, and the regional storage–LST tests across 27 ensemble members with effective sample size and FDR control. Every statistic is written to `outputs/tables/paper2_erc/build_log.txt`.
6. `build_paper2_aqua_check.py`: compares Terra and Aqua over 2003–2023 (trends, sensor agreement, urban associations and urban contrast). Log: `outputs/tables/paper2_erc/aqua_build_log.txt`.
7. `figures_paper2_erc.py`: Figures 1–9 and Supplementary Figures S1–S2 of the current manuscript. `figures_paper2_v2.py` produced the earlier figure set.

Cell-level results are in `data/derived/paper2/` and `data/derived/paper2/erc/`.

GRACE/GRACE-FO storage is used only as regional temporal context and is not a component of the UHGPI.

## Lake Volta storage in GRACE-derived groundwater (`scripts/paper3/`)

Code for the manuscript *Reservoir storage masks the groundwater drought signal in GRACE over southern Ghana: separating Lake Volta from aquifer storage, 2002–2025* (in preparation). It uses the regional GRACE domain described above: the 123 JPL land cells of 0.5° in 4.0–11.5°N, 3.5°W–1.5°E.

1. `download_paper3_inputs.py` downloads:
   - CSR RL06.3 and GSFC RL06 v2.0 mascons;
   - GLDAS-2.1 1° monthly Noah, CLSM and VIC (January 2002 – December 2025; earthaccess, Earthdata login in `~/.netrc`);
   - the operational G-REALM 10-day Lake Volta levels (target 000393, NASA GSFC Global Water Monitor).
2. `build_paper3_v2.py` builds:
   - a 9-member GWSA ensemble (3 GRACE solutions × 3 land surface models, 2004–2009 baseline);
   - the Lake Volta storage term (A·Δh/A_domain with A = 6,000 or 8,502 km², plus a half-removal case) and Bui reservoir filling scenarios;
   - trends (Sen slope on a time axis, Hamed–Rao test), standardised and detrended drought indices, episode windows across members, and correlations with PDSI and SPI-1 to SPI-24 (effective sample size).
   
   It also compares the JPL × Noah 1° and 0.25° series and runs the uncertainty, seasonal-cycle, threshold, baseline, start/end-year, FDR and lake-extent checks. Every reported statistic is written to `outputs/tables/paper3_v2/build_log.txt`. Derived series are in `data/derived/paper3/` (written to `data/processed/paper3/` by the script).
3. `figures_paper3_v2.py`: Figs 1–7 and Supplementary Figs S1–S3 of the manuscript.

In GLDAS-2.1, CLSM has no separate groundwater store (its terrestrial storage equals profile soil moisture plus canopy water), so CLSM-based members are also reported separately.

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
- G-REALM V2 lake heights (PO.DAAC, doi:10.5067/UCLRS-GREV2) and the operational G-REALM 10-day product (NASA GSFC Global Water Monitor)
- CSR GRACE/GRACE-FO RL06.3 and GSFC RL06 v2.0 mascons; GLDAS-2.1 Noah, CLSM and VIC (1°)
- TerraClimate PDSI
- Nevada Geodetic Laboratory MIDAS GNSS velocities

The processed datasets will be archived on Zenodo with a DOI on acceptance.

## Licence

MIT (see `LICENSE`).
