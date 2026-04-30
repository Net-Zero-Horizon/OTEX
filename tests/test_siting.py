# -*- coding: utf-8 -*-
"""
Tests for the siting layer: configuration plumbing, hazard cost multipliers,
graceful degradation when geospatial extras are not installed, and regression
tests for raster-sampling bugs found during the Jamaica smoke test (NaN
propagation in masked rasters; int-dtype rasters refusing NaN fill).

These tests deliberately avoid the network and any real WDPA / vessel
density / GEM / IBTrACS files. They cover the plumbing only; an integration
test against real layers would belong in tests/integration/ once a fixture
is available.
"""

import importlib
import zipfile

import numpy as np
import pandas as pd
import pytest

from otex.config import OTEXConfig, SitingConfig, parameters_and_constants
from otex.data.siting import enrich_sites
from otex.data.siting.download import (
    SitingDownloadError,
    ensure_layers,
)
from otex.data.siting import download as dl_mod


class TestSitingConfig:
    def test_defaults_disabled(self):
        s = SitingConfig()
        assert s.enable_mpa_filter is False
        assert s.enable_ais_filter is False
        assert s.enable_hazard_costs is False

    def test_defaults_buffers_5km(self):
        s = SitingConfig()
        assert s.mpa_buffer_km == 5.0
        assert s.ais_buffer_km == 5.0

    def test_legacy_dict_exposes_keys(self):
        cfg = OTEXConfig()
        cfg.siting.enable_mpa_filter = True
        cfg.siting.w_seismic = 0.3
        legacy = cfg.to_legacy_dict()
        assert legacy['siting_enable_mpa_filter'] is True
        assert legacy['siting_w_seismic'] == 0.3
        assert 'siting_mpa_buffer_km' in legacy

    def test_legacy_function_still_works(self):
        # Backward compatibility: parameters_and_constants must still produce
        # a dict with all siting keys present and defaults benign.
        inputs = parameters_and_constants()
        assert inputs['siting_enable_mpa_filter'] is False
        assert inputs['siting_enable_ais_filter'] is False
        assert inputs['siting_w_ais'] == 0.20


class TestHazardMultipliers:
    """Verify the formulas in costs.py without invoking the full plant."""

    @staticmethod
    def _multipliers(ais_pct, pga_g, cyc_yr,
                     w_ais=0.2, w_seismic=0.15, w_cyclone=0.25,
                     ais_norm_pct=95.0, pga_ref=0.4, cyc_ref=0.5):
        ais_norm = np.clip(ais_pct / ais_norm_pct, 0.0, 1.0)
        pga_norm = np.clip(pga_g / pga_ref, 0.0, 1.0)
        cyc_norm = np.clip(cyc_yr / cyc_ref, 0.0, 1.0)
        capex = 1.0 + w_ais * ais_norm + w_seismic * pga_norm
        opex = 1.0 + w_ais * ais_norm + w_cyclone * cyc_norm
        return capex, opex

    def test_zero_hazards_no_penalty(self):
        capex, opex = self._multipliers(0.0, 0.0, 0.0)
        assert capex == pytest.approx(1.0)
        assert opex == pytest.approx(1.0)

    def test_seismic_only_hits_capex(self):
        capex, opex = self._multipliers(0.0, 0.4, 0.0)
        assert capex == pytest.approx(1.15)
        assert opex == pytest.approx(1.0)

    def test_cyclone_only_hits_opex(self):
        capex, opex = self._multipliers(0.0, 0.0, 0.5)
        assert capex == pytest.approx(1.0)
        assert opex == pytest.approx(1.25)

    def test_ais_hits_both(self):
        capex, opex = self._multipliers(95.0, 0.0, 0.0)
        assert capex == pytest.approx(1.20)
        assert opex == pytest.approx(1.20)

    def test_normalization_caps_at_one(self):
        # PGA above the reference must clamp, not blow up
        capex, _ = self._multipliers(0.0, 5.0, 0.0)
        assert capex == pytest.approx(1.15)


class TestDownloadCache:
    def test_unknown_layer_raises(self, tmp_path):
        with pytest.raises(SitingDownloadError):
            ensure_layers(layers=["nonexistent"], cache_dir=tmp_path)

    def test_missing_url_raises_with_hint(self, tmp_path, monkeypatch):
        # When the in-package default and env var are both empty, the error
        # message should name the env var and the cache target so the user
        # knows exactly how to recover.
        from otex.data.siting import download as dl
        monkeypatch.setitem(dl.LAYER_SOURCES["wdpa"], "default_url", "")
        monkeypatch.delenv("OTEX_WDPA_URL", raising=False)
        with pytest.raises(SitingDownloadError) as exc:
            ensure_layers(layers=["wdpa"], cache_dir=tmp_path)
        msg = str(exc.value)
        assert "OTEX_WDPA_URL" in msg
        assert "wdpa" in msg

    def test_existing_cache_reused(self, tmp_path):
        # Pre-create the file → ensure_layers should NOT re-download
        target = tmp_path / "ibtracs_all.nc"
        target.write_bytes(b"stub")
        result = ensure_layers(layers=["ibtracs"], cache_dir=tmp_path)
        assert result["ibtracs"] == target
        assert target.read_bytes() == b"stub"  # untouched


class TestEnrichGracefulDegradation:
    def test_no_layers_available_returns_neutral_columns(self, tmp_path, monkeypatch):
        # Force every layer URL to be empty so ensure_layers raises and
        # enrich_sites falls back to neutral defaults. We have to overwrite
        # in-package defaults, not just env vars, since AIS/PGA/IBTrACS now
        # ship with hardcoded URLs.
        from otex.data.siting import download as dl
        for layer in dl.LAYER_SOURCES.values():
            monkeypatch.setitem(layer, "default_url", "")
        for var in ("OTEX_WDPA_URL", "OTEX_AIS_URL", "OTEX_PGA_URL", "OTEX_IBTRACS_URL"):
            monkeypatch.delenv(var, raising=False)

        sites = pd.DataFrame({
            "id": [1, 2],
            "longitude": [-80.0, 145.0],
            "latitude": [25.0, -15.0],
        })
        out = enrich_sites(sites, cache_dir=tmp_path)
        assert list(out["in_mpa_strict"]) == [False, False]
        assert (out["ais_density_pct"] == 0.0).all()
        assert (out["pga_475"] == 0.0).all()
        assert (out["cyclone_freq_per_yr"] == 0.0).all()


# ---------------------------------------------------------------------------
# Raster-sampling regression tests
# ---------------------------------------------------------------------------

# These exercise the actual sampling routines against tiny synthetic GeoTIFFs.
# They require rasterio (siting extra). When rasterio is missing the tests
# skip rather than fail, which keeps the core install green.

rasterio = pytest.importorskip("rasterio")


def _write_geotiff(path, arr, *, dtype, nodata, transform=None, crs="EPSG:4326"):
    """Write a tiny GeoTIFF for use as a test fixture."""
    from rasterio.transform import from_origin

    if transform is None:
        # 0.1° pixels, top-left at (0, 1) so x in [0, 0.1*ncols], y in [...]
        transform = from_origin(0.0, float(arr.shape[0]) * 0.1, 0.1, 0.1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


class TestRasterSamplingBugs:
    """Regressions for bugs found during the Jamaica smoke test."""

    def test_int32_raster_does_not_crash(self, tmp_path):
        """Sampling an int32 raster (vessel density) used to crash on
        np.ma.filled(arr, NaN) because NaN does not fit in int32."""
        from otex.data.siting.enrich import _sample_raster_window

        path = tmp_path / "intraster.tif"
        # 5x5 raster, all positive integers, no nodata
        arr = np.arange(25, dtype=np.int32).reshape(5, 5) + 1
        _write_geotiff(path, arr, dtype="int32", nodata=2147483647)

        sites = pd.DataFrame({"longitude": [0.25], "latitude": [0.25]})
        out = _sample_raster_window(sites, path, buffer_km=10.0)

        assert np.isfinite(out[0])
        assert out[0] > 0  # max within window must be positive

    def test_float_raster_with_nan_uses_nanmax(self, tmp_path):
        """Sampling a float raster that mixes valid values with NaN
        (typical of GEM PGA over the ocean) must NOT propagate NaN through
        the window-max — np.nanmax should win."""
        from otex.data.siting.enrich import _sample_raster_window

        path = tmp_path / "nanraster.tif"
        arr = np.array([
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 0.10,    np.nan, np.nan, np.nan],
            [np.nan, np.nan, 0.30,    np.nan, np.nan],
            [np.nan, np.nan, np.nan, 0.50,    np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ], dtype=np.float64)
        _write_geotiff(path, arr, dtype="float64", nodata=None)

        sites = pd.DataFrame({"longitude": [0.25], "latitude": [0.25]})
        out = _sample_raster_window(sites, path, buffer_km=300.0)

        assert np.isfinite(out[0])
        assert out[0] == pytest.approx(0.50)

    def test_extreme_sentinel_treated_as_missing(self, tmp_path):
        """GEM PGA encodes nodata as DBL_MAX (~1.8e308). The sampler must
        treat values > 1e30 as NaN before computing nanmax."""
        from otex.data.siting.enrich import _sample_raster_window

        path = tmp_path / "sentinel.tif"
        arr = np.full((5, 5), 1.7976931348623157e+308, dtype=np.float64)
        arr[2, 2] = 0.25  # one real value
        _write_geotiff(path, arr, dtype="float64", nodata=None)

        sites = pd.DataFrame({"longitude": [0.25], "latitude": [0.25]})
        out = _sample_raster_window(sites, path, buffer_km=300.0)

        assert out[0] == pytest.approx(0.25)


class TestZipExtraction:
    """The download helper extracts ZIP archives in two modes."""

    def test_zip_member_extracts_matching_suffix(self, tmp_path):
        from otex.data.siting.download import _extract_zip_member

        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "ignore me")
            zf.writestr("payload.tif", b"GTIFF_BYTES")

        dest = tmp_path / "out.tif"
        _extract_zip_member(zip_path, dest)
        assert dest.read_bytes() == b"GTIFF_BYTES"

    def test_zip_member_raises_when_no_match(self, tmp_path):
        from otex.data.siting.download import _extract_zip_member

        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "no raster here")

        dest = tmp_path / "out.tif"
        with pytest.raises(SitingDownloadError):
            _extract_zip_member(zip_path, dest)

    def test_zip_all_extracts_every_member(self, tmp_path):
        """WDPA distributions ship as multi-file shapefiles; the zip_all
        mode must extract all entries into the destination directory."""
        from otex.data.siting.download import _extract_zip_all

        zip_path = tmp_path / "wdpa.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.shp", b"shp")
            zf.writestr("a.dbf", b"dbf")
            zf.writestr("a.shx", b"shx")
            zf.writestr("nested/b.shp", b"shp2")

        dest = tmp_path / "wdpa"
        _extract_zip_all(zip_path, dest)
        assert (dest / "a.shp").read_bytes() == b"shp"
        assert (dest / "a.dbf").exists()
        assert (dest / "a.shx").exists()
        assert (dest / "nested" / "b.shp").read_bytes() == b"shp2"

    def test_directory_cache_must_be_non_empty(self, tmp_path, monkeypatch):
        """An empty cached directory should NOT be considered a valid
        cached layer — that would persist a failed extraction forever."""
        empty_dir = tmp_path / "wdpa"
        empty_dir.mkdir()

        # Force WDPA URL empty so download cannot rescue us; the test asserts
        # that ensure_layers refuses to call the empty dir cached.
        monkeypatch.setitem(dl_mod.LAYER_SOURCES["wdpa"], "default_url", "")
        monkeypatch.delenv("OTEX_WDPA_URL", raising=False)
        with pytest.raises(SitingDownloadError):
            ensure_layers(layers=["wdpa"], cache_dir=tmp_path)


class TestHaversineEdgeCases:
    """Float rounding can push the haversine intermediate above 1; the
    arcsin must not produce a warning or NaN for coincident points."""

    def test_zero_distance_no_nan(self):
        from otex.data.siting.enrich import _haversine_km

        d = _haversine_km(15.0, -77.0, np.array([15.0]), np.array([-77.0]))
        assert np.all(np.isfinite(d))
        assert d[0] == pytest.approx(0.0, abs=1e-6)
