Cubed-sphere study deliverables and data archive
================================================

Read ``cubed_sphere_diffusion_convection_summary.pdf`` for the eight-page
summary, or ``cubed_sphere_diffusion_convection.rst`` for the full technical
note. Figures, generated RST tables, source scripts, and tests remain loose.

``cubed_sphere_study_data.tar.gz`` is the single archive of the study's JSON,
CSV, log, and eigenprofile files, plus their hardware reports and launch
records. Paths inside it are relative to ``docs/notes``. Every payload file
has a SHA-256 entry in the internal ``ARCHIVE_SHA256SUMS`` inventory.
The snapshot contains 436 payload files: 18,838,224 uncompressed bytes in a
4,174,224-byte archive. The checksum inventory is an additional archive member.

Retained evidence
-----------------

* All 77 spherical runs used by the final analysis, including grid, timestep,
  seed, orientation, precision, and neighboring-degree controls.
* The three long-duration runs, transport comparisons, and the documented
  failed GPU-allocation attempt. Failed-attempt evidence is not simulation data.
* The planar benchmark, earlier spherical benchmark, and independent reference
  results used in the full note.
* Derived analysis JSON, run manifests, per-run metadata, eigenprofiles,
  hardware reports, and the PDF's original provenance manifest.

No numerical evidence was discarded: all CSV files in the main spherical
sample directory are used by the final analysis. Repeated metadata is retained
for provenance and compresses efficiently. Cache files are not archive inputs.

Verify and restore
------------------

From the repository root, verify all archived checksums without extraction::

   python docs/notes/study_archive.py verify

Restore the original paths before running sample-based analysis or GPU batch
commands shown in the technical note::

   python docs/notes/study_archive.py extract

Existing identical files are retained; a changed file causes extraction to
stop before writing any file. Extraction rejects archive links and paths that
escape the destination. To inspect a separate copy instead::

   python docs/notes/study_archive.py extract --directory /tmp/snapy-study-data

For example, analyze that copy without restoring data into the repository::

   OPENBLAS_NUM_THREADS=1 python docs/notes/analyze_spherical_onset.py \
     --input-dir /tmp/snapy-study-data/spherical_samples \
     --reference /tmp/snapy-study-data/spherical_reference_results.json \
     --output-dir /tmp/snapy-study-analysis --baseline-step-scale 8

The PDF builder and runner regression test can read their JSON inputs directly
from the verified archive. Other analysis and batch scripts retain their
existing filesystem interface and require extraction first. Rebuilding a PDF
or rerunning an analysis creates new loose result files; the archive remains
the immutable snapshot of this completed study.

The cleanup changes data packaging only. It makes no solver or external API
change. Existing distributed runs still require matching diffusion libraries,
as documented in the technical note.

Cleanup validation
------------------

``python docs/notes/study_archive.py verify`` verified all 436 payload hashes.
A fresh extraction reproduced all 77 primary-analysis runs and all 22 modal
neutral estimates exactly; supplemental control coverage remained complete.
The PDF builder also produced eight pages from a copy containing the archive
and no loose data files. The original PDF provenance hashes remain valid.

The following command passed all 56 tests, including six archive integrity,
restoration, conflict, and path-safety tests::

   OPENBLAS_NUM_THREADS=1 python -m unittest discover -s docs/notes -p 'test_*.py'
