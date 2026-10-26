"""Archive integrity, restore conflict, and path-safety checks with tiny fixtures."""

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import study_archive as archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp", prefix="snapy-archive-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_archive(self, name="samples/run.csv", bad_hash=False):
        path = self.root / "data.tar.gz"
        data = b"time,amplitude\n0,1\n"
        digest = "0" * 64 if bad_hash else hashlib.sha256(data).hexdigest()
        entries = {name: data, archive.CHECKSUMS: f"{digest}  {name}\n".encode()}
        with tarfile.open(path, "w:gz") as tar:
            for filename, value in entries.items():
                member = tarfile.TarInfo(filename)
                member.size = len(value)
                tar.addfile(member, io.BytesIO(value))
        return path

    def test_round_trip_and_identical_restore(self):
        contents = archive.verified_files(self.make_archive())
        destination = self.root / "restore"
        self.assertEqual(archive.restore(contents, destination), 1)
        self.assertEqual(archive.restore(contents, destination), 0)
        self.assertEqual(
            (destination / "samples/run.csv").read_bytes(), contents["samples/run.csv"]
        )

    def test_bad_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            archive.verified_files(self.make_archive(bad_hash=True))

    def test_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            archive.verified_files(self.make_archive("../outside.csv"))

    def test_conflict_checked_before_any_writes(self):
        (self.root / "changed.csv").write_bytes(b"new work")
        with self.assertRaises(FileExistsError):
            archive.restore({"new.csv": b"a", "changed.csv": b"old"}, self.root)
        self.assertFalse((self.root / "new.csv").exists())

    def test_symlink_destination_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        destination = self.root / "restore"
        destination.mkdir()
        (destination / "samples").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink destination"):
            archive.restore({"samples/run.csv": b"data"}, destination)

    def test_loose_file_has_precedence_and_archive_fallback_is_verified(self):
        self.make_archive()
        path = self.root / "samples/run.csv"
        with (
            patch.object(archive, "HERE", self.root),
            patch.object(
                archive, "verified_files", return_value={"samples/run.csv": b"archived"}
            ) as verify,
        ):
            self.assertEqual(archive.read_data(path), b"archived")
            path.parent.mkdir()
            path.write_bytes(b"updated")
            self.assertEqual(archive.read_data(path), b"updated")
            self.assertEqual(verify.call_count, 1)


if __name__ == "__main__":
    unittest.main()
