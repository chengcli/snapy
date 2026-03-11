// C/C++
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// archive
#include <archive.h>
#include <archive_entry.h>

// kintera
#include <kintera/utils/serialize.hpp>

// snap
#include <snap/layout/layout.hpp>

namespace fs = std::filesystem;

namespace snap {

// -------------------------
// Small helpers
// -------------------------

struct RestartFields {
  std::string basename;
  std::string blockid;
  std::string filenumber;
};

static RestartFields parse_part_filename(const std::string& name) {
  constexpr std::string_view suffix = ".part";

  if (name.size() <= suffix.size() ||
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    throw std::invalid_argument("filename does not end with .part");
  }

  // Strip ".part"
  const std::string_view core(name.data(), name.size() - suffix.size());

  // Find last two dots
  const size_t dot2 = core.rfind('.');
  if (dot2 == std::string::npos) {
    throw std::invalid_argument("filename missing filenumber field");
  }

  const size_t dot1 = core.rfind('.', dot2 - 1);
  if (dot1 == std::string::npos) {
    throw std::invalid_argument("filename missing block_id field");
  }

  RestartFields out;
  out.basename = std::string(core.substr(0, dot1));
  out.blockid = std::string(core.substr(dot1 + 1, dot2 - dot1 - 1));
  out.filenumber = std::string(core.substr(dot2 + 1));

  if (out.basename.empty() || out.blockid.empty() || out.filenumber.empty()) {
    throw std::invalid_argument("one or more filename fields are empty");
  }

  return out;
}

static std::string dtype_to_string(const at::ScalarType t) {
  // at::toString exists in many builds; this is safe enough.
  return std::string(at::toString(t));
}

static std::string device_to_string(const at::Device& d) {
  std::ostringstream oss;
  oss << d;
  return oss.str();
}

static std::string shape_to_string(const at::Tensor& t) {
  std::ostringstream oss;
  oss << "(";
  for (int64_t i = 0; i < t.dim(); ++i) {
    oss << t.size(i);
    if (i + 1 < t.dim()) oss << ", ";
  }
  oss << ")";
  return oss.str();
}

static bool ends_with(std::string const& s, std::string const& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool is_tar_archive(std::string const& path) {
  if (!fs::is_regular_file(path)) return false;

  struct archive* ar = archive_read_new();
  if (!ar) return false;

  archive_read_support_filter_all(ar);
  archive_read_support_format_all(ar);

  // Try opening as an archive; if it succeeds, treat as tar-like.
  int r = archive_read_open_filename(ar, path.c_str(), 10240);
  if (r != ARCHIVE_OK) {
    archive_read_free(ar);
    return false;
  }

  // Some files might be recognized as other archive formats too; in practice
  // this matches Python's "is_tarfile" intent well.
  archive_read_close(ar);
  archive_read_free(ar);
  return true;
}

// Create a unique temp file path; not bulletproof, but good enough
static fs::path make_temp_path(std::string_view suffix) {
  fs::path dir = fs::temp_directory_path();

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (int tries = 0; tries < 20; ++tries) {
    uint64_t r = dis(gen);
    std::ostringstream name;
    name << "tmp_" << std::hex << r << suffix;
    fs::path p = dir / name.str();
    if (!fs::exists(p)) return p;
  }

  // Fallback (very unlikely to collide)
  return dir / ("tmp_fallback" + std::string(suffix));
}

static Variables load_pt_from_tar(struct archive* ar,
                                  struct archive_entry* entry) {
  const char* name_c = archive_entry_pathname(entry);
  std::string member_name =
      name_c ? std::string{name_c} : std::string{"<unknown>"};

  // Extract this entry into a temporary file (TorchScript loader prefers
  // real/seekable file)
  fs::path tmp_path = make_temp_path(".part");
  std::ofstream out(tmp_path, std::ios::binary);
  if (!out) {
    std::cerr << "\n=== " << member_name << " ===\n";
    std::cerr << "  ERROR: could not create temp file: " << tmp_path.string()
              << "\n";
    // Must still consume/skip entry data:
    archive_read_data_skip(ar);
    return {};
  }

  std::vector<char> buf(1 << 20);
  while (true) {
    la_ssize_t n = archive_read_data(ar, buf.data(), buf.size());
    if (n == 0) break;  // end of this entry
    if (n < 0) {
      std::cerr << "\n=== " << member_name << " ===\n";
      std::cerr << "  ERROR: could not extract file from tar: "
                << archive_error_string(ar) << "\n";
      out.close();
      std::error_code ec;
      fs::remove(tmp_path, ec);
      return {};
    }
    out.write(buf.data(), static_cast<std::streamsize>(n));
    if (!out) {
      std::cerr << "\n=== " << member_name << " ===\n";
      std::cerr << "  ERROR: failed writing temp file\n";
      out.close();
      std::error_code ec;
      fs::remove(tmp_path, ec);
      return {};
    }
  }

  out.flush();
  out.close();

  // load the extracted .part
  auto vars = kintera::load_tensors(tmp_path.string());

  // remove empty tensors (if any)
  for (auto it = vars.begin(); it != vars.end();) {
    if (!it->second.defined() || it->second.numel() == 0) {
      it = vars.erase(it);
    } else {
      ++it;
    }
  }

  // Cleanup
  std::error_code ec;
  fs::remove(tmp_path, ec);

  return vars;
}

Variables load_restart(std::string const& path, int block_rank) {
  // Dispatch based on whether `path` is a .part file or a tar archive.
  if (is_tar_archive(path)) {
    struct archive* ar = archive_read_new();
    if (!ar) {
      std::cerr << path << ": failed to allocate archive reader\n";
      return {};
    }

    archive_read_support_filter_all(ar);
    archive_read_support_format_all(ar);

    int r = archive_read_open_filename(ar, path.c_str(), 10240);
    if (r != ARCHIVE_OK) {
      std::cerr << path
                << ": failed to open archive: " << archive_error_string(ar)
                << "\n";
      archive_read_free(ar);
      return {};
    }

    bool found_part = false;

    struct archive_entry* entry = nullptr;
    while ((r = archive_read_next_header(ar, &entry)) == ARCHIVE_OK) {
      const char* name_c = archive_entry_pathname(entry);
      std::string name = name_c ? std::string{name_c} : std::string{};

      if (ends_with(name, ".part")) {
        found_part = true;
        auto out = parse_part_filename(name);
        // find the block rank number after "block"
        int rank = std::stoi(out.blockid.substr(5, out.blockid.size() - 5));
        if (rank != block_rank) {
          // Not for this rank; skip
          archive_read_data_skip(ar);
        } else {
          return load_pt_from_tar(ar, entry);
        }

        // Note: consume the entry data (via archive_read_data*)
        // or skip it, otherwise the next header read will misbehave.
      } else {
        // Skip non-.part entries quickly
        archive_read_data_skip(ar);
      }
    }

    if (!found_part) {
      std::cerr << path << ": no .part files found in tar archive\n";
    }

    if (r != ARCHIVE_EOF && r != ARCHIVE_OK) {
      std::cerr << path
                << ": error while reading archive: " << archive_error_string(ar)
                << "\n";
    }

    archive_read_close(ar);
    archive_read_free(ar);
  } else {
    // Treat as a single .part TorchScript file
    std::cout << "single .part file detected\n";
    return kintera::load_tensors(path);
  }

  return {};
}

}  // namespace snap
