.PHONY: clean clean_all configure debug release test test_debug test_release check

PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

EXTENSION_NAME=ddsketch

# Set to 0 to use stable C API (allows loading with compatible DuckDB versions)
# Set to 1 to pin to exact DuckDB version (for unstable API features)
USE_UNSTABLE_C_API=0

# IMPORTANT: This is the C API version, NOT the DuckDB release version!
# DuckDB v1.4.x uses C API version v1.2.0
# Check with: duckdb -c "PRAGMA version;" and look at what API it expects
TARGET_DUCKDB_VERSION=v1.2.0

all: configure debug

# Include makefiles from DuckDB
include extension-ci-tools/makefiles/c_api_extensions/base.Makefile
include extension-ci-tools/makefiles/c_api_extensions/rust.Makefile

configure: venv platform extension_version

debug: build_extension_library_debug build_extension_with_metadata_debug
release: build_extension_library_release build_extension_with_metadata_release

test: test_debug
test_debug: test_extension_debug
test_release: test_extension_release

clean: clean_build clean_rust
clean_all: clean_configure clean

# Full check: build, run unit tests, run integration tests
check: release
	cargo test
	./test/run_integration_tests.sh
