#!/usr/bin/env bash
# Release check: rebuild the submission from a fresh clone of HEAD and compare it with
# SUBMISSION_SHA256.txt.
#
#   release/check.sh [--bundle DIR] [--quick] [--mac N] [--linux "arm64 amd64" | --linux none]
#                    [--no-tests] [--python PATH] [--cpus N] [--workdir DIR] [--keep-logs DIR]
#
# Steps, all inside one temporary folder that is deleted on exit, also on failure:
#   1. clone HEAD of this repository and record the commit;
#   2. copy the four BAR-ATH tactical files from the bundle into a minimal bundle (tactical/ only);
#   3. N macOS builds (default 2), each in a fresh venv from Homebrew python3.12 with
#      dfl_submission/requirements.txt;
#   4. one build per Linux platform (default arm64 and amd64) in the pinned python:3.12.14-slim image;
#   5. for each build, print only the validator's OK/FAIL line and the SHA-256, compared with
#      SUBMISSION_SHA256.txt;
#   6. run tests/run_all.py in the clone, in a fresh venv with dfl_submission/requirements-training.txt.
# The build command is the single line in release/BUILD_COMMAND, read from the commit under test.
#
# The output CSVs and the build logs derive from competition data: they are never printed, and they
# are deleted with the bundle copy. --keep-logs copies the build logs to DIR for a person to read.
#
# Exit status: 0 if every build matches and validates and every test passes, 1 otherwise,
# 2 for a usage or setup error.

set -u -o pipefail

IMAGE="python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f"
BUNDLE="/Users/derekko/Projects/more-than-a-hack-2027/data"
PYTHON="/opt/homebrew/bin/python3.12"
MAC_BUILDS=2
LINUX_PLATFORMS="arm64 amd64"
RUN_TESTS=1
DOCKER_CPUS=4
WORKDIR_BASE="${TMPDIR:-/tmp}"
KEEP_LOGS=""
TACTICAL_FILES="BAR-ATH_tactical_atd.xml BAR-ATH_tactical_atd.txt BAR-ATH_tactical_homography.csv BAR-ATH_tactical_smart_tagging.xml"

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
fail_setup() { echo "release check: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE="${2:?}"; shift 2 ;;
    --quick) MAC_BUILDS=1; LINUX_PLATFORMS="none"; shift ;;
    --mac) MAC_BUILDS="${2:?}"; shift 2 ;;
    --linux) LINUX_PLATFORMS="${2:?}"; shift 2 ;;
    --no-tests) RUN_TESTS=0; shift ;;
    --python) PYTHON="${2:?}"; shift 2 ;;
    --cpus) DOCKER_CPUS="${2:?}"; shift 2 ;;
    --workdir) WORKDIR_BASE="${2:?}"; shift 2 ;;
    --keep-logs) KEEP_LOGS="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done
[ "$LINUX_PLATFORMS" = "none" ] && LINUX_PLATFORMS=""
case "$MAC_BUILDS" in ''|*[!0-9]*) fail_setup "--mac takes a number" ;; esac
for platform in $LINUX_PLATFORMS; do
  case "$platform" in arm64|amd64) ;; *) fail_setup "unknown Linux platform '$platform' (arm64, amd64 or none)" ;; esac
done

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)" || fail_setup "not inside the repository"
[ -x "$PYTHON" ] || fail_setup "no Python at $PYTHON (use --python)"
for name in $TACTICAL_FILES; do
  [ -f "$BUNDLE/tactical/$name" ] || fail_setup "missing $BUNDLE/tactical/$name (use --bundle)"
done
if [ -n "$LINUX_PLATFORMS" ]; then
  docker info >/dev/null 2>&1 || fail_setup "Docker is not running (or use --linux none)"
fi

WORK="$(mktemp -d "${WORKDIR_BASE%/}/mtah-release.XXXXXX")" || fail_setup "cannot create a work folder"
CONTAINERS=""

cleanup() {
  for name in $CONTAINERS; do docker rm -f "$name" >/dev/null 2>&1; done
  if [ -n "$KEEP_LOGS" ] && [ -d "$WORK/logs" ]; then
    mkdir -p "$KEEP_LOGS" && cp -R "$WORK/logs/." "$KEEP_LOGS/" && echo "Build logs kept in $KEEP_LOGS (not for AI tools)."
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

sha256() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi
}

FAILURES=0
SUMMARY=""
record() {  # name, validator line, sha, verdict
  SUMMARY="${SUMMARY}$(printf '%-14s %-22s %s  %s' "$1" "$2" "$3" "$4")"$'\n'
  [ "$4" = "match" ] || FAILURES=$((FAILURES + 1))
}

# -- 1. clone ----------------------------------------------------------------------------------------
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
echo "== Clone of $REPO at $(git -C "$REPO" log -1 --format='%h %s' "$COMMIT")"
dirty="$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
[ "$dirty" = "0" ] || echo "   note: $dirty uncommitted change(s) in the working tree are not part of this check"
CLONE="$WORK/clone"
git clone -q "$REPO" "$CLONE" && git -C "$CLONE" checkout -q --detach "$COMMIT" || fail_setup "clone failed"
echo "   commit $COMMIT"

EXPECTED="$(awk 'NR == 1 {print $1}' "$CLONE/SUBMISSION_SHA256.txt")"
[ ${#EXPECTED} -eq 64 ] || fail_setup "SUBMISSION_SHA256.txt has no SHA-256 at $COMMIT"
COMMAND_FILE="$CLONE/release/BUILD_COMMAND"
if [ ! -f "$COMMAND_FILE" ]; then
  COMMAND_FILE="$REPO/release/BUILD_COMMAND"
  echo "   note: release/BUILD_COMMAND is not committed at $COMMIT; using the working-tree copy"
fi
COMMAND_LINE="$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$COMMAND_FILE")"
[ "$(printf '%s\n' "$COMMAND_LINE" | wc -l | tr -d ' ')" = "1" ] || fail_setup "BUILD_COMMAND must hold exactly one command line"
read -r -a TEMPLATE <<< "$COMMAND_LINE"
case "$COMMAND_LINE" in *"{out}"*) ;; *) fail_setup "BUILD_COMMAND has no {out} placeholder" ;; esac
PLACEHOLDER="{out}"
build_args() {  # output path -> BUILD_ARGS
  BUILD_ARGS=()
  for arg in "${TEMPLATE[@]}"; do BUILD_ARGS+=("${arg//"$PLACEHOLDER"/$1}"); done
}
echo "   expected SHA-256 $EXPECTED"
echo "   build: python $COMMAND_LINE"

# -- 2. minimal bundle ---------------------------------------------------------------------------------
mkdir -p "$WORK/bundle/tactical" "$WORK/out" "$WORK/logs"
for name in $TACTICAL_FILES; do
  cp -L "$BUNDLE/tactical/$name" "$WORK/bundle/tactical/$name" || fail_setup "could not copy $name"
done
echo "== Minimal bundle: tactical/ with the four BAR-ATH files"

verdict() {  # validator line, sha
  case "$1" in OK*) [ "$2" = "$EXPECTED" ] && echo match || echo MISMATCH ;; *) echo "VALIDATOR FAILED" ;; esac
}

# -- 3. macOS builds -----------------------------------------------------------------------------------
i=1
while [ "$i" -le "$MAC_BUILDS" ]; do
  name="macos-$i"
  venv="$WORK/venv-$name"
  echo "== $name: fresh venv from $("$PYTHON" --version 2>&1) ($PYTHON)"
  if ! { "$PYTHON" -m venv "$venv" && "$venv/bin/python" -m pip install -q -r "$CLONE/dfl_submission/requirements.txt"; } \
       > "$WORK/logs/$name-pip.log" 2>&1; then
    tail -n 20 "$WORK/logs/$name-pip.log"
    record "$name" "-" "-" "PIP FAILED"; i=$((i + 1)); continue
  fi
  csv="$WORK/out/$name.csv"
  build_args "$csv"
  (cd "$CLONE" && MTAH_DATA_DIR="$WORK/bundle" PYTHONDONTWRITEBYTECODE=1 nice -n 10 "$venv/bin/python" "${BUILD_ARGS[@]}") \
    > "$WORK/logs/$name-build.log" 2>&1
  code=$?
  if [ "$code" -ne 0 ] || [ ! -f "$csv" ]; then
    echo "   build failed with exit code $code (log withheld: it derives from competition data)"
    record "$name" "-" "-" "BUILD FAILED"; i=$((i + 1)); continue
  fi
  line="$(cd "$CLONE" && "$venv/bin/python" evaluation/validate_submission.py "$csv" 2>&1 | grep -E '^(OK|FAIL)' | head -n 1)"
  sha="$(sha256 "$csv")"
  rm -f "$csv"
  echo "   $line"
  echo "   $sha"
  record "$name" "${line:-no validator line}" "$sha" "$(verdict "$line" "$sha")"
  i=$((i + 1))
done

# -- 4. Linux builds -----------------------------------------------------------------------------------
for platform in $LINUX_PLATFORMS; do
  name="linux-$platform"
  out="$WORK/out/$name"
  mkdir -p "$out"
  container="mtah-release-$$-$platform"
  CONTAINERS="$CONTAINERS $container"
  build_args "/out/submission.csv"
  echo "== $name: $IMAGE"
  docker run --rm --name "$container" --platform "linux/$platform" --cpus "$DOCKER_CPUS" \
    -v "$CLONE":/repo:ro -v "$WORK/bundle":/bundle:ro -v "$out":/out \
    -e MTAH_DATA_DIR=/bundle -e PYTHONDONTWRITEBYTECODE=1 "$IMAGE" bash -c '
      cd /repo || exit 90
      { python -m venv /venv && /venv/bin/pip install -q -r dfl_submission/requirements.txt; } > /out/pip.log 2>&1 \
        || { echo "PIP failed"; exit 91; }
      echo "PYTHON $(python --version 2>&1) $(uname -m)"
      /venv/bin/python "$@" > /out/build.log 2>&1
      code=$?
      echo "BUILD $code"
      [ "$code" -eq 0 ] && [ -f /out/submission.csv ] || exit 92
      echo "VALIDATOR $(/venv/bin/python evaluation/validate_submission.py /out/submission.csv 2>&1 | grep -E "^(OK|FAIL)" | head -n 1)"
      echo "SHA $(sha256sum /out/submission.csv | cut -d" " -f1)"
    ' _ "${BUILD_ARGS[@]}" > "$WORK/logs/$name-docker.out" 2> "$WORK/logs/$name-docker.err"
  code=$?
  [ -f "$out/build.log" ] && mv "$out/build.log" "$WORK/logs/$name-build.log"
  [ -f "$out/pip.log" ] && mv "$out/pip.log" "$WORK/logs/$name-pip.log"
  rm -f "$out/submission.csv"
  grep '^PYTHON ' "$WORK/logs/$name-docker.out" | sed 's/^PYTHON /   /'
  line="$(sed -n 's/^VALIDATOR //p' "$WORK/logs/$name-docker.out")"
  sha="$(sed -n 's/^SHA //p' "$WORK/logs/$name-docker.out")"
  if [ "$code" -ne 0 ] || [ -z "$sha" ]; then
    if grep -q '^PIP failed' "$WORK/logs/$name-docker.out"; then
      tail -n 20 "$WORK/logs/$name-pip.log" 2>/dev/null; record "$name" "-" "-" "PIP FAILED"
    elif grep -q '^BUILD ' "$WORK/logs/$name-docker.out"; then
      echo "   build failed (log withheld: it derives from competition data)"; record "$name" "-" "-" "BUILD FAILED"
    else
      tail -n 5 "$WORK/logs/$name-docker.err"; record "$name" "-" "-" "DOCKER FAILED"
    fi
    continue
  fi
  echo "   $line"
  echo "   $sha"
  record "$name" "${line:-no validator line}" "$sha" "$(verdict "$line" "$sha")"
done

# -- 5. tests ------------------------------------------------------------------------------------------
TESTS="not run"
if [ "$RUN_TESTS" -eq 1 ]; then
  echo "== Tests: tests/run_all.py in the clone (fresh venv, requirements-training.txt)"
  venv="$WORK/venv-tests"
  if ! { "$PYTHON" -m venv "$venv" && "$venv/bin/python" -m pip install -q -r "$CLONE/dfl_submission/requirements-training.txt"; } \
       > "$WORK/logs/tests-pip.log" 2>&1; then
    tail -n 20 "$WORK/logs/tests-pip.log"; TESTS="PIP FAILED"; FAILURES=$((FAILURES + 1))
  elif (cd "$CLONE" && nice -n 10 "$venv/bin/python" tests/run_all.py); then
    TESTS="passed"
  else
    TESTS="FAILED"; FAILURES=$((FAILURES + 1))
  fi
fi

# -- summary -------------------------------------------------------------------------------------------
echo
echo "== Release check of $COMMIT"
echo "   expected       $EXPECTED"
printf '%s' "$SUMMARY" | sed 's/^/   /'
echo "   tests          $TESTS"
builds=$((MAC_BUILDS + $(echo $LINUX_PLATFORMS | wc -w)))
if [ "$builds" -eq 0 ]; then echo "   (no builds requested)"; fi
if [ "$FAILURES" -eq 0 ]; then
  echo "RELEASE CHECK PASSED ($builds build(s), tests $TESTS)"
  exit 0
fi
echo "RELEASE CHECK FAILED ($FAILURES problem(s))"
exit 1
