#!/bin/zsh
# Stage the two private Kaggle datasets for the D20 cloud runs (DECISIONS.md D20, cloud amendment).
# Derek uploads them; see cloud/README.md. Output: data/d20/kaggle/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
R=/Users/derekko/Projects/more-than-a-hack-2027/data
CURRENT=$R/external_temporal/train_data
AUGMENTED=data/d13/sequences_augmented
OUT=data/d20/kaggle
CODE_PATHS=(LICENSE LICENSE-SKILLCORNER dfl_submission evaluation experiments games.py starter)

# The code must be committed: the cloud run records this revision, and the repo checks it before use.
git ls-files --error-unmatch dfl_submission/zoo.py > /dev/null 2>&1 || { echo "dfl_submission/zoo.py is not committed"; exit 1; }
git diff --quiet HEAD -- ${CODE_PATHS[@]} || { echo "Uncommitted changes under the code paths; commit them first"; exit 1; }
COMMIT=$(git rev-parse HEAD)

# Only the two prepared external folders, checked against the published training manifests and their own hashes.
MTAH_DATA_DIR=/nonexistent/mtah .venv/bin/python -c "
import sys; sys.path[:0] = ['.', 'experiments']
from pathlib import Path
from dfl_submission import zoo
print(sorted(zoo.verify_inputs({'current': Path('$CURRENT'), 'augmented': Path('$AUGMENTED')})))"

rm -rf $OUT
mkdir -p $OUT/mtah-d20-sequences $OUT/mtah-d20-code
(cd $CURRENT && zip -0 -q -X $OLDPWD/$OUT/mtah-d20-sequences/train_data.zip manifest.json *.npz)
(cd $AUGMENTED && zip -0 -q -X $OLDPWD/$OUT/mtah-d20-sequences/sequences_augmented.zip manifest.json *.npz)
git archive --format=zip -o $OUT/mtah-d20-code/mtah-code.zip $COMMIT -- ${CODE_PATHS[@]}
echo $COMMIT > $OUT/mtah-d20-code/CODE_COMMIT.txt

cat > $OUT/mtah-d20-sequences/README.md <<'TXT'
Prepared training sequences for DECISIONS.md D20 (private). Derived only from external data:
DFL Bundesliga tracking and events (Bassek et al. 2025, CC BY 4.0) and SkillCorner open data (MIT).
train_data.zip: the original model set's 48 prepared halves (manifest d3891c24...).
sequences_augmented.zip: the D2-augmented set's 48 prepared halves (manifest 2828b0c3...).
TXT
cat > $OUT/mtah-d20-code/README.md <<TXT
Repository code for DECISIONS.md D20 at commit $COMMIT (private until the repository is published).
TXT
for name in mtah-d20-sequences mtah-d20-code; do
  print -r -- "{\"title\": \"$name\", \"id\": \"${KAGGLE_USERNAME:-YOUR_KAGGLE_USERNAME}/$name\", \"licenses\": [{\"name\": \"other\"}]}" \
    > $OUT/$name/dataset-metadata.json
done
echo "Staged at commit $COMMIT:"
(cd $OUT && shasum -a 256 */*.zip */CODE_COMMIT.txt && du -sh *)
