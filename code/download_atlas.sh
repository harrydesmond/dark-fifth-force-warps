#!/bin/bash
# Download the NSA v1_0 atlas child images (public SDSS DR17) listed in catalogue/atlas_urls.txt into paths.ATLAS
# (default <repo>/atlas): 4357 files, 7.2 GB. Resumable: files already present and passing a gzip test are skipped.
# Usage: bash download_atlas.sh [parallel downloads, default 8] [URL list, default catalogue/atlas_urls.txt]
set -uo pipefail
cd "$(dirname "$0")"
LIST=$(realpath "${2:-$(python3 -c 'import paths; print(paths.cat("atlas_urls.txt"))')}")
ATLAS=$(python3 -c 'import paths; print(paths.ATLAS)')
NP=${1:-8}
mkdir -p "$ATLAS" && cd "$ATLAS"
xargs -P "$NP" -I{} sh -c 'f=$(basename {}); if [ -s $f ] && gzip -t $f 2>/dev/null; then exit 0; fi; wget -q --tries=5 --waitretry=10 --timeout=300 -O $f.part {} && gzip -t $f.part && mv $f.part $f || { rm -f $f.part; echo FAIL {}; }' < "$LIST"
echo "$(ls *.fits.gz 2>/dev/null | wc -l) of $(wc -l < "$LIST") images in $ATLAS"
