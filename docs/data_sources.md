# Data Sources

## Poly domain

### GTZAN
- Purpose: quick poly/general music prototype dataset
- Contents: 1,000 tracks, 30s each, 10 genres, 22,050 Hz mono WAV
- Source: TensorFlow Datasets GTZAN catalog
- Local path: $DATA_ROOT/poly/gtzan

### FMA-small
- Purpose: larger and more varied poly/general music domain
- Contents: 8,000 tracks, 30s each, 8 genres
- Source: Free Music Archive dataset / FMA-small
- Local path: $DATA_ROOT/poly/fma_small

## Chip domain

### NES-MDB
- Purpose: chiptune/NES target domain
- Contents: 5,278 songs from 397 NES games
- Source: NES-MDB
- Local path: $DATA_ROOT/chip/nesmdb
- Notes: Keep original symbolic files and rendered WAV files separately.
