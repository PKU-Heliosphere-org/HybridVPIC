#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-help}"
nranks="${2:-1}"
PYTHON="${PYTHON:-python3}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
uname_s="$(uname -s)"
case "$uname_s" in
  Darwin) deck="./psp_counterpropagating_icw.Darwin" ;;
  *) deck="./psp_counterpropagating_icw.Linux" ;;
esac

case "$cmd" in
  build)
    make
    ;;
  run)
    if [[ ! -x "$deck" ]]; then
      make
    fi
    if command -v srun >/dev/null 2>&1; then
      srun -n "$nranks" "$deck" 2>&1 | tee hvpic.out
    elif command -v mpirun >/dev/null 2>&1; then
      mpirun -np "$nranks" "$deck" 2>&1 | tee hvpic.out
    else
      "$deck" 2>&1 | tee hvpic.out
    fi
    ;;
  translate)
    make translate
    mkdir -p data
    if command -v mpirun >/dev/null 2>&1; then
      mpirun -np 1 ./translate_psp_icw
    else
      ./translate_psp_icw
    fi
    ;;
  demo-figures)
    "$PYTHON" scripts/visualize_counterpropagating_icw.py --demo --out figures
    ;;
  figures)
    "$PYTHON" scripts/visualize_counterpropagating_icw.py --data data --out figures
    ;;
  figures-delta)
    initial_step="${2:-0}"
    final_step="${3:-14000}"
    "$PYTHON" scripts/visualize_counterpropagating_icw.py \
      --data data \
      --out figures \
      --core-initial-csv "data/particles/core_t${initial_step}.csv" \
      --beam-initial-csv "data/particles/beam_t${initial_step}.csv" \
      --core-final-csv "data/particles/core_t${final_step}.csv" \
      --beam-final-csv "data/particles/beam_t${final_step}.csv"
    ;;
  particle-csv)
    step="${2:-0}"
    max_particles="${3:-200000}"
    mkdir -p data/particles
    "$PYTHON" scripts/convert_particle_dump_to_csv.py "particle/T.${step}/ion_c" --out "data/particles/core_t${step}.csv" --max-particles "$max_particles"
    "$PYTHON" scripts/convert_particle_dump_to_csv.py "particle/T.${step}/ion_b" --out "data/particles/beam_t${step}.csv" --max-particles "$max_particles"
    ;;
  particle-vdf)
    step="${2:-0}"
    "$PYTHON" scripts/plot_particle_vdf.py --core-csv "data/particles/core_t${step}.csv" --beam-csv "data/particles/beam_t${step}.csv" --out "figures/vdf_t${step}.png"
    ;;
  particle-delta-vdf)
    initial_step="${2:-0}"
    final_step="${3:-14000}"
    "$PYTHON" scripts/plot_particle_vdf.py \
      --core-initial-csv "data/particles/core_t${initial_step}.csv" \
      --beam-initial-csv "data/particles/beam_t${initial_step}.csv" \
      --core-final-csv "data/particles/core_t${final_step}.csv" \
      --beam-final-csv "data/particles/beam_t${final_step}.csv" \
      --out "figures/fig4_resonant_scattering.png"
    ;;
  *)
    echo "Usage: $0 {build|run [nranks]|translate|figures|figures-delta [initial_step] [final_step]|demo-figures|particle-csv [step] [max_particles]|particle-vdf [step]|particle-delta-vdf [initial_step] [final_step]}"
    exit 2
    ;;
esac
