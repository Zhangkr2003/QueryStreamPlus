#!/usr/bin/env bash

case "${VISUAL_BUDGET:-4k}" in
  2k|2K)
    budget_keep_rate=0.50
    budget_qim_read=512
    budget_im_read=512
    ;;
  4k|4K)
    budget_keep_rate=0.75
    budget_qim_read=1024
    budget_im_read=1536
    ;;
  6k|6K)
    budget_keep_rate=0.75
    budget_qim_read=2048
    budget_im_read=2560
    ;;
  *)
    echo "VISUAL_BUDGET must be 2k, 4k, or 6k." >&2
    exit 2
    ;;
esac

keep_rate="${KEEP_RATE:-$budget_keep_rate}"
qim_read_budget="${QIM_READ_BUDGET:-$budget_qim_read}"
im_read_budget="${IM_READ_BUDGET:-$budget_im_read}"
qim_capacity="${QIM_CAPACITY:-2048}"
im_capacity="${IM_CAPACITY:-3072}"
