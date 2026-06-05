> 返回 [[summary]]

## [18:15:30]
- `ls outputs/2026-01-23 | wc -l` (确认实验总数: 44)
- `for d in outputs/2026-01-23/*; do [ ! -f "$d/*test_result.csv" ] && echo "$d"; done` (定位未完成实验)
- `cat downstream_train_rerun.sh` (核对重跑配置，包含 scale 消融实验)
