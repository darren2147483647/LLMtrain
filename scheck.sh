#!/bin/bash
# 一次檢查自己所有的 Slurm 工作、工作裡的程式（nohup 背景程式、tunnel）和最近結束的工作。
# 用法：sh scheck.sh   （在登入節點或 tunnel 裡都可以；只查一次，不要放進迴圈或 watch）

line() { echo "======== $1 ========"; }

line "排隊中 / 執行中的工作"
squeue -u "$USER" -o "%.10i %.16j %.3t %.10M %.10L %.6D %R"
echo "（ST：R=執行中，PD=排隊中；TIME=已執行；TIME_LEFT=剩餘時間）"

running=$(squeue -h -u "$USER" -t R -o %i)
for job in $running; do
    name=$(squeue -h -j "$job" -o %j)
    out=$(scontrol show job "$job" | sed -n 's/^ *StdOut=//p')
    line "工作 $job（$name）"

    echo "-- 輸出檔最後 3 行：$out"
    if [ -r "$out" ]; then tail -n 3 "$out"; else echo "（讀不到輸出檔）"; fi

    # 到工作所在的節點列出自己的程式（含 nohup 背景程式、code tunnel、python）。
    # --overlap：共用工作已分配的資源，不另外申請，也不影響原本的程式。
    echo "-- 工作裡的程式："
    timeout 30 srun --jobid="$job" --overlap --ntasks=1 --quiet \
        ps -u "$USER" -o pid=,etime=,args= 2>/dev/null \
        | grep -E "python|llamafactory|code tunnel|nohup" \
        | grep -v -E "grep|ps -u" \
        || echo "（沒有找到 python / llamafactory / tunnel 程式，或查詢逾時）"
done

line "這台機器上的程式（$(hostname)）"
# 在登入節點上看到 python / llamafactory，代表有東西跑在登入節點上，應該停掉改用 sbatch。
ps -u "$USER" -o pid=,etime=,args= \
    | grep -E "python|llamafactory|code tunnel" \
    | grep -v -E "grep|ps -u" \
    || echo "（沒有）"
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "（目前在登入節點上；上面如果有 python / llamafactory，請停掉並改用 sbatch）"
fi

line "今天結束的工作"
sacct -X -u "$USER" -S today -s CD,F,CA,TO,OOM,NF \
    -o JobID%10,JobName%16,State%12,Elapsed,ExitCode 2>/dev/null \
    || echo "（無法查詢 sacct）"
