#!/bin/bash
# Monitor panel experiments and plot when complete

JOB_ID=20704695
TOTAL=55
LOG_FILE="logs/monitor_panel_jobs.log"

echo "$(date): Starting monitor for job $JOB_ID" >> $LOG_FILE

while true; do
    completed=0
    for i in $(seq 0 $((TOTAL-1))); do
        f="logs/panel_exp_${JOB_ID}_${i}.out"
        if [ -f "$f" ] && grep -q "✓ Completed" "$f" 2>/dev/null; then
            ((completed++))
        fi
    done
    
    echo "$(date): $completed/$TOTAL completed" >> $LOG_FILE
    
    if [ $completed -eq $TOTAL ]; then
        echo "$(date): All jobs completed! Running plotting script..." >> $LOG_FILE
        
        source ~/.bashrc
        conda activate mtl_ft
        python scripts/diagonal/plot_panels.py >> $LOG_FILE 2>&1
        
        echo "$(date): Plotting complete! Check figures/panels/" >> $LOG_FILE
        break
    fi
    
    # Check if any jobs still in queue
    running=$(squeue -u na658 2>/dev/null | grep -c panel_ex || echo 0)
    if [ $running -eq 0 ] && [ $completed -lt $TOTAL ]; then
        echo "$(date): WARNING - No jobs running but only $completed/$TOTAL completed. Some may have failed." >> $LOG_FILE
        echo "$(date): Running plotting anyway with available data..." >> $LOG_FILE
        
        source ~/.bashrc
        conda activate mtl_ft
        python scripts/diagonal/plot_panels.py >> $LOG_FILE 2>&1
        break
    fi
    
    # Wait 1 hour
    sleep 3600
done

echo "$(date): Monitor finished" >> $LOG_FILE
