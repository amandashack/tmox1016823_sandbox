#!/bin/bash
#SBATCH --job-name=process_ports
#SBATCH --output=process_ports_%j.out
#SBATCH --error=process_ports_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       # Adjust based on the number of ports or cores available
#SBATCH --mem=16G               # Adjust based on your memory requirements
#SBATCH --time=02:00:00         # Adjust based on the expected runtime
#SBATCH --partition=standard    # Adjust based on available partitions

source activate ps-amanda  # Replace with your conda environment name

# Navigate to the directory containing your script
cd /path/to/your/script

# Run your Python script
python process_ports_batch.py
