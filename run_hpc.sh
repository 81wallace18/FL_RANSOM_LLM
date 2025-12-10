#!/bin/bash
# Script simples para executar no cluster HPC
# Uso: qsub run_hpc.sh

#PBS -N FL-RANSOM
#PBS -l nodes=1:ncpus=56:ngpus=1:gpus=v100
#PBS -l walltime=24:00:00
#PBS -q CCAD_QGPU
#PBS -o job_output_$PBS_JOBID.out
#PBS -e job_error_$PBS_JOBID.err
#PBS -V

# Carregar módulos
module load python/3.11
module load cuda/11.8
module load gcc/11.3

# Diretório do projeto
cd $HOME/FL_RANSOM_LLM

# Ativar ambiente UV
source .venv/bin/activate

# Configurar CUDA
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=8

echo "Iniciando Federated Learning..."
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
nvidia-smi

# Executar aplicação
python main.py --config configs/config.yaml

echo "Job concluído em $(date)"