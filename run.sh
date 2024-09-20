#!/bin/bash
#PBS -l select=1:ncpus=128:ngpus=8:mem=256GB:accelerator_model=a100
#PBS -l walltime=47:59:00
#PBS -A UP
#PBS -N model_run
#PBS -j oe
#PBS -o "model_run_babylm_3.log"

## Log-File definieren
export LOGFILE=$PBS_O_WORKDIR/$PBS_JOBNAME"."$PBS_JOBID".log"

##Scratch-Laufwerk definieren und erzeugen
SCRATCHDIR=/gpfs/scratch/$USER/$PBS_JOBID
mkdir -p "$SCRATCHDIR" 

##Information zum Start in das Log-File schreiben
cd $PBS_O_WORKDIR  
echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" START" > $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE  

module load Python/3.11.8

##Daten vom Arbeitsverzeichnis auf das Scratch-Laufwerk kopieren
cp -r $PBS_O_WORKDIR/* $SCRATCHDIR/.
cd $SCRATCHDIR
rm $PBS_JOBNAME"."$PBS_JOBID".log"

cd $PBS_O_WORKDIR
cd ..
source venv6/bin/activate
cd $SCRATCHDIR
sh run_several.sh babylm3 1

##Daten zurück kopieren
cp -r "$SCRATCHDIR"/log/* "$PBS_O_WORKDIR"/log/.
cd $PBS_O_WORKDIR 

##Verfügbare Informationen zum Auftrag in das Log-File schreiben
echo >> $LOGFILE
qstat -f $PBS_JOBID >> $LOGFILE   

echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" END" >> $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE
