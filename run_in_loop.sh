####################################################################################
###### CONFIG 1 ####################################################################
####################################################################################
N=80
MAX_ITERATION=50
OBSERVATION_COUNT=50
THRESHOLD=0.8
RUN_NUMBER=1
NUMBER_OF_RUNS=1
EXPERIMENT_SETUP_NUM=0

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS $EXPERIMENT_SETUP_NUM
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD-EMEEM(NO)"
mkdir -p $FOLDER_NAME
rm -rf "Layered Monte Carlo HMM."*"INFO"*
rm -rf "Monte Carlo HMM."*"INFO"*
rm -rf "Monte-Carlo-HMM"*"INFO"*
mv "Layered Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte-Carlo-HMM"*"WARNING"* $FOLDER_NAME


####################################################################################
###### CONFIG 2 ####################################################################
####################################################################################
N=80
MAX_ITERATION=50
OBSERVATION_COUNT=50
THRESHOLD=0.8
RUN_NUMBER=1
NUMBER_OF_RUNS=1
EXPERIMENT_SETUP_NUM=0

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS $EXPERIMENT_SETUP_NUM
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD-EMEEM(WITH)"
mkdir -p $FOLDER_NAME
rm -rf "Layered Monte Carlo HMM."*"INFO"*
rm -rf "Monte Carlo HMM."*"INFO"*
rm -rf "Monte-Carlo-HMM"*"INFO"*
mv "Layered Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte-Carlo-HMM"*"WARNING"* $FOLDER_NAME


####################################################################################
###### CONFIG 3 ####################################################################
####################################################################################
N=80
MAX_ITERATION=50
OBSERVATION_COUNT=50
THRESHOLD=0.8
RUN_NUMBER=1
NUMBER_OF_RUNS=1
EXPERIMENT_SETUP_NUM=0

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS $EXPERIMENT_SETUP_NUM
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD-EEMM(NO)"
mkdir -p $FOLDER_NAME
rm -rf "Layered Monte Carlo HMM."*"INFO"*
rm -rf "Monte Carlo HMM."*"INFO"*
rm -rf "Monte-Carlo-HMM"*"INFO"*
mv "Layered Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte-Carlo-HMM"*"WARNING"* $FOLDER_NAME


####################################################################################
###### CONFIG 4 ####################################################################
####################################################################################
N=80
MAX_ITERATION=50
OBSERVATION_COUNT=50
THRESHOLD=0.8
RUN_NUMBER=1
NUMBER_OF_RUNS=1
EXPERIMENT_SETUP_NUM=0

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS $EXPERIMENT_SETUP_NUM
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD-EEMM(WITH)"
mkdir -p $FOLDER_NAME
rm -rf "Layered Monte Carlo HMM."*"INFO"*
rm -rf "Monte Carlo HMM."*"INFO"*
rm -rf "Monte-Carlo-HMM"*"INFO"*
mv "Layered Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte-Carlo-HMM"*"WARNING"* $FOLDER_NAME


####################################################################################
###### CONFIG 5 ####################################################################
####################################################################################
N=80
MAX_ITERATION=50
OBSERVATION_COUNT=50
THRESHOLD=0.8
RUN_NUMBER=1
NUMBER_OF_RUNS=1
EXPERIMENT_SETUP_NUM=0

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS $EXPERIMENT_SETUP_NUM
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD-EMSEMS(NO)"
mkdir -p $FOLDER_NAME
rm -rf "Layered Monte Carlo HMM."*"INFO"*
rm -rf "Monte Carlo HMM."*"INFO"*
rm -rf "Monte-Carlo-HMM"*"INFO"*
mv "Layered Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte-Carlo-HMM"*"WARNING"* $FOLDER_NAME


####################################################################################
###### CONFIG 6 ####################################################################
####################################################################################
N=80
MAX_ITERATION=50
OBSERVATION_COUNT=50
THRESHOLD=0.8
RUN_NUMBER=1
NUMBER_OF_RUNS=1
EXPERIMENT_SETUP_NUM=0

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS $EXPERIMENT_SETUP_NUM
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD-EMSEMS(WITH)"
mkdir -p $FOLDER_NAME
rm -rf "Layered Monte Carlo HMM."*"INFO"*
rm -rf "Monte Carlo HMM."*"INFO"*
rm -rf "Monte-Carlo-HMM"*"INFO"*
mv "Layered Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte-Carlo-HMM"*"WARNING"* $FOLDER_NAME