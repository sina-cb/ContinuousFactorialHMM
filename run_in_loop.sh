####################################################################################
###### CONFIG 1 ####################################################################
####################################################################################
N=50
MAX_ITERATION=100
OBSERVATION_COUNT=50
THRESHOLD=0.4
RUN_NUMBER=1
NUMBER_OF_RUNS=1

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD"
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
N=50
MAX_ITERATION=100
OBSERVATION_COUNT=100
THRESHOLD=0.4
RUN_NUMBER=1
NUMBER_OF_RUNS=1

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD"
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
N=50
MAX_ITERATION=100
OBSERVATION_COUNT=150
THRESHOLD=0.4
RUN_NUMBER=1
NUMBER_OF_RUNS=1

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD"
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
N=50
MAX_ITERATION=100
OBSERVATION_COUNT=200
THRESHOLD=0.4
RUN_NUMBER=1
NUMBER_OF_RUNS=1

while test "$RUN_NUMBER" -lt "20"; do
	./build/Monte-Carlo-HMM $N $MAX_ITERATION $OBSERVATION_COUNT $THRESHOLD $RUN_NUMBER $NUMBER_OF_RUNS
	let "RUN_NUMBER = RUN_NUMBER + 1"
done

# Move all collected data into a folder
FOLDER_NAME="LMCHMM-dump_data-$N-$MAX_ITERATION-$OBSERVATION_COUNT-$THRESHOLD"
mkdir -p $FOLDER_NAME
rm -rf "Layered Monte Carlo HMM."*"INFO"*
rm -rf "Monte Carlo HMM."*"INFO"*
rm -rf "Monte-Carlo-HMM"*"INFO"*
mv "Layered Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte Carlo HMM."*"WARNING"* $FOLDER_NAME
mv "Monte-Carlo-HMM"*"WARNING"* $FOLDER_NAME
