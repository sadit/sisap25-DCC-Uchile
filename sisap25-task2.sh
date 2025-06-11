## url=https://github.com/metricsearch/metric_space_rust/pkgs/container/sisap2025
echo "RUN AS: bash -x task2.sh 2>&1 | tee log-task2.txt"

PATH_TO_HOST_DIR=/home/sisap23evaluation/data2025/without-gold
PATH_TO_CONTAINER_DIR=/workspace/data
OUT_PATH_TO_HOST_DIR=$(pwd)/results-task2
OUT_PATH_TO_CONTAINER_DIR=/workspace/results

mkdir $OUT_PATH_TO_HOST_DIR
echo "==== pwd: $(pwd)"
echo "==== directory listing: "
ls
echo "==== environment"
set
echo "==== RUN BEGINS $(date)"
docker run \
    -it \
    --cpus=8 \
    --memory=16g \
    --volume $PATH_TO_HOST_DIR:$PATH_TO_CONTAINER_DIR:ro \
    --volume $OUT_PATH_TO_HOST_DIR:$OUT_PATH_TO_CONTAINER_DIR:rw \
      sisap25/dcc-uchile python root_join.py --task task2 --dataset gooaq 
    #--memory-swap=16g \


echo "==== RUN ENDS $(date)"


