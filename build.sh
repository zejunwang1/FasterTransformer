export CUDNN_ROOT='/data/zhangqingguo/local/cudnn/cudnn-linux-x86_64-9.2.0.82_cuda12-archive'

mkdir -p build_fix_sm80
cd build_fix_sm80

cmake -DSM=80 -DCUDNN_ROOT=${CUDNN_ROOT} -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON ..
make -j12

