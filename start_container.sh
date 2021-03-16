docker run -it --name pix2meshpp --gpus all -v ${PWD}:/pix2meshpp -w /pix2meshpp --env P2MPP_DIR=/pix2meshpp --env TF_CPP_MIN_LOG_LEVEL=1 pix2meshpp:tf1.13-cuda10 bash 
