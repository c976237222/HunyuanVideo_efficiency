docker run --gpus all --rm -it \
  -v /mnt/public/wangsiyuan/HunyuanVideo_efficiency:/home/anaconda/workspace \
  -e VIDIA_DRIVER_CAPABILITIES=video\
  docker-0.unsee.tech/xychelsea/ffmpeg-nvidia:latest \
  /bin/bash