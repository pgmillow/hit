  cd /home/xudi_ge/openpi

  # 单个 run
  /home/xudi_ge/openpi/.venv/bin/tensorboard \
    --logdir /data/gxdcheckpoint/backup/gxd_pi05_standard_state_from6000_lrhalf/tb\
    --host 0.0.0.0 \
    --port 6015 \
    --reload_interval 15

  /home/xudi_ge/openpi/.venv/bin/tensorboard \
    --logdir  /data/checkpoint_629/gxd_pi05_629/gxd_pi05_629_full/tb \
    --host 0.0.0.0 \
    --port 6014 \
    --reload_interval 15

  /home/xudi_ge/openpi/.venv/bin/tensorboard \
    --logdir  /data/checkpoint_629/gxd_pi05_629/gxd_pi05_629_ema099_2gpu/tb \
    --host 0.0.0.0 \
    --port 6025 \
    --reload_interval 15


/data/gxdcheckpoint/gxd_pi05_V5_mcap0625_rgb/gxd_pi05_V5_mcap0625_rgb/tb

/data/gxdcheckpoint/train_from5k_V5/train_from5k_V5/tb
  # 合并多个 run（必须用 --logdir_spec，不能用逗号拼 --logdir）
  /home/xudi_ge/openpi/.venv/bin/tensorboard \
    --logdir_spec=from10000:/data/gxdcheckpoint/gxd_pi05_from10000_staged_lr/gxd_pi05_from10000_staged_lr/tb,from2000:/data/gxdcheckpoint/gxd_pi05_from2000_staged_lr/gxd_pi05_from2000_staged_lr/tb \
    --host 0.0.0.0 \
    --port 6011 \
    --reload_interval 15


    pour water from bottle to cup

cd /data/gxdcheckpoint/trans
tar -czvf pi05_ckpt1000_infer.tar.gz \
  -C /data/gxdcheckpoint/gxd_pi05_stateconti/gxd_pi05_standard_state_from4000_3epoch/1000 \
  params assets



    用法：

  /home/xudi_ge/openpi/.venv/bin/python /home/xudi_ge/data/check_mcap_integrity.py /data/MCAP0625

  默认会输出到：

  /data/MCAP0625/mcap_integrity_summary.txt

  也可以指定输出路径：

  /home/xudi_ge/openpi/.venv/bin/python /home/xudi_ge/data/check_mcap_integrity.py \
    /data/MCAP0625 \
    --out /home/xudi_ge/data/data_check.txt\
    --image-decode-stride 50


      /home/xudi_ge/openpi/.venv/bin/python /home/xudi_ge/data/
  check_mcap_integrity.py /data/MCAP0625 --out /home/xudi_ge/data/
  mcap_integrity_summary.txt --image-decode-stride 50



    GPUS=2,3 \
  FSDP_DEVICES=2 \
  EXP_NAME=gxd_pi05_629_ema099_2gpu \
  OVERWRITE=false \
  RESUME=false \
  bash scripts/train_pi05.sh



  • 用这个看你当前 4 卡 low_lr 这次的 TensorBoard：

  /home/xudi_ge/openpi/.venv/bin/tensorboard \
    --logdir /data/gxdcheckpoint/4_card/gxd_pi05_629_from4k_low_lr/gxd_pi05_629_from4k_low_lr_gpu0123/tb,630:from4k_low_lr/gxd_pi05_629_params4k_low_lr_gpu23/tb  \
    --host 0.0.0.0 \
    --port 6025 \
    --reload_interval 15

  如果要和原来的 629 一起对比：

  /home/xudi_ge/openpi/.venv/bin/tensorboard \
    --logdir_spec=4card:/data/gxdcheckpoint/4_card/gxd_pi05_629_from4k_low_lr/gxd_pi05_629_from4k_low_lr_gpu0123/tb,630:from4k_low_lr/gxd_pi05_629_params4k_low_lr_gpu23/tb  \
    --host 0.0.0.0 \
    --port 6026 \
    --reload_interval 15


 /data/checkpoint_629/gxd_pi05_629_from4k_low_lr/gxd_pi05_629_params4k_low_lr_gpu23/1000
  /data/gxdcheckpoint/4_card/gxd_pi05_629_from4k_low_lr/gxd_pi05_629_from4k_low_lr_gpu0123/4000


    scp -r \
    /data/gxdcheckpoint/4_card/gxd_pi05_629_from4k_low_lr/
    gxd_pi05_629_from4k_low_lr_gpu0123/5000 \
    hit1@192.168.102.152:/media/hit1/数据1/pi05_deploy_0701