  cd /home/xudi_ge/openpi

  /home/xudi_ge/openpi/.venv/bin/tensorboard \
    --logdir /data/gxdcheckpoint/gxd_pi05_stateconti/gxd_pi05_standard_state_from4000_3epoch/tb
    --host 0.0.0.0 \
    --port 6010 \
    --reload_interval 15


    pour water from bottle to cup

cd /data/gxdcheckpoint/trans
tar -czvf pi05_ckpt1000_infer.tar.gz \
  -C /data/gxdcheckpoint/gxd_pi05_stateconti/gxd_pi05_standard_state_from4000_3epoch/1000 \
  params assets