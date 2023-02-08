exec docker run --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=7\
      --user=root \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
      -t \
	  test_env /bin/bash -c "cd /mnt/shared && python /mnt/shared/surrogate_model/HVAC_env_linear.py"  
exit $