exec docker run --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=5\
      --user=root \
	  --detach=false \
	  -e DISPLAY=${DISPLAY} \
	  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	  --rm \
	  -v `pwd`:/mnt/shared \
	  -i \
      -t \
	  test_env /bin/bash -c "cd /mnt/shared && python /mnt/shared/main_final_ver.py"  
exit $