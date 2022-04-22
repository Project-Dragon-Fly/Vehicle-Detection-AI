### Set paths
```bash
weight=path_to_weights
cfg=path_to_config_file
data=path_to_data_file
video=path_to_mp4_file
image=path_to_single_image_file
```

#### Detect video
```bash
./darknet detector demo $data $cfg $weight $video -i 0 -thresh 0.25
```

#### Process net camera
```bash
./darknet detector demo $data $cfg $weight rtsp://admin:admin12345@192.168.0.228:554 -i 0 -thresh 0.25
```

#### Detect image
```bash
./darknet detector test $data $cfg $weight $image -i 0 -thresh 0.25
```

#### Stream feeds
> Run this script and then open URL in Chrome/Firefox in 2 tabs: http://localhost:8070 and http://localhost:8090 <br>
> Or open: http://ip-address:8070 and http://ip-address:8090 <br>
> To get <ip-address> run: sudo ifconfig

```bash
./darknet detector demo $data $cfg $weight $video -json_port 8070 -mjpeg_port 8090 -ext_output
```
