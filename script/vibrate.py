import myo as libmyo; libmyo.init("/Users/chenyihan/Desktop/USC/hackathon/LAHacks2018/FeelTheWorld/MyoSDK/myo.framework")
import time
import sys

def vibrate(interval):
    for i in range(3):
        myo.vibrate("short")
        time.sleep(float(interval))

feed = libmyo.device_listener.Feed()
hub = libmyo.Hub()
hub.run(1000, feed)
try:
    myo = feed.wait_for_single_device(timeout=10.0)  # seconds
    if not myo:
        print("No Myo connected after 10 seconds.")
        sys.exit()

    # on connect
    if (myo.connected):
        myo.vibrate("short")
        myo.vibrate("short")
    while hub.running and myo.connected:
        degree = input("speed in float? ")
        vibrate(degree)
except KeyboardInterrupt:
    print("Quitting...")
finally:
    hub.shutdown()

