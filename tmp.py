import time
from pprint import pprint

import cv2
from harvesters.core import Harvester
from genicam.gentl import TimeoutException

from frame_source.genicam_capture import GenicamCapture

import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', or 'GTK3Agg' depending on your system
import matplotlib.pyplot as plt

plt.ion()  # Interactive mode on

fig, ax = plt.subplots()
image_display = None

h = Harvester()

h.add_file('/opt/pylon/lib/gentlproducer/gtl/ProducerU3V.cti')
# h.add_file('/opt/pylon/lib/gentlproducer/gtl/ProducerGEV.cti')
# h.add_file('/usr/lib/ids/cti/ids_gevgentl.cti')
# h.add_file('/usr/lib/ids/cti/ids_u3vgentl.cti')
h.add_file('/usr/lib/ids/cti/ids_ueyegentl.cti')

h.update()

pprint(h.device_info_list)

ia = h.create({'serial_number': '4103534524'})
print(ia)

ia.start()

# Setup Matplotlib window
fig.show()

try:
    while True:
        try:
            with ia.fetch(timeout=3) as bf:
                print(bf)
                nump = GenicamCapture.to_np(bf)
                print(nump.min(), nump.max(), nump.mean())

            if image_display is None:
                # Create the image once
                image_display = ax.imshow(nump, cmap='gray', vmin=0, vmax=255)
            else:
                # Update image data
                image_display.set_data(nump)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Optional: slow down for testing (remove this for real-time)
            time.sleep(0.01)

        except TimeoutException as e:
            print("Timeout occurred:", e)
            continue

except KeyboardInterrupt:
    print("Exiting...")

finally:
    ia.stop()
    ia.destroy()
    h.reset()
