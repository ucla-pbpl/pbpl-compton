#!/bin/sh

# run headless blender jobs!
#blender --background --python blenderize.py gamma-10MeV
#blender --background --python blenderize.py gamma-2GeV

convert -flatten gamma-2GeV.png A.png
mogrify -crop 3000x2000+400+30 A.png
#optipng A.png
#mv A.png 2GeV.png
convert A.png -quality 80 2GeV.jpg

convert -flatten gamma-10MeV.png A.png
mogrify -crop 3000x2000+400+30 A.png
#optipng A.png
#mv A.png 10MeV.png
convert A.png -quality 80 10MeV.jpg

#rm A.png
