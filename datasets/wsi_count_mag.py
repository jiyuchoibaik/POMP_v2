import openslide, glob
from collections import Counter

files = glob.glob('./downloads/wsi/*/*.svs')
mags = Counter()
for f in files:
    s = openslide.OpenSlide(f)
    mag = s.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 'None')
    mags[mag] += 1
    s.close()

print(f'전체: {len(files)}개')
for mag, cnt in sorted(mags.items()):
    print(f'  mag={mag}: {cnt}개')
