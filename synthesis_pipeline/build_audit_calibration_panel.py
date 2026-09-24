"""Render labeled development-only photographic QA examples, never edit pixels."""
import argparse
import json
from pathlib import Path
import textwrap
from PIL import Image,ImageDraw,ImageFont,ImageOps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=json.loads(a.manifest.read_text())
    if len(rows)!=4:raise ValueError('Exactly four reference tiles required')
    panel=Image.new('RGB',(1024,1024),'white');draw=ImageDraw.Draw(panel)
    try:font=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError:font=ImageFont.load_default()
    for i,row in enumerate(rows):
        x=(i%2)*512;y=(i//2)*512
        im=Image.open(row['path']).convert('RGB').crop(row['crop'])
        tile=ImageOps.contain(im,(496,370))
        panel.paste(tile,(x+8+(496-tile.width)//2,y+32+(370-tile.height)//2))
        draw.text((x+8,y+6),row['label'],fill='black',font=font)
        draw.multiline_text((x+8,y+410),'\n'.join(textwrap.wrap(row['reason'],50)),fill='black',font=font,spacing=4)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    panel.save(a.output)
    print(a.output)


if __name__=='__main__':main()
